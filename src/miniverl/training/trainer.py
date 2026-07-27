"""The miniVERL trainer.

One class runs all three modes, because they differ only in *where the
trajectories come from* and *where the targets come from*:

======================  ==========================  ==========================
mode                    trajectories                targets
======================  ==========================  ==========================
``sft``                 environment oracle traces   the tokens themselves (CE)
``offline_kd``          a fixed trajectory set      one frozen teacher cache
``opd``                 sampled from the *current*  the teacher scoring those
                        student every cycle         exact states, every cycle
======================  ==========================  ==========================

The distinction is enforced, not documented: ``opd`` requires
``cache.strict_policy_version``, so a target produced under policy version *v*
raises :class:`~miniverl.errors.StaleCacheError` if anything tries to consume it
at version *v+1*.  ``offline_kd`` is the only mode that may set
``cache.reuse_across_policy_versions``.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from miniverl import __version__
from miniverl.agent.loop import RolloutRunner, RolloutStats
from miniverl.cache.store import TeacherCache
from miniverl.config.models import (
    LossMode,
    MemoryStrategy,
    ModelBackend,
    RunConfig,
    TeacherContextMode,
    TrainingMode,
)
from miniverl.environments.base import Task, ToolEnvironment, make_splits
from miniverl.environments.registry import make_environment
from miniverl.errors import ConfigError, MiniVerlError
from miniverl.models.factory import build_student, build_teacher, build_tokenizer, resolve_device
from miniverl.schemas.alignment import AlignmentMap
from miniverl.schemas.trajectory import Trajectory
from miniverl.selection.selectors import SelectionResult, aggregate_selection_stats, select_positions
from miniverl.teachers.local import LocalTeacherScorer
from miniverl.trajectory.alignment import build_alignment_map
from miniverl.trajectory.io import append_trajectories
from miniverl.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from miniverl.training.memory import (
    MemoryPlan,
    move_optimizer_state,
    resolve_strategy,
    run_with_oom_retry,
)
from miniverl.training.optim import LearningRateSchedule, build_optimizer
from miniverl.utils import gpu
from miniverl.utils.env import collect_environment
from miniverl.utils.logging import EventLog, get_logger
from miniverl.utils.runs import JsonlWriter, RunPaths, make_run_id, utc_now, write_json
from miniverl.utils.seeding import capture_rng, seed_everything

if TYPE_CHECKING:  # pragma: no cover - typing only
    from miniverl.teachers.base import TeacherScoreResult

__all__ = ["OPDTrainer", "TrainResult", "TrainSample"]

logger = get_logger("trainer")


@dataclass
class TrainSample:
    """One trajectory plus everything needed to compute its loss."""

    trajectory: Trajectory
    alignment: AlignmentMap
    selection: SelectionResult
    teacher: TeacherScoreResult | None = None


@dataclass
class TrainResult:
    """Summary returned by :meth:`OPDTrainer.train`."""

    run_id: str
    run_dir: Path
    mode: str
    cycles_completed: int
    global_step: int
    policy_version: int
    duration_seconds: float
    final_metrics: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] | None = None
    baseline_eval: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "mode": self.mode,
            "cycles_completed": self.cycles_completed,
            "global_step": self.global_step,
            "policy_version": self.policy_version,
            "duration_seconds": round(self.duration_seconds, 3),
            "final_metrics": self.final_metrics,
            "eval": self.eval,
            "baseline_eval": self.baseline_eval,
        }


class OPDTrainer:
    """Runs SFT, offline KD or genuine on-policy distillation."""

    def __init__(
        self,
        *,
        config: RunConfig,
        paths: RunPaths,
        run_id: str,
        environment: ToolEnvironment,
        splits: dict[str, list[Task]],
        tokenizer: Any,
        student: Any,
        teacher: Any | None,
        plan: MemoryPlan,
    ) -> None:
        self.config = config
        self.paths = paths
        self.run_id = run_id
        self.environment = environment
        self.splits = splits
        self.tokenizer = tokenizer
        self.student = student
        self.teacher = teacher
        self.plan = plan

        self.metrics_log = JsonlWriter(paths.metrics)
        self.events = EventLog(JsonlWriter(paths.events))
        self.runner = RolloutRunner(
            backend=student, environment=environment, config=config.rollout
        )
        self.scorer = (
            LocalTeacherScorer(
                teacher,
                config.loss,
                keep_exact_resident=plan.strategy is MemoryStrategy.RESIDENT,
                device=plan.device,
            )
            if teacher is not None
            else None
        )
        self.optimizer = build_optimizer(student.trainable_parameters(), config.train)
        steps_per_cycle = self.optimizer_steps_per_cycle
        total_steps = max(
            1,
            steps_per_cycle * (config.train.cycles + config.train.sft_warmup_cycles),
        )
        self.schedule = LearningRateSchedule(
            kind=config.train.lr_schedule,
            base_lr=config.train.learning_rate,
            warmup_steps=config.train.warmup_steps,
            total_steps=total_steps,
        )
        self.global_step = 0
        self.policy_version = 0
        self.cycle = 0
        self.task_cursor = 0
        self._task_order = self._build_task_order()
        self._cache: TeacherCache | None = None
        self._offline_samples: list[TrainSample] | None = None
        self._teacher_on_device = teacher is not None and plan.strategy is MemoryStrategy.RESIDENT

    # -- construction --------------------------------------------------------

    @property
    def optimizer_steps_per_cycle(self) -> int:
        """Optimizer steps performed per training cycle."""
        accum = self.config.train.gradient_accumulation_steps
        rollouts = self.config.train.rollouts_per_cycle
        return max(1, (rollouts + accum - 1) // accum)

    @classmethod
    def from_config(
        cls,
        config: RunConfig,
        *,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
        local_files_only: bool = False,
        write_artifacts: bool = True,
    ) -> OPDTrainer:
        """Validate, seed, load models and create the run directory.

        ``write_artifacts=False`` attaches to an existing run directory without
        rewriting ``config.original.yaml`` or ``manifest.json``.  The standalone
        evaluator uses it so re-evaluating a run cannot destroy the provenance of
        the run being evaluated.
        """
        seed_everything(config.run.seed, deterministic=config.run.deterministic)
        resolved_id = make_run_id(config.run.name, explicit=run_id or config.run.run_id)
        paths = RunPaths.create(output_dir or config.run.output_dir, resolved_id)
        if write_artifacts:
            paths.config_original.write_text(config.to_yaml(), encoding="utf-8")

        environment = make_environment(config.environment.name, **config.environment.params)
        splits = make_splits(
            environment,
            counts={
                "train": config.environment.train_tasks,
                "eval": config.environment.eval_tasks,
                "test": config.environment.test_tasks,
            },
            seed=config.environment.split_seed,
            difficulty=config.environment.difficulty,
        )
        if config.models.teacher.mode is TeacherContextMode.PRIVILEGED_CONTEXT:
            probe = environment.privileged_context(splits["train"][0])
            if not probe:
                raise ConfigError(
                    f"environment {environment.name!r} provides no privileged context, but "
                    "models.teacher.mode is 'privileged_context'",
                    hint="set models.teacher.mode: standard, or use an environment that "
                    "implements privileged_context()",
                )

        device = resolve_device(config.models)
        tokenizer = build_tokenizer(config, local_files_only=local_files_only)
        student = build_student(config, tokenizer, device=device, local_files_only=local_files_only)

        teacher = None
        plan = resolve_strategy(
            config.memory, device=device, chunk_size=config.loss.chunk_size
        )
        if config.run.mode is not TrainingMode.SFT:
            if config.memory.strategy is MemoryStrategy.AUTO and device.startswith("cuda"):
                loaded: list[Any] = []

                def teacher_fits() -> bool:
                    try:
                        loaded.append(
                            build_teacher(
                                config,
                                tokenizer,
                                device=device,
                                local_files_only=local_files_only,
                            )
                        )
                        return True
                    except (RuntimeError, MemoryError) as exc:
                        if not gpu.is_oom_error(exc):
                            raise
                        gpu.empty_cache()
                        return False

                plan = resolve_strategy(
                    config.memory,
                    device=device,
                    chunk_size=config.loss.chunk_size,
                    teacher_fits=teacher_fits,
                )
                teacher = loaded[0] if loaded else None
            if teacher is None:
                teacher_device = (
                    device if plan.strategy is MemoryStrategy.RESIDENT else "cpu"
                )
                teacher = build_teacher(
                    config,
                    tokenizer,
                    device=teacher_device,
                    local_files_only=local_files_only,
                )

        trainer = cls(
            config=config,
            paths=paths,
            run_id=resolved_id,
            environment=environment,
            splits=splits,
            tokenizer=tokenizer,
            student=student,
            teacher=teacher,
            plan=plan,
        )
        if write_artifacts:
            trainer._write_startup_artifacts()
        return trainer

    def _write_startup_artifacts(self) -> None:
        config = self.config
        resolved = config.model_copy(deep=True)
        resolved.run.run_id = self.run_id
        resolved.memory.strategy = self.plan.strategy
        resolved.loss.chunk_size = self.plan.chunk_size
        resolved.models.device = self.plan.device
        if config.models.backend is ModelBackend.HF:
            resolved.loss.top_k = min(config.loss.top_k, self.student.vocab_size)
        self.paths.config_resolved.write_text(resolved.to_yaml(), encoding="utf-8")
        write_json(self.paths.environment, collect_environment())
        write_json(self.paths.manifest, self.build_manifest())

    def build_manifest(self) -> dict[str, Any]:
        """Full provenance record for the run."""
        env_info = collect_environment()
        config = self.config
        return {
            "miniverl_version": __version__,
            "run_id": self.run_id,
            "run_name": config.run.name,
            "created_at": utc_now(),
            "git_commit": env_info["git_commit"],
            "python_version": env_info["python_version"],
            "os": env_info["os"],
            "os_release": env_info["os_release"],
            "platform": env_info["platform"],
            "packages": env_info["packages"],
            "gpu": env_info["gpu"],
            "mode": config.run.mode.value,
            "seed": config.run.seed,
            "deterministic": config.run.deterministic,
            "environment": {
                **self.environment.describe(),
                "difficulty": config.environment.difficulty,
                "split_seed": config.environment.split_seed,
                "split_sizes": {k: len(v) for k, v in self.splits.items()},
            },
            "models": {
                "backend": config.models.backend.value,
                "device": self.plan.device,
                "student": {
                    "model_id": config.models.student.model_id,
                    "revision": config.models.student.revision,
                    "tokenizer_revision": config.models.student.tokenizer_revision,
                    "quantization": config.models.student.quantization.value,
                    "precision": config.models.student.dtype.value,
                    "capabilities": self.student.capabilities.to_dict(),
                },
                "teacher": (
                    {
                        "model_id": config.models.teacher.model_id,
                        "revision": config.models.teacher.revision,
                        "tokenizer_revision": config.models.teacher.tokenizer_revision,
                        "quantization": config.models.teacher.quantization.value,
                        "precision": config.models.teacher.dtype.value,
                        "context_mode": config.models.teacher.mode.value,
                        "capabilities": self.teacher.capabilities.to_dict(),
                    }
                    if self.teacher is not None
                    else None
                ),
                "tokenizer_fingerprint": self.tokenizer.fingerprint,
                "tokenizer_vocab_size": self.tokenizer.vocab_size,
            },
            "objective": {
                "loss_mode": config.loss.mode.value,
                "divergence": config.loss.divergence.value,
                "temperature": config.loss.temperature,
                "scale_by_temperature_squared": config.loss.scale_by_temperature_squared,
                "top_k": (
                    self.student.vocab_size
                    if config.loss.mode is LossMode.EXACT_FULL_VOCAB
                    else min(config.loss.top_k, self.student.vocab_size)
                ),
                "jsd_beta": config.loss.jsd_beta,
                "ce_weight": config.loss.ce_weight,
                "selector": config.selection.selector.value,
                "selection_ratio": config.selection.ratio,
            },
            "memory": self.plan.to_dict(),
            "policy_version": self.policy_version,
            "measurement_status": {
                "cpu_metrics": "measured",
                "cuda_metrics": "measured" if gpu.cuda_available() else "not_run_no_cuda",
                "simulated_results": "none",
            },
        }

    # -- task sampling --------------------------------------------------------

    def _build_task_order(self) -> list[int]:
        order = list(range(len(self.splits["train"])))
        random.Random(self.config.run.seed ^ 0x5EED).shuffle(order)
        return order

    def _next_tasks(self, count: int) -> list[Task]:
        train = self.splits["train"]
        out: list[Task] = []
        for _ in range(count):
            index = self._task_order[self.task_cursor % len(self._task_order)]
            out.append(train[index])
            self.task_cursor += 1
        return out

    # -- teacher preparation ---------------------------------------------------

    def _prepare_toy_teacher(self) -> None:
        """Fit the toy teacher on oracle traces so it is a meaningful oracle."""
        config = self.config
        if (
            config.models.backend is not ModelBackend.TOY
            or self.teacher is None
            or config.models.teacher.toy_pretrain_steps <= 0
        ):
            return
        from miniverl.models.toy import ToyBackend, fit_toy_model

        assert isinstance(self.teacher, ToyBackend)
        teacher_runner = RolloutRunner(
            backend=self.teacher, environment=self.environment, config=config.rollout
        )
        # Fit on the whole training split, not a handful of traces: measured on
        # the calculator environment, 48 oracle traces memorize (train loss
        # 0.0006) and generalize at 25%, while 256 traces reach 87.5% -- the
        # copy behaviour the tool protocol needs only emerges with diversity.
        batches = []
        for task in self.splits["train"]:
            traj = teacher_runner.oracle_rollout(task)
            targets = [j for j, flag in enumerate(traj.model_generated_mask) if flag and j > 0]
            if targets:
                batches.append((traj.token_ids, targets))
        started = time.perf_counter()
        losses = fit_toy_model(
            self.teacher,
            batches,
            steps=config.models.teacher.toy_pretrain_steps,
            lr=config.models.teacher.toy_pretrain_lr,
            seed=config.models.teacher.toy_teacher_seed,
            chunk_size=self.plan.chunk_size,
            batch_size=config.train.gradient_accumulation_steps,
        )
        for param in self.teacher.model.parameters():
            param.requires_grad_(False)
        self.teacher.set_train(False)
        self.events.emit(
            "toy_teacher_fitted",
            steps=len(losses),
            traces=len(batches),
            first_loss=round(losses[0], 4) if losses else None,
            last_loss=round(losses[-1], 4) if losses else None,
            seconds=round(time.perf_counter() - started, 2),
        )

    # -- rollout + scoring ------------------------------------------------------

    def _collect(self, tasks: list[Task], *, oracle: bool) -> tuple[list[Trajectory], RolloutStats]:
        stats = RolloutStats()
        trajectories: list[Trajectory] = []
        for offset, task in enumerate(tasks):
            if oracle:
                traj = self.runner.oracle_rollout(
                    task,
                    policy_version=self.policy_version,
                    trajectory_id=f"{task.task_id}:oracle:c{self.cycle}",
                )
            else:
                traj = self.runner.rollout(
                    task,
                    policy_version=self.policy_version,
                    seed=self.config.run.seed + self.global_step * 1013 + offset,
                )
            trajectories.append(traj)
            stats.observe(traj)
        append_trajectories(self.paths.trajectories, trajectories)
        return trajectories, stats

    def _open_cache(self) -> TeacherCache:
        if self._cache is None:
            assert self.teacher is not None
            top_k = (
                self.student.vocab_size
                if self.config.loss.mode is LossMode.EXACT_FULL_VOCAB
                else min(self.config.loss.top_k, self.student.vocab_size)
            )
            self._cache = TeacherCache.create(
                self.config.cache.dir or self.paths.teacher_cache,
                miniverl_version=__version__,
                teacher_model_id=self.config.models.teacher.model_id,
                teacher_model_revision=self.config.models.teacher.revision,
                tokenizer_fingerprint=self.tokenizer.fingerprint,
                vocab_size=self.student.vocab_size,
                top_k=top_k,
                temperature=self.config.loss.temperature,
                loss_mode=self.config.loss.mode.value,
                dtype=self.config.cache.dtype,
                entries_per_shard=self.config.cache.entries_per_shard,
            )
        return self._cache

    def _build_samples(self, trajectories: list[Trajectory]) -> list[TrainSample]:
        """Select positions, align, and (for KD modes) score with the teacher."""
        config = self.config
        samples: list[TrainSample] = []
        privileged = config.models.teacher.mode is TeacherContextMode.PRIVILEGED_CONTEXT
        task_by_id = {t.task_id: t for split in self.splits.values() for t in split}

        for traj in trajectories:
            selection = select_positions(traj, config.selection, run_seed=config.run.seed)
            if not selection.positions:
                continue
            teacher_view: Trajectory | None = None
            if privileged and self.scorer is not None:
                task = task_by_id.get(traj.task_id)
                if task is not None:
                    teacher_view = self.runner.privileged_render(traj, task)
            alignment = build_alignment_map(
                traj, selection.positions, selection.weights, teacher=teacher_view
            )
            samples.append(
                TrainSample(trajectory=traj, alignment=alignment, selection=selection)
            )

        if self.scorer is None or config.run.mode is TrainingMode.SFT:
            return samples

        cache = self._open_cache()
        for sample in samples:
            teacher_view = None
            if privileged:
                task = task_by_id.get(sample.trajectory.task_id)
                if task is not None:
                    teacher_view = self.runner.privileged_render(sample.trajectory, task)
            result = self.scorer.score(
                student=sample.trajectory,
                alignment=sample.alignment,
                teacher_view=teacher_view,
            )
            sample.teacher = result
            if result.cacheable is not None:
                cache.write(
                    result.cacheable,
                    selector=config.selection.selector.value,
                    tail_is_exact_zero=config.loss.mode is LossMode.EXACT_FULL_VOCAB,
                )
        cache.flush()
        return samples

    # -- optimization --------------------------------------------------------------

    def _loss_by_span_type(self, sample: TrainSample, per_token: Any) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        weights = sample.alignment.token_weights
        for value, weight, span in zip(
            per_token.tolist(), weights, sample.alignment.span_types
        ):
            entry = totals.setdefault(span, [0.0, 0.0])
            entry[0] += value * weight
            entry[1] += weight
        return {k: (v[0] / v[1] if v[1] else 0.0) for k, v in totals.items()}

    def _run_group(self, group: list[TrainSample], chunk_size: int) -> dict[str, Any]:
        from miniverl.losses.chunked import chunked_selected_position_loss

        import torch

        config = self.config
        self.optimizer.zero_grad(set_to_none=True)
        self.student.set_train(True)
        device = self.student.device
        loss_total = 0.0
        positions_total = 0
        entropy_sum = 0.0
        entropy_count = 0
        span_losses: dict[str, list[float]] = {}
        scale = 1.0 / max(len(group), 1)

        for sample in group:
            alignment = sample.alignment
            hidden = self.student.hidden_states_at(
                sample.trajectory.token_ids,
                alignment.student_prediction_positions,
                with_grad=True,
            )
            weights = torch.tensor(alignment.token_weights, dtype=torch.float32, device=device)
            targets = torch.tensor(alignment.target_token_ids, dtype=torch.long, device=device)
            provider = sample.teacher.provider if sample.teacher is not None else None
            ce_weight = 1.0 if provider is None else config.loss.ce_weight
            output = chunked_selected_position_loss(
                hidden_states=hidden,
                lm_head=self.student.project,
                weights=weights,
                provider=provider,
                target_token_ids=targets,
                ce_weight=ce_weight,
                chunk_size=chunk_size,
                backward=True,
                loss_scale=scale,
            )
            loss_total += output.loss * scale
            positions_total += output.num_positions
            for name, value in self._loss_by_span_type(sample, output.per_token).items():
                span_losses.setdefault(name, []).append(value)
            if sample.teacher is not None and sample.teacher.teacher_entropy.numel():
                entropy_sum += float(sample.teacher.teacher_entropy.sum())
                entropy_count += int(sample.teacher.teacher_entropy.numel())
            del hidden, output

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.student.trainable_parameters(), config.train.max_grad_norm
            )
        )
        lr = self.schedule.lr_at(self.global_step)
        for group_params in self.optimizer.param_groups:
            group_params["lr"] = lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": loss_total,
            "grad_norm": grad_norm,
            "lr": lr,
            "selected_positions": positions_total,
            "trajectories_in_step": len(group),
            "teacher_entropy_mean": (entropy_sum / entropy_count) if entropy_count else None,
            "loss_by_span_type": {
                k: sum(v) / len(v) for k, v in sorted(span_losses.items()) if v
            },
        }

    def _optimize(self, samples: list[TrainSample], *, phase: str) -> list[dict[str, Any]]:
        accum = self.config.train.gradient_accumulation_steps
        records: list[dict[str, Any]] = []
        for start in range(0, len(samples), accum):
            group = samples[start : start + accum]
            if not group:
                continue
            if self.config.memory.reset_peak_stats_each_cycle:
                gpu.reset_peak_stats()
            started = time.perf_counter()
            record = run_with_oom_retry(
                lambda chunk, g=group: self._run_group(g, chunk),
                plan=self.plan,
                memory=self.config.memory,
                on_retry=lambda old, new: self.events.emit(
                    "oom_chunk_retry", old_chunk=old, new_chunk=new, note="objective unchanged"
                ),
                cleanup=lambda: self.optimizer.zero_grad(set_to_none=True),
            )
            elapsed = max(time.perf_counter() - started, 1e-9)
            record.update(
                {
                    "phase": phase,
                    "cycle": self.cycle,
                    "step": self.global_step,
                    "policy_version": self.policy_version,
                    "seconds": round(elapsed, 4),
                    "train_selected_tokens_per_second": record["selected_positions"] / elapsed,
                    "projection_chunk_size": self.plan.chunk_size,
                    "memory": gpu.snapshot().to_dict(),
                    "ts": utc_now(),
                }
            )
            self.metrics_log.write(record)
            records.append(record)
            self.global_step += 1
            if self.global_step % self.config.train.log_every_steps == 0:
                logger.info(
                    "%s step %d cycle %d loss %.4f lr %.2e positions %d",
                    phase,
                    self.global_step,
                    self.cycle,
                    record["loss"],
                    record["lr"],
                    record["selected_positions"],
                )
        return records

    # -- memory strategy ---------------------------------------------------------

    def _teacher_to_device(self) -> None:
        if self.teacher is None or self._teacher_on_device:
            return
        self.teacher.to_device(self.plan.device)
        self._teacher_on_device = True

    def _teacher_off_device(self) -> None:
        if self.teacher is None or not self._teacher_on_device:
            return
        self.teacher.release()
        self._teacher_on_device = False

    def _student_off_device(self) -> dict[str, Any]:
        state = self.student.trainable_state_dict()
        move_optimizer_state(self.optimizer, "cpu")
        self.student.release()
        if self.config.memory.empty_cache_between_phases:
            gpu.empty_cache()
        return state

    def _student_on_device(self) -> None:
        self.student.to_device(self.plan.device)
        move_optimizer_state(self.optimizer, self.plan.device)

    # -- public API -----------------------------------------------------------------

    def train(self) -> TrainResult:
        """Run the configured schedule and return a summary."""
        config = self.config
        started = time.perf_counter()
        self.events.emit(
            "run_start",
            run_id=self.run_id,
            mode=config.run.mode.value,
            device=self.plan.device,
            strategy=self.plan.strategy.value,
            loss_mode=config.loss.mode.value,
            divergence=config.loss.divergence.value,
            cycles=config.train.cycles,
        )
        if config.run.mode is TrainingMode.OPD and self.optimizer_steps_per_cycle > 1:
            self.events.emit(
                "opd_multi_update_warning",
                steps_per_cycle=self.optimizer_steps_per_cycle,
                note=(
                    "more than one optimizer step per rollout batch: steps after the first "
                    "are only approximately on-policy. Set "
                    "train.gradient_accumulation_steps = train.rollouts_per_cycle for "
                    "strict on-policy updates."
                ),
            )

        self._prepare_toy_teacher()

        baseline = None
        if config.eval.enabled:
            baseline = self.evaluate(tag="baseline")

        if config.train.sft_warmup_cycles > 0 and config.run.mode is not TrainingMode.SFT:
            self._run_sft_warmup(config.train.sft_warmup_cycles)

        last_records: list[dict[str, Any]] = []
        for cycle in range(config.train.cycles):
            self.cycle = cycle
            last_records = self._run_cycle()
            if (
                config.train.eval_every_cycles
                and (cycle + 1) % config.train.eval_every_cycles == 0
                and cycle + 1 < config.train.cycles
            ):
                self.evaluate(tag=f"cycle{cycle + 1}")
            if (
                config.train.save_every_cycles
                and (cycle + 1) % config.train.save_every_cycles == 0
            ):
                self.save_checkpoint()

        final_eval = self.evaluate(tag="final") if config.eval.enabled else None
        self.save_checkpoint(name="final")
        duration = time.perf_counter() - started

        result = TrainResult(
            run_id=self.run_id,
            run_dir=self.paths.root,
            mode=config.run.mode.value,
            cycles_completed=config.train.cycles,
            global_step=self.global_step,
            policy_version=self.policy_version,
            duration_seconds=duration,
            final_metrics=last_records[-1] if last_records else {},
            eval=final_eval,
            baseline_eval=baseline,
        )
        summary = result.to_dict()
        if self._cache is not None:
            summary["cache"] = self._cache.stats().model_dump(mode="json")
        write_json(self.paths.eval_json, summary)
        self.events.emit(
            "run_end",
            run_id=self.run_id,
            steps=self.global_step,
            policy_version=self.policy_version,
            seconds=round(duration, 2),
            success_rate=(final_eval or {}).get("success_rate"),
        )
        return result

    def _run_sft_warmup(self, cycles: int) -> None:
        self.events.emit("sft_warmup_start", cycles=cycles)
        for index in range(cycles):
            self.cycle = -(cycles - index)
            tasks = self._next_tasks(self.config.train.rollouts_per_cycle)
            trajectories, stats = self._collect(tasks, oracle=True)
            samples = self._build_samples_ce_only(trajectories)
            self._optimize(samples, phase="sft_warmup")
            self.events.emit(
                "sft_warmup_cycle", cycle=index, trajectories=stats.rollouts
            )
        self.policy_version += 1
        self.cycle = 0

    def _build_samples_ce_only(self, trajectories: list[Trajectory]) -> list[TrainSample]:
        samples: list[TrainSample] = []
        for traj in trajectories:
            selection = select_positions(
                traj, self.config.selection, run_seed=self.config.run.seed
            )
            if not selection.positions:
                continue
            alignment = build_alignment_map(traj, selection.positions, selection.weights)
            samples.append(TrainSample(trajectory=traj, alignment=alignment, selection=selection))
        return samples

    def _run_cycle(self) -> list[dict[str, Any]]:
        config = self.config
        mode = config.run.mode
        cycle_started = time.perf_counter()

        if mode is TrainingMode.OFFLINE_KD and self._offline_samples is not None:
            samples = self._offline_samples
            stats = RolloutStats()
            self.events.emit(
                "offline_kd_reuse",
                cycle=self.cycle,
                trajectories=len(samples),
                note="fixed teacher targets reused; this run is explicitly not on-policy",
            )
        else:
            tasks = self._next_tasks(config.train.rollouts_per_cycle)
            oracle = mode in (TrainingMode.SFT, TrainingMode.OFFLINE_KD)
            rollout_started = time.perf_counter()
            trajectories, stats = self._collect(tasks, oracle=oracle)
            rollout_seconds = max(time.perf_counter() - rollout_started, 1e-9)

            if mode is TrainingMode.SFT or self.teacher is None:
                samples = self._build_samples_ce_only(trajectories)
            else:
                swap = self.plan.strategy is MemoryStrategy.SWAP
                student_state = None
                if swap:
                    student_state = self._student_off_device()
                    self._teacher_to_device()
                try:
                    samples = self._build_samples(trajectories)
                finally:
                    if swap:
                        self._teacher_off_device()
                        self._student_on_device()
                        if student_state is not None:
                            self.student.load_trainable_state_dict(student_state)
                if swap:
                    samples = self._reload_targets_from_cache(samples)

            self.events.emit(
                "rollouts_collected",
                cycle=self.cycle,
                policy_version=self.policy_version,
                trajectories=stats.rollouts,
                success_rate=round(stats.to_dict()["success_rate"], 4),
                generated_tokens=stats.generated_tokens,
                rollout_tokens_per_second=round(stats.generated_tokens / rollout_seconds, 2),
            )
            if mode is TrainingMode.OFFLINE_KD:
                self._offline_samples = samples

        records = self._optimize(samples, phase=config.run.mode.value)
        if config.report.enabled and self.cycle == config.train.cycles - 1:
            tokens = self._write_token_analysis(samples)
            if tokens:
                self.events.emit("token_analysis_written", tokens=tokens)
        selection_stats = aggregate_selection_stats([s.selection.stats for s in samples])
        cycle_metrics: dict[str, Any] = {
            "phase": f"{config.run.mode.value}_cycle",
            "cycle": self.cycle,
            "step": self.global_step,
            "policy_version": self.policy_version,
            "seconds": round(time.perf_counter() - cycle_started, 3),
            "rollouts": stats.to_dict(),
            "selection": selection_stats,
            "memory": gpu.snapshot().to_dict(),
            "ts": utc_now(),
        }
        if self._cache is not None:
            cycle_metrics["cache"] = self._cache.stats().model_dump(mode="json")
        self.metrics_log.write(cycle_metrics)

        if mode is TrainingMode.OPD:
            self.policy_version += 1
            if self._cache is not None and config.cache.keep_cycles:
                removed = self._cache.prune_before(
                    self.policy_version - config.cache.keep_cycles
                )
                if removed:
                    self.events.emit("cache_pruned", removed=removed)
        elif mode is TrainingMode.SFT:
            self.policy_version += 1
        return records

    def _write_token_analysis(self, samples: list[TrainSample]) -> int:
        """Persist per-token divergence data for the report's token view.

        Only tokens that are already in the stored trajectory are described --
        the report never invents or reveals anything the policy did not emit.
        """
        import torch

        from miniverl.utils.runs import JsonlWriter

        limit = self.config.report.max_trajectories
        if limit <= 0 or not samples:
            return 0
        writer = JsonlWriter(self.paths.root / "token_analysis.jsonl")
        piece_of = getattr(self.tokenizer, "token_piece", None)
        written = 0
        self.student.set_train(False)
        with torch.no_grad():
            for sample in samples[:limit]:
                alignment = sample.alignment
                traj = sample.trajectory
                hidden = self.student.hidden_states_at(
                    traj.token_ids, alignment.student_prediction_positions, with_grad=False
                )
                per_token: list[float] = []
                student_top: list[int] = []
                for start in range(0, hidden.shape[0], self.plan.chunk_size):
                    end = min(start + self.plan.chunk_size, hidden.shape[0])
                    logits = self.student.project(hidden[start:end]).to(torch.float32)
                    student_top.extend(logits.argmax(dim=-1).tolist())
                    if sample.teacher is not None:
                        per_token.extend(
                            sample.teacher.provider.divergence(start, end, logits).tolist()
                        )
                    else:
                        log_probs = torch.log_softmax(logits, dim=-1)
                        targets = torch.tensor(
                            alignment.target_token_ids[start:end],
                            dtype=torch.long,
                            device=logits.device,
                        )
                        per_token.extend(
                            (-log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).tolist()
                        )
                    del logits
                entropy = (
                    sample.teacher.teacher_entropy.tolist()
                    if sample.teacher is not None
                    else [None] * len(per_token)
                )
                teacher_top: list[int | None] = [None] * len(per_token)
                if sample.teacher is not None and sample.teacher.cacheable is not None:
                    teacher_top = sample.teacher.cacheable.topk_indices[:, 0].tolist()
                records = []
                max_tokens = self.config.report.max_tokens_per_trajectory
                for i, target_position in enumerate(alignment.student_prediction_positions):
                    if max_tokens and i >= max_tokens:
                        break
                    token_id = alignment.target_token_ids[i]
                    records.append(
                        {
                            "trajectory_id": traj.trajectory_id,
                            "target_position": target_position + 1,
                            "prediction_position": target_position,
                            "token_id": token_id,
                            "token_piece": piece_of(token_id) if callable(piece_of) else None,
                            "span_type": alignment.span_types[i],
                            "selected": True,
                            "weight": alignment.token_weights[i],
                            "token_loss": per_token[i] if i < len(per_token) else None,
                            "teacher_entropy": entropy[i] if i < len(entropy) else None,
                            "teacher_top_token": teacher_top[i] if i < len(teacher_top) else None,
                            "student_top_token": student_top[i] if i < len(student_top) else None,
                        }
                    )
                for record in records:
                    writer.write(record)
                written += len(records)
                del hidden
        self.student.set_train(True)
        return written

    def _reload_targets_from_cache(self, samples: list[TrainSample]) -> list[TrainSample]:
        """Re-attach providers from the on-disk cache after the teacher is gone."""
        from miniverl.losses.chunked import BucketedTargetProvider

        cache = self._open_cache()
        config = self.config
        for sample in samples:
            if sample.teacher is None:
                continue
            batch = cache.read(
                sample.trajectory.trajectory_id,
                expect_policy_version=(
                    sample.trajectory.policy_version
                    if config.cache.strict_policy_version
                    else None
                ),
                device=self.plan.device,
            )
            sample.teacher.provider = BucketedTargetProvider(
                topk_indices=batch.topk_indices,
                topk_log_probs=batch.topk_log_probs,
                tail_log_prob=batch.tail_log_prob,
                divergence_name=config.loss.divergence.value,
                temperature=config.loss.temperature,
                scale_by_temperature_squared=config.loss.scale_by_temperature_squared,
                jsd_beta=config.loss.jsd_beta,
                tail_epsilon=config.loss.tail_epsilon,
            )
        return samples

    def evaluate(
        self,
        *,
        split: str | None = None,
        tasks: list[Task] | None = None,
        tag: str = "eval",
        write: bool = True,
    ) -> dict[str, Any]:
        """Deterministic greedy evaluation on a held-out split."""
        config = self.config
        chosen_split = split or config.eval.split
        pool = tasks if tasks is not None else self.splits.get(chosen_split, [])
        limit = config.effective_eval_tasks if tasks is None else len(pool)
        pool = pool[:limit]
        if not pool:
            return {"tag": tag, "tasks": 0, "success_rate": 0.0, "note": "no eval tasks"}

        self.student.set_train(False)
        gpu.reset_peak_stats()
        started = time.perf_counter()
        stats = RolloutStats()
        trajectories: list[Trajectory] = []
        by_difficulty: dict[str, list[int]] = {}
        for offset, task in enumerate(pool):
            traj = self.runner.rollout(
                task,
                policy_version=self.policy_version,
                seed=config.eval.seed + offset,
                temperature=config.eval.temperature,
                max_turns=config.eval.max_turns,
                trajectory_id=f"{task.task_id}:{tag}:v{self.policy_version}",
            )
            trajectories.append(traj)
            stats.observe(traj)
            solved = int(bool(traj.verification and traj.verification.solved))
            by_difficulty.setdefault(task.difficulty, []).append(solved)
        elapsed = max(time.perf_counter() - started, 1e-9)
        append_trajectories(self.paths.eval_trajectories, trajectories)
        self.student.set_train(True)

        payload = {
            "tag": tag,
            "split": chosen_split,
            "tasks": len(pool),
            "policy_version": self.policy_version,
            "global_step": self.global_step,
            "temperature": config.eval.temperature,
            "seconds": round(elapsed, 3),
            "rollout_tokens_per_second": round(stats.generated_tokens / elapsed, 2),
            "success_by_difficulty": {
                k: sum(v) / len(v) for k, v in sorted(by_difficulty.items())
            },
            "memory": gpu.snapshot().to_dict(),
            **stats.to_dict(),
        }
        if write:
            self.metrics_log.write({"phase": "eval", **payload, "ts": utc_now()})
        self.events.emit(
            "eval",
            tag=tag,
            tasks=len(pool),
            success_rate=round(payload["success_rate"], 4),
            avg_turns=round(payload["avg_turns"], 2),
        )
        return payload

    # -- checkpointing -----------------------------------------------------------

    def _config_digest(self) -> str:
        return hashlib.sha256(self.config.to_yaml().encode("utf-8")).hexdigest()

    def save_checkpoint(self, *, name: str | None = None) -> Path:
        """Write a resumable checkpoint."""
        label = name or f"step-{self.global_step:06d}"
        target = self.paths.checkpoints / label
        state = CheckpointState(
            miniverl_version=__version__,
            global_step=self.global_step,
            policy_version=self.policy_version,
            cycle=self.cycle,
            task_cursor=self.task_cursor,
            scheduler=self.schedule.state_dict(),
            config_digest=self._config_digest(),
        )
        save_checkpoint(
            target,
            trainable_state=self.student.trainable_state_dict(),
            optimizer=self.optimizer,
            state=state,
            rng=capture_rng(),
        )
        self.events.emit("checkpoint_saved", path=str(target), step=self.global_step)
        return target

    def load_from_checkpoint(self, directory: str | Path) -> CheckpointState:
        """Restore a checkpoint into this trainer."""
        state = load_checkpoint(
            directory,
            backend=self.student,
            optimizer=self.optimizer,
            device=self.student.device,
        )
        digest = self._config_digest()
        if state.config_digest and state.config_digest != digest:
            raise ConfigError(
                "the checkpoint was written by a different configuration",
                hint="resume with the run's config.resolved.yaml, not a modified recipe",
            )
        self.global_step = state.global_step
        self.policy_version = state.policy_version
        self.cycle = state.cycle
        self.task_cursor = state.task_cursor
        if state.scheduler:
            self.schedule = LearningRateSchedule.from_state_dict(state.scheduler)
        self.events.emit("checkpoint_loaded", path=str(directory), step=self.global_step)
        return state

    def close(self) -> None:
        """Release models and connections."""
        closer = getattr(self.environment, "close", None)
        if callable(closer):
            closer()
        if self._cache is not None:
            self._cache.flush()

    def __enter__(self) -> OPDTrainer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _raise_config(message: str, hint: str | None = None) -> None:  # pragma: no cover - helper
    raise MiniVerlError(message, hint)
