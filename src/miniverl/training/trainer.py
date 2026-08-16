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

import gc
import hashlib
import random
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from miniverl import __version__
from miniverl.agent.loop import RolloutRunner, RolloutStats
from miniverl.alignment.workflow import build_alignment_stage_plan
from miniverl.cache.store import TeacherCache
from miniverl.config.models import (
    LossAggregation,
    LossMode,
    MemoryStrategy,
    ModelBackend,
    ModelRuntime,
    OfflineKDTrajectorySource,
    OPDFreshness,
    Quantization,
    RunConfig,
    SourceKind,
    TeacherContextMode,
    TrainingMode,
    VerlParquetSourceConfig,
)
from miniverl.environments.base import Task, ToolEnvironment, make_splits
from miniverl.environments.registry import make_environment
from miniverl.errors import (
    BackendError,
    CheckpointError,
    ConfigError,
    GpuMemoryError,
    LifecycleError,
    MiniVerlError,
)
from miniverl.models.factory import (
    build_shared_backends,
    build_student,
    build_teacher,
    build_tokenizer,
    resolve_device,
)
from miniverl.schemas.alignment import AlignmentMap
from miniverl.schemas.trajectory import Trajectory
from miniverl.selection.selectors import (
    SelectionResult,
    SelectionStats,
    aggregate_selection_stats,
    select_positions,
)
from miniverl.training.artifacts import ManifestFinalization, RunArtifactRecorder
from miniverl.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from miniverl.training.memory import (
    MemoryPlan,
    move_optimizer_state,
    resolve_strategy,
    run_with_oom_retry,
)
from miniverl.training.optim import LearningRateSchedule, build_optimizer
from miniverl.trajectory.alignment import build_alignment_map
from miniverl.trajectory.io import append_trajectories
from miniverl.utils import gpu
from miniverl.utils.env import collect_environment
from miniverl.utils.locking import RunLock
from miniverl.utils.logging import EventLog, get_logger
from miniverl.utils.runs import (
    JsonlWriter,
    RunPaths,
    make_run_id,
    read_jsonl,
    utc_now,
    write_bytes,
    write_json,
    write_json_atomic,
    write_text,
)
from miniverl.utils.seeding import capture_rng, restore_rng, seed_everything

if TYPE_CHECKING:  # pragma: no cover - typing only
    from miniverl.teachers.base import TeacherScoreResult

__all__ = ["OPDTrainer", "TrainerState", "TrainResult", "TrainSample"]

logger = get_logger("trainer")
RUN_MANIFEST_SCHEMA_VERSION = 3


class TrainerState(str, Enum):
    """One-shot lifecycle of an :class:`OPDTrainer` instance."""

    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CLOSED = "closed"


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
    parameter_version: int
    rollout_policy_version: int
    duration_seconds: float
    stop_criterion: dict[str, Any] = field(default_factory=dict)
    overshoot: dict[str, Any] = field(default_factory=dict)
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
            "global_optimizer_step": self.global_step,
            "policy_version": self.policy_version,
            "parameter_version": self.parameter_version,
            "rollout_policy_version": self.rollout_policy_version,
            "rollout_iteration": self.cycles_completed,
            "duration_seconds": round(self.duration_seconds, 3),
            "stop_criterion": self.stop_criterion,
            "overshoot": self.overshoot,
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
        validated_config: RunConfig,
        paths: RunPaths,
        run_id: str,
        environment: ToolEnvironment | None,
        splits: dict[str, list[Task]],
        rollout_runtime: Any,
        prompt_dataset: Any | None,
        prompt_dataset_manifest: Any | None,
        tokenizer: Any,
        student: Any,
        teacher: Any | None,
        reference: Any | None,
        plan: MemoryPlan,
        run_lock: RunLock,
        evaluation_only: bool = False,
    ) -> None:
        self._closed = False
        self._state = TrainerState.READY
        self._state_guard = threading.Lock()
        self._operation_guard = threading.Lock()
        self._run_lock: RunLock | None = run_lock
        self._evaluation_only = evaluation_only
        self.config = config
        self.validated_config = validated_config
        self.paths = paths
        self.run_id = run_id
        self.environment = environment
        self.splits = splits
        self.rollout_runtime = rollout_runtime
        self.prompt_dataset = prompt_dataset
        self.prompt_dataset_manifest = prompt_dataset_manifest
        self._prompt_train_iterator: Any | None = None
        self._prompt_train_epoch = 0
        self.tokenizer = tokenizer
        self.student = student
        self.teacher = teacher
        self.plan = plan
        self._started_at = utc_now()
        if paths.manifest_start.is_file():
            try:
                import json

                self._started_at = json.loads(paths.manifest_start.read_text(encoding="utf-8")).get(
                    "started_at", self._started_at
                )
            except (OSError, UnicodeError, ValueError):
                pass

        self.metrics_log = JsonlWriter(paths.metrics)
        self.events = EventLog(JsonlWriter(paths.events))
        self.runner = getattr(rollout_runtime, "runner", None)
        # Imported here rather than at module scope: the scorer pulls in torch, and
        # `import miniverl.trainer` must stay readable on a bare install so the CLI
        # can raise MissingDependencyError instead of ModuleNotFoundError.
        if teacher is not None:
            from miniverl.teachers.local import LocalTeacherScorer

            self.scorer: Any = LocalTeacherScorer(
                teacher,
                config.loss,
                keep_exact_resident=(
                    plan.strategy is MemoryStrategy.RESIDENT
                    and config.run.mode is not TrainingMode.OFFLINE_KD
                ),
                device=plan.device,
            )
        else:
            self.scorer = None
        from miniverl.runtime.roles import LocalArtifactBridge, LocalRoleGraph

        self.reference = reference
        self.artifact_bridge = LocalArtifactBridge(paths.root)
        self.role_graph = LocalRoleGraph(
            actor_policy=self.student,
            rollout_runtime=self.rollout_runtime,
            teacher_policy=self.teacher,
            reference_policy=self.reference,
            reward_or_verifier=self.environment,
            target_builder=self.scorer,
            update_runtime=self,
            evaluation_runtime=self,
            artifact_bridge=self.artifact_bridge,
        )
        self.optimizer = (
            None
            if evaluation_only
            else build_optimizer(student.trainable_parameters(), config.train)
        )
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
        self.parameter_version = 0
        self.cycle = 0
        self._last_rollout_policy_version = 0
        self.task_cursor = 0
        self._task_order = self._build_task_order()
        self._cache: TeacherCache | None = None
        self._offline_samples: list[TrainSample] | None = None
        self.offline_dataset_digest = ""
        self._offline_collection_checkpoint_digest: str | None = None
        self._teacher_on_device = teacher is not None and plan.strategy is MemoryStrategy.RESIDENT
        #: First cycle `train()` will execute. Set by `load_from_checkpoint` so a
        #: resumed run continues instead of redoing completed cycles.
        self._start_cycle = 0
        self._resumed = False
        self._resumed_from: dict[str, Any] | None = None
        self._cycles_completed = 0
        self._last_cycle_metrics: dict[str, Any] = {}
        self._last_selection_stats: list[SelectionStats] = []
        self._last_rollout_execution: dict[str, Any] | None = None

    # -- construction --------------------------------------------------------

    @property
    def optimizer_steps_per_cycle(self) -> int:
        """Optimizer steps performed per training cycle."""
        accum = self.config.train.gradient_accumulation_steps
        rollouts = self.config.train.rollouts_per_cycle
        return max(1, (rollouts + accum - 1) // accum)

    @property
    def state(self) -> TrainerState:
        """Current one-shot lifecycle state."""
        with self._state_guard:
            return self._state

    @property
    def policy_version(self) -> int:
        """Deprecated alias for the exact student parameter version."""
        return self.parameter_version

    @policy_version.setter
    def policy_version(self, value: int) -> None:
        self.parameter_version = value

    def set_offline_collection_checkpoint_digest(self, digest: str) -> None:
        """Bind frozen-student collection to the exact loaded cold-start checkpoint."""
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ConfigError("cold-start checkpoint digest must be 64 lowercase hex characters")
        self._offline_collection_checkpoint_digest = digest

    def _apply_checkpoint_progress(self, state: CheckpointState) -> None:
        """Copy validated, non-tensor progress into this trainer.

        Training resume and weights-only standalone evaluation both need the
        evaluated model to report the checkpoint's real parameter version.
        This helper mutates counters only; it never loads optimizer or RNG
        state.
        """
        self.global_step = state.global_step
        self.parameter_version = (
            state.policy_version if state.parameter_version is None else state.parameter_version
        )
        self.cycle = state.cycle
        self._last_rollout_policy_version = (
            state.policy_version
            if state.rollout_policy_version is None
            else state.rollout_policy_version
        )
        self.task_cursor = state.task_cursor
        self._cycles_completed = (
            max(state.cycle + 1, 0) if state.rollout_iteration is None else state.rollout_iteration
        )

    @classmethod
    def from_config(
        cls,
        config: RunConfig,
        *,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
        local_files_only: bool = False,
        write_artifacts: bool = True,
        overwrite: bool = False,
        resume: str | Path | None = None,
        resume_from: str | Path | None = None,
        for_evaluation: bool = False,
        lock_timeout: float = 0.0,
        already_acquired_lock: RunLock | None = None,
    ) -> OPDTrainer:
        """Validate, seed, load models and create the run directory.

        ``write_artifacts=False`` attaches to an existing run directory without
        rewriting ``config.original.yaml`` or ``manifest.json``.  The standalone
        evaluator uses it so re-evaluating a run cannot destroy the provenance of
        the run being evaluated.
        """
        resume_options = int(resume is not None) + int(resume_from is not None)
        if resume_options > 1:
            raise ConfigError("--resume and --resume-from are mutually exclusive")
        if overwrite and resume_options:
            raise ConfigError("--overwrite cannot be combined with --resume or --resume-from")
        if not write_artifacts and (overwrite or resume_options):
            raise ConfigError(
                "write_artifacts=False is an evaluator attachment mode and cannot resume "
                "or overwrite a run"
            )
        if for_evaluation and write_artifacts:
            raise ConfigError("for_evaluation=True requires write_artifacts=False")
        if for_evaluation and resume_options:
            raise ConfigError("evaluation attachment cannot use training resume options")
        if already_acquired_lock is not None and not (for_evaluation and not write_artifacts):
            raise ConfigError(
                "already_acquired_lock is reserved for read/write evaluation attachments"
            )
        if already_acquired_lock is not None and not already_acquired_lock.acquired:
            raise ConfigError("already_acquired_lock must already own the run lock")

        validated_config = config
        config = config.resolved_for_runtime()

        # bitsandbytes 4-bit parameters are pinned to the device they were
        # quantized on, so a quantized model cannot be moved off the GPU and
        # back. `swap` is therefore only available for unquantized pairs.
        quantized = (
            config.models.student.quantization is not Quantization.NONE
            or config.models.teacher.quantization is not Quantization.NONE
        )
        memory_config = config.memory
        from miniverl.bridge.opd_capabilities import assert_runtime_placement_legal

        assert_runtime_placement_legal(
            strategy=memory_config.strategy.value,
            student_quantization=config.models.student.quantization.value,
            teacher_quantization=config.models.teacher.quantization.value,
        )
        if (
            config.models.runtime is ModelRuntime.SHARED_BACKBONE
            and memory_config.strategy is MemoryStrategy.SWAP
        ):
            raise ConfigError(
                "memory.strategy=swap is unavailable with shared_backbone because all "
                "policy roles own one resident physical base",
                hint="use memory.strategy: resident/auto, or models.runtime: dual_model "
                "when host/device teacher swapping is required",
            )
        if (
            config.models.runtime is ModelRuntime.SHARED_BACKBONE
            and memory_config.strategy is MemoryStrategy.AUTO
        ):
            memory_config = memory_config.model_copy(update={"strategy": MemoryStrategy.RESIDENT})
        if quantized and memory_config.strategy is MemoryStrategy.SWAP:
            raise ConfigError(
                "memory.strategy=swap cannot be used with a quantized model: "
                "bitsandbytes 4-bit/8-bit parameters are pinned to the device they "
                "were quantized on and cannot be moved to host memory and back.",
                hint="use memory.strategy: resident (a 0.6B QLoRA student plus a bf16 "
                "1.7B teacher fits in 16 GB), or set both quantization fields to "
                "'none' if you really need swap",
            )
        if quantized and memory_config.strategy is MemoryStrategy.AUTO:
            memory_config = memory_config.model_copy(update={"strategy": MemoryStrategy.RESIDENT})

        seed_everything(config.run.seed, deterministic=config.run.deterministic)
        resolved_id = make_run_id(config.run.name, explicit=run_id or config.run.run_id)
        resume_checkpoint: Path | None = None
        if resume is not None:
            target_root = Path(resume).resolve()
            resolved_id = target_root.name
        elif resume_from is not None:
            resume_checkpoint = Path(resume_from).resolve()
            target_root = resume_checkpoint.parent.parent
            resolved_id = target_root.name
        else:
            target_root = Path(output_dir or config.run.output_dir).resolve() / resolved_id

        external_run_lock = already_acquired_lock is not None
        if already_acquired_lock is not None:
            run_lock = already_acquired_lock
            if (
                run_lock.output_root != target_root.parent.resolve()
                or run_lock.run_id != resolved_id
            ):
                raise ConfigError("already_acquired_lock does not own the requested run directory")
        else:
            run_lock = RunLock(target_root.parent, resolved_id, timeout=lock_timeout)
            run_lock.acquire()
        try:
            if resume is not None or resume_from is not None or not write_artifacts:
                paths = RunPaths.open(target_root)
            else:
                paths = RunPaths.create(
                    target_root.parent,
                    resolved_id,
                    overwrite=overwrite,
                )
        except BaseException:
            if not external_run_lock:
                run_lock.release()
            raise

        environment: ToolEnvironment | None = None
        rollout_runtime: Any | None = None
        prompt_dataset: Any | None = None
        prompt_dataset_manifest: Any | None = None
        student: Any | None = None
        teacher: Any | None = None
        reference: Any | None = None
        loaded_teachers: list[Any] = []
        trainer: OPDTrainer | None = None
        try:
            if write_artifacts and resume_options == 0:
                if validated_config.submitted_bytes is not None:
                    write_bytes(paths.config_submitted, validated_config.submitted_bytes)
                write_text(paths.config_validated, validated_config.to_yaml())
                write_text(paths.config_original, config.to_yaml())
            splits: dict[str, list[Task]] = {"train": [], "eval": [], "test": []}
            if config.source.kind is SourceKind.ENVIRONMENT:
                assert config.environment is not None
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
            else:
                from miniverl.data.verl_parquet import VerlParquetDataset

                assert isinstance(config.source, VerlParquetSourceConfig)
                prompt_dataset = VerlParquetDataset(config.source)
                prompt_dataset_manifest = prompt_dataset.inspect()
                if prompt_dataset_manifest.rows["train"] == 0:
                    raise ConfigError("the Parquet training source contains zero prompt rows")
                if config.eval.enabled and prompt_dataset_manifest.rows["val"] == 0:
                    raise ConfigError(
                        "evaluation is enabled but source.val_files contains zero prompt rows",
                        hint="provide validation prompts or set eval.enabled=false",
                    )
                if (
                    config.eval.enabled
                    and config.eval.tasks is not None
                    and config.eval.task_offset + config.eval.tasks
                    > prompt_dataset_manifest.rows["val"]
                ):
                    raise ConfigError(
                        "eval task range exceeds the number of Parquet validation rows"
                    )
            if config.models.teacher.mode is TeacherContextMode.PRIVILEGED_CONTEXT:
                assert environment is not None
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
            plan = resolve_strategy(memory_config, device=device, chunk_size=config.loss.chunk_size)
            if config.models.runtime is ModelRuntime.SHARED_BACKBONE:
                student, teacher, reference = build_shared_backends(
                    config,
                    tokenizer,
                    device=device,
                    local_files_only=local_files_only,
                    include_teacher=(
                        config.run.mode is not TrainingMode.SFT and not for_evaluation
                    ),
                )
                plan.reason = "shared_backbone -> resident: all policy roles share one base"
            else:
                student = build_student(
                    config, tokenizer, device=device, local_files_only=local_files_only
                )
            if quantized and config.memory.strategy is MemoryStrategy.AUTO:
                plan.reason = (
                    "auto -> resident: a quantized model cannot be moved off the "
                    "accelerator, so swap is unavailable"
                )
            if (
                config.models.runtime is ModelRuntime.DUAL_MODEL
                and config.run.mode is not TrainingMode.SFT
                and not for_evaluation
            ):
                if memory_config.strategy is MemoryStrategy.AUTO and device.startswith("cuda"):

                    def teacher_fits() -> bool:
                        try:
                            loaded_teachers.append(
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
                        memory_config,
                        device=device,
                        chunk_size=config.loss.chunk_size,
                        teacher_fits=teacher_fits,
                    )
                    teacher = loaded_teachers[0] if loaded_teachers else None
                if teacher is None:
                    teacher_device = device if plan.strategy is MemoryStrategy.RESIDENT else "cpu"
                    teacher = build_teacher(
                        config,
                        tokenizer,
                        device=teacher_device,
                        local_files_only=local_files_only,
                    )
                if teacher.vocab_size != student.vocab_size:
                    raise BackendError(
                        "student and teacher LM-head output dimensions differ "
                        f"({student.vocab_size} vs {teacher.vocab_size})",
                        hint="miniVERL requires a shared output vocabulary for distillation",
                    )

            if config.source.kind is SourceKind.ENVIRONMENT:
                assert environment is not None
                from miniverl.runtime.rollout import ToolEnvironmentRolloutRuntime

                rollout_runtime = ToolEnvironmentRolloutRuntime(
                    RolloutRunner(
                        backend=student,
                        environment=environment,
                        config=config.rollout,
                    )
                )
            else:
                from miniverl.runtime.rollout import PromptDatasetRolloutRuntime

                assert isinstance(config.source, VerlParquetSourceConfig)
                rollout_runtime = PromptDatasetRolloutRuntime(
                    backend=student,
                    source_config=config.source,
                    rollout_config=config.rollout,
                )

            trainer = cls(
                config=config,
                validated_config=validated_config,
                paths=paths,
                run_id=resolved_id,
                environment=environment,
                splits=splits,
                rollout_runtime=rollout_runtime,
                prompt_dataset=prompt_dataset,
                prompt_dataset_manifest=prompt_dataset_manifest,
                tokenizer=tokenizer,
                student=student,
                teacher=teacher,
                reference=reference,
                plan=plan,
                run_lock=run_lock,
                evaluation_only=for_evaluation,
            )
            if write_artifacts and resume_options == 0:
                trainer._write_startup_artifacts()
            elif resume_options:
                from miniverl.training.checkpoint import latest_checkpoint

                selected = resume_checkpoint or latest_checkpoint(paths.checkpoints)
                if selected is None:
                    raise ConfigError(
                        f"cannot resume {paths.root}: no valid checkpoint was found",
                        hint="pass --resume-from <checkpoint-dir> or start a new run",
                    )
                if not trainer._operation_guard.acquire(blocking=False):  # pragma: no cover
                    raise LifecycleError("trainer construction could not own checkpoint loading")
                try:
                    state = trainer._load_from_checkpoint_impl(selected)
                finally:
                    trainer._operation_guard.release()
                trainer.events.emit(
                    "resume_start",
                    checkpoint=str(selected),
                    previous_global_step=state.global_step,
                    previous_policy_version=state.policy_version,
                    next_cycle=trainer._start_cycle,
                )
            return trainer
        except BaseException as construction_error:
            # Construction owns every resource it has allocated. Teardown errors
            # are logged because the construction failure is the actionable root
            # cause and must retain its original traceback.
            if trainer is not None:
                trainer._run_lock = None
                try:
                    trainer.close()
                except LifecycleError as cleanup_error:
                    logger.warning("cleanup after trainer construction failure: %s", cleanup_error)
            else:
                owned_teachers = [reference, teacher, *loaded_teachers]
                seen: set[int] = set()
                for backend in [*owned_teachers, student]:
                    if backend is None or id(backend) in seen:
                        continue
                    seen.add(id(backend))
                    try:
                        backend.release()
                    except BaseException as cleanup_error:
                        logger.warning(
                            "backend cleanup after trainer construction failure: %s",
                            cleanup_error,
                        )
                backend = None
                owned_teachers.clear()
                loaded_teachers.clear()
                teacher = None
                reference = None
                student = None
                if environment is not None:
                    closer = getattr(environment, "close", None)
                    if callable(closer):
                        try:
                            closer()
                        except BaseException as cleanup_error:
                            logger.warning(
                                "environment cleanup after trainer construction failure: %s",
                                cleanup_error,
                            )
                environment = None
                closer = None
                try:
                    gc.collect()
                    gpu.empty_cache()
                except BaseException as cleanup_error:
                    logger.warning(
                        "allocator cleanup after trainer construction failure: %s",
                        cleanup_error,
                    )
            if write_artifacts and resume_options == 0 and paths.root.exists():
                try:
                    shutil.rmtree(paths.root)
                except BaseException as cleanup_error:
                    logger.warning(
                        "run-directory cleanup after trainer construction failure: %s",
                        cleanup_error,
                    )
                    failed_manifest = {
                        "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                        "status": "failed_construction",
                        "run_id": resolved_id,
                        "failed_at": utc_now(),
                        "failure": {
                            "type": type(construction_error).__name__,
                            "message": str(construction_error),
                        },
                        "cleanup_failure": {
                            "type": type(cleanup_error).__name__,
                            "message": str(cleanup_error),
                        },
                    }
                    try:
                        write_json_atomic(paths.manifest, failed_manifest)
                    except BaseException as manifest_error:
                        logger.warning(
                            "failed-construction manifest cleanup fallback failed: %s",
                            manifest_error,
                        )
            if not external_run_lock:
                run_lock.release()
            raise

    def _write_startup_artifacts(self) -> None:
        config = self.config
        resolved = config.model_copy(deep=True)
        resolved.run.run_id = self.run_id
        resolved.memory.strategy = self.plan.strategy
        resolved.loss.chunk_size = self.plan.chunk_size
        resolved.models.device = self.plan.device
        if config.models.backend is ModelBackend.HF:
            resolved.loss.top_k = min(config.loss.top_k, self.student.vocab_size)
        write_text(self.paths.config_resolved, resolved.to_yaml())
        write_json(self.paths.environment, collect_environment())
        startup = self.build_manifest()
        startup.update(
            {
                "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "status": "ready",
                "started_at": self._started_at,
                "original_config_digest": hashlib.sha256(
                    self.paths.config_original.read_bytes()
                ).hexdigest(),
                "resolved_config_digest": hashlib.sha256(
                    self.paths.config_resolved.read_bytes()
                ).hexdigest(),
                "config_provenance": {
                    "submitted": (
                        "verbatim_file_bytes"
                        if self.paths.config_submitted.is_file()
                        else "generated_no_source_bytes"
                    ),
                    "validated": "canonical_logical_config",
                    "legacy_original": "canonical_runtime_input_compatibility_alias",
                    "resolved": "runtime_choices_and_auto_resolution",
                },
                "config_digests": {
                    "submitted": (
                        hashlib.sha256(self.paths.config_submitted.read_bytes()).hexdigest()
                        if self.paths.config_submitted.is_file()
                        else None
                    ),
                    "validated": hashlib.sha256(
                        self.paths.config_validated.read_bytes()
                    ).hexdigest(),
                    "legacy_original": hashlib.sha256(
                        self.paths.config_original.read_bytes()
                    ).hexdigest(),
                    "resolved": hashlib.sha256(self.paths.config_resolved.read_bytes()).hexdigest(),
                },
                "initial_memory": self.plan.to_dict(),
                "global_step": 0,
                "global_optimizer_step": 0,
                "parameter_version": 0,
                "policy_version": 0,
                "rollout_iteration": 0,
                "rollout_policy_version": 0,
            }
        )
        write_json_atomic(self.paths.manifest_start, startup)
        write_json_atomic(self.paths.manifest, startup)

    def _transition_manifest_to_running(self) -> None:
        """Atomically publish RUNNING immediately before training emits events."""
        RunArtifactRecorder(self.paths, started_at=self._started_at).mark_running()

    def build_manifest(self) -> dict[str, Any]:
        """Full provenance record for the run."""
        env_info = collect_environment()
        config = self.config
        if config.run.mode is TrainingMode.SFT:
            objective: dict[str, Any] = {
                "name": "sft_cross_entropy",
                "loss_mode": None,
                "divergence": None,
                "temperature": None,
                "scale_by_temperature_squared": None,
                "top_k": None,
                "jsd_beta": None,
                "sampled_token_nll_weight": None,
                "selector": config.selection.selector.value,
                "selection_ratio": config.selection.ratio,
                "opd_freshness": None,
            }
        else:
            name = "offline_knowledge_distillation"
            if config.run.mode is TrainingMode.OPD:
                name = (
                    "on_policy_distillation"
                    if config.is_on_policy
                    else "online_distillation_with_replay"
                )
            objective = {
                "name": name,
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
                "sampled_token_nll_weight": config.loss.sampled_token_nll_weight,
                "selector": config.selection.selector.value,
                "selection_ratio": config.selection.ratio,
                "opd_freshness": (
                    config.train.opd_freshness.value
                    if config.run.mode is TrainingMode.OPD
                    else None
                ),
            }
        if self.environment is not None and config.environment is not None:
            environment_info: dict[str, Any] | None = {
                **self.environment.describe(),
                "difficulty": config.environment.difficulty,
                "split_seed": config.environment.split_seed,
                "split_sizes": {k: len(v) for k, v in self.splits.items()},
            }
            source_info: dict[str, Any] = {
                "kind": "environment",
                "environment": environment_info,
            }
        else:
            assert isinstance(config.source, VerlParquetSourceConfig)
            assert self.prompt_dataset_manifest is not None
            environment_info = None
            source_info = {
                "kind": "verl_parquet",
                "prompt_key": config.source.prompt_key,
                "use_task_rewards": config.source.use_task_rewards,
                "rows": self.prompt_dataset_manifest.rows,
                "schema_digest": self.prompt_dataset_manifest.schema_digest,
                "content_digest": self.prompt_dataset_manifest.content_digest,
                "files": list(self.prompt_dataset_manifest.files),
                "shuffle": config.source.shuffle,
                "seed": config.source.seed,
            }
        return {
            "miniverl_version": __version__,
            "execution_plan_digest": config.run.execution_plan_digest,
            "profile_identity": config.run.profile_identity,
            "run_id": self.run_id,
            "run_name": config.run.name,
            "created_at": self._started_at,
            "git_commit": env_info["git_commit"],
            "python_version": env_info["python_version"],
            "os": env_info["os"],
            "os_release": env_info["os_release"],
            "platform": env_info["platform"],
            "packages": env_info["packages"],
            "gpu": env_info["gpu"],
            "mode": config.run.mode.value,
            "is_on_policy": config.is_on_policy,
            "opd_freshness": (
                config.train.opd_freshness.value if config.run.mode is TrainingMode.OPD else None
            ),
            "seed": config.run.seed,
            "deterministic": config.run.deterministic,
            "source": source_info,
            "environment": environment_info,
            "models": {
                "backend": config.models.backend.value,
                "runtime": config.models.runtime.value,
                "device": self.plan.device,
                "student": {
                    "model_id": config.models.student.model_id,
                    "revision": config.models.student.revision,
                    "tokenizer_revision": config.models.student.tokenizer_revision,
                    "quantization": config.models.student.quantization.value,
                    "precision": config.models.student.dtype.value,
                    "adapter": getattr(self.student, "adapter_provenance", None),
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
                        "adapter": getattr(self.teacher, "adapter_provenance", None),
                        "capabilities": self.teacher.capabilities.to_dict(),
                    }
                    if self.teacher is not None
                    else None
                ),
                "reference": (
                    {
                        "model_id": config.models.reference.model_id,
                        "revision": config.models.reference.revision,
                        "tokenizer_revision": config.models.reference.tokenizer_revision,
                        "quantization": config.models.reference.quantization.value,
                        "precision": config.models.reference.dtype.value,
                        "adapter": getattr(self.reference, "adapter_provenance", None),
                        "capabilities": self.reference.capabilities.to_dict(),
                    }
                    if self.reference is not None and config.models.reference is not None
                    else None
                ),
                "role_residency": (
                    "one resident physical base with adapter role switching"
                    if config.models.runtime is ModelRuntime.SHARED_BACKBONE
                    else (
                        "separate resident role models"
                        if self.plan.strategy is MemoryStrategy.RESIDENT
                        else "separate role models with teacher host/device swapping"
                    )
                ),
                "tokenizer_fingerprint": self.tokenizer.fingerprint,
                "tokenizer_identity": getattr(self.tokenizer, "identity", {}),
                "tokenizer_vocab_size": self.tokenizer.vocab_size,
                "student_lm_head_vocab_size": self.student.vocab_size,
                "teacher_lm_head_vocab_size": (
                    self.teacher.vocab_size if self.teacher is not None else None
                ),
            },
            "objective": objective,
            "alignment_workflow": (
                build_alignment_stage_plan(
                    config.alignment,
                    sft_warmup_cycles=config.train.sft_warmup_cycles,
                )
                if config.alignment is not None
                else None
            ),
            "runtime_role_graph": self.role_graph.describe(),
            "artifact_bridge": {
                "kind": "local_filesystem",
                "run_root": self.paths.root.name,
            },
            "memory": self.plan.to_dict(),
            "global_optimizer_step": self.global_step,
            "parameter_version": self.parameter_version,
            "policy_version": self.policy_version,
            "rollout_iteration": self._cycles_completed,
            "rollout_policy_version": self._last_rollout_policy_version,
            "measurement_status": {
                "cpu_metrics": "measured",
                "cuda_metrics": "measured" if gpu.cuda_available() else "not_run_no_cuda",
                "simulated_results": "none",
            },
        }

    def _finalize_manifest(
        self,
        *,
        status: Literal["completed", "failed", "interrupted"],
        result: TrainResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Atomically combine immutable startup provenance with final run state."""
        result_payload = None
        if result is not None:
            result_payload = result.to_dict()
            result_payload["run_dir"] = self.paths.root.name
        RunArtifactRecorder(self.paths, started_at=self._started_at).finalize(
            ManifestFinalization(
                status=status,
                global_step=self.global_step,
                parameter_version=self.parameter_version,
                policy_version=self.policy_version,
                cycles_completed=self._cycles_completed,
                rollout_policy_version=self._last_rollout_policy_version,
                projection_chunk_size=self.plan.chunk_size,
                chunk_size_history=tuple(self.plan.chunk_size_history),
                oom_retries=self.plan.oom_retries_used,
                final_memory=self.plan.to_dict(),
                offline_dataset_digest=self.offline_dataset_digest,
                resumed_from=self._resumed_from,
                require_offline_dataset=(
                    self.config.run.mode is TrainingMode.OFFLINE_KD and self.config.train.cycles > 0
                ),
                result=result_payload,
                error=error,
            ),
            fallback_manifest=(
                {} if self.paths.manifest_start.is_file() else self.build_manifest()
            ),
        )

    # -- task sampling --------------------------------------------------------

    def _build_task_order(self) -> list[int]:
        if self.prompt_dataset is not None:
            return []
        order = list(range(len(self.splits["train"])))
        random.Random(self.config.run.seed ^ 0x5EED).shuffle(order)
        return order

    def _next_tasks(self, count: int) -> list[Any]:
        if self.prompt_dataset is not None:
            from miniverl.data.verl_parquet import render_prompt

            assert isinstance(self.config.source, VerlParquetSourceConfig)
            assert self.prompt_dataset_manifest is not None
            output: list[Any] = []
            while len(output) < count:
                if self._prompt_train_iterator is None:
                    rows_per_epoch = int(self.prompt_dataset_manifest.rows["train"])
                    self._prompt_train_epoch = self.task_cursor // rows_per_epoch
                    row_offset = self.task_cursor % rows_per_epoch
                    self._prompt_train_iterator = iter(
                        self.prompt_dataset.iter_split("train", epoch=self._prompt_train_epoch)
                    )
                    for _ in range(row_offset):
                        next(self._prompt_train_iterator)
                try:
                    record = next(self._prompt_train_iterator)
                except StopIteration:
                    self._prompt_train_iterator = None
                    continue
                output.append(render_prompt(record, self.tokenizer, self.config.source))
                self.task_cursor += 1
            return output
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
            or self.environment is None
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
        # fork_rng: fit_toy_model calls torch.manual_seed, which would otherwise
        # clobber the RNG stream the rollouts (and a restored checkpoint) depend on.
        import torch

        with torch.random.fork_rng(devices=[]):
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

    def _collect(
        self,
        tasks: list[Any],
        *,
        oracle: bool,
        rollout_seed_base: int | None = None,
    ) -> tuple[list[Trajectory], RolloutStats]:
        stats = RolloutStats()
        trajectories: list[Trajectory] = []
        if self.prompt_dataset is not None:
            if oracle:
                raise ConfigError("Parquet prompt rollouts have no oracle trajectory source")
            seed = (
                rollout_seed_base
                if rollout_seed_base is not None
                else self.config.run.seed + self.global_step * 1013
            )
            prepared = self.rollout_runtime.prepare_batch(tasks)
            generated = self.rollout_runtime.generate(
                prepared,
                policy_version=self.policy_version,
                seed=seed,
            )
            self._last_rollout_execution = {
                "physical_batch_sizes": list(generated.physical_batch_sizes),
                "oom_downshifts": generated.oom_downshifts,
            }
            trajectories = self.rollout_runtime.to_trajectories(
                prepared,
                generated,
                policy_version=self.policy_version,
            )
            for offset, trajectory in enumerate(trajectories):
                trajectory.metadata["generation_seed"] = seed * 1_000_003 + offset
                stats.observe(trajectory)
            append_trajectories(self.paths.trajectories, trajectories)
            return trajectories, stats
        assert self.runner is not None
        for offset, task in enumerate(tasks):
            if oracle:
                traj = self.runner.oracle_rollout(
                    task,
                    policy_version=self.policy_version,
                    trajectory_id=f"{task.task_id}:oracle:c{self.cycle}",
                )
            else:
                generation_seed = (
                    rollout_seed_base + offset
                    if rollout_seed_base is not None
                    else self.config.run.seed + self.global_step * 1013 + offset
                )
                traj = self.runner.rollout(
                    task,
                    policy_version=self.policy_version,
                    seed=generation_seed,
                )
                traj.metadata["generation_seed"] = generation_seed
            trajectories.append(traj)
            stats.observe(traj)
        append_trajectories(self.paths.trajectories, trajectories)
        return trajectories, stats

    def _open_cache(self) -> TeacherCache:
        if self._cache is None:
            from miniverl.bridge.opd_pg_contract import VERL_PG_K1_IMPLEMENTATION_VERSION
            from miniverl.losses.verl_topk import VERL_TOPK_SCORE_IMPLEMENTATION

            assert self.teacher is not None
            pg_k1 = self.config.loss.mode is LossMode.VERL_PG_K1
            top_k = (
                1
                if pg_k1
                else (
                    self.student.vocab_size
                    if self.config.loss.mode is LossMode.EXACT_FULL_VOCAB
                    else min(self.config.loss.top_k, self.student.vocab_size)
                )
            )
            path = Path(self.config.cache.dir or self.paths.teacher_cache)
            identity = {
                "teacher_model_id": self.config.models.teacher.model_id,
                "teacher_model_revision": self.config.models.teacher.revision,
                "tokenizer_fingerprint": self.tokenizer.fingerprint,
                "tokenizer_identity": getattr(self.tokenizer, "identity", {}),
                "teacher_adapter_provenance": getattr(self.teacher, "adapter_provenance", None),
                "vocab_size": self.student.vocab_size,
                "top_k": top_k,
                "temperature": self.config.loss.temperature,
                "loss_mode": self.config.loss.mode.value,
                "target_representation": (
                    "sampled_token_log_probs" if pg_k1 else "topk_distribution"
                ),
                "score_implementation_version": (
                    VERL_PG_K1_IMPLEMENTATION_VERSION
                    if pg_k1
                    else (
                        VERL_TOPK_SCORE_IMPLEMENTATION
                        if self.config.loss.mode is LossMode.VERL_FORWARD_KL_TOPK
                        else "miniverl-native-v1"
                    )
                ),
                "estimator_implementation_version": (
                    VERL_PG_K1_IMPLEMENTATION_VERSION if pg_k1 else None
                ),
                "execution_plan_digest": self.config.run.execution_plan_digest,
                "profile_identity": self.config.run.profile_identity,
                "dtype": self.config.cache.dtype,
            }
            if (path / "index.json").is_file():
                self._cache = TeacherCache.open(
                    path,
                    verify_checksums=self.config.cache.verify_checksums_on_load,
                )
                self._cache.assert_compatible(**identity)
                return self._cache
            self._cache = TeacherCache.create(
                path,
                miniverl_version=__version__,
                **identity,
                entries_per_shard=self.config.cache.entries_per_shard,
            )
        return self._cache

    def _teacher_identity(self) -> dict[str, Any]:
        teacher = self.config.models.teacher
        return {
            "model_id": teacher.model_id,
            "revision": teacher.revision,
            "adapter": getattr(self.teacher, "adapter_provenance", None),
        }

    def _persist_offline_dataset(self, samples: list[TrainSample]) -> None:
        from miniverl.training.offline_dataset import create_offline_dataset

        if self._cache is None:
            raise CheckpointError("offline KD produced no teacher cache")
        self.offline_dataset_digest = create_offline_dataset(
            self.paths,
            samples=samples,
            cache=self._cache,
            config=self.config,
            tokenizer_identity=getattr(self.tokenizer, "identity", {}),
            teacher_identity=self._teacher_identity(),
            source=self.config.offline_kd.trajectory_source.value,
            cold_start_checkpoint_digest=self._offline_collection_checkpoint_digest,
            parameter_version=self.parameter_version,
            generation_seeds=[
                int(
                    sample.trajectory.metadata.get(
                        "generation_seed", self.config.offline_kd.collection_seed + offset
                    )
                )
                for offset, sample in enumerate(samples)
            ],
        )
        self.events.emit(
            "offline_dataset_created",
            dataset_digest=self.offline_dataset_digest,
            trajectories=len(samples),
            source=self.config.offline_kd.trajectory_source.value,
        )

    def _offline_batch_for_cycle(self) -> list[TrainSample]:
        samples = self._offline_samples or []
        if not samples:
            return []
        width = self.config.train.rollouts_per_cycle
        start = (self.cycle * width) % len(samples)
        return [samples[(start + offset) % len(samples)] for offset in range(width)]

    def prepare_offline_dataset(self) -> dict[str, Any]:
        """Collect and score a frozen-student dataset without updating parameters."""
        if not self._operation_guard.acquire(blocking=False):
            raise LifecycleError("cannot prepare offline KD while another operation is active")
        try:
            self._ensure_state("prepare_offline_dataset", TrainerState.READY)
            if self.config.run.mode is not TrainingMode.OFFLINE_KD:
                raise ConfigError("prepare-offline-kd requires run.mode=offline_kd")
            if (
                self.config.offline_kd.trajectory_source
                is not OfflineKDTrajectorySource.FROZEN_STUDENT
            ):
                raise ConfigError(
                    "prepare-offline-kd requires offline_kd.trajectory_source=frozen_student"
                )
            if not self._offline_collection_checkpoint_digest:
                raise CheckpointError(
                    "prepare-offline-kd requires a validated cold-start checkpoint digest"
                )
            self._prepare_toy_teacher()
            count = self.config.offline_kd.collection_tasks or self.config.train.rollouts_per_cycle
            tasks = self._next_tasks(count)
            trajectories, stats = self._collect(
                tasks,
                oracle=False,
                rollout_seed_base=self.config.offline_kd.collection_seed,
            )
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
            self._offline_samples = samples
            self._persist_offline_dataset(samples)

            import json

            manifest = json.loads(self.paths.offline_dataset_manifest.read_text(encoding="utf-8"))
            summary = {
                "dataset_digest": self.offline_dataset_digest,
                "trajectories": len(samples),
                "rollouts": stats.rollouts,
                "parameter_version": self.parameter_version,
                "manifest": manifest,
            }
            write_json(self.paths.root / "offline-preparation.json", summary)
            self.events.emit(
                "offline_dataset_prepared",
                dataset_digest=self.offline_dataset_digest,
                trajectories=len(samples),
                parameter_version=self.parameter_version,
            )
            return summary
        finally:
            self._operation_guard.release()

    def _attach_persisted_offline_dataset(self) -> None:
        """Copy a validated immutable bundle into this run before optimization."""
        raw_source = self.config.offline_kd.dataset_path
        if not raw_source:
            raise ConfigError("persisted offline KD has no dataset_path")
        source = Path(raw_source)
        if (source / "offline-dataset" / "manifest.json").is_file():
            dataset = source / "offline-dataset"
            cache = source / "teacher-cache"
        elif (source / "manifest.json").is_file():
            dataset = source
            cache = source.parent / "teacher-cache"
        else:
            raise CheckpointError(f"persisted offline dataset manifest not found under {source}")
        if not cache.is_dir():
            raise CheckpointError(f"persisted offline teacher cache not found: {cache}")

        import json

        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        dataset_checkpoint_digest = manifest.get("cold_start_checkpoint_digest")
        expected_checkpoint_digest = self._offline_collection_checkpoint_digest
        if expected_checkpoint_digest and dataset_checkpoint_digest != expected_checkpoint_digest:
            raise CheckpointError(
                "persisted offline dataset cold-start checkpoint digest does not match "
                "the loaded checkpoint"
            )
        expected_schedule = self.config.offline_kd.task_schedule_digest
        if expected_schedule and manifest.get("task_schedule_digest") != expected_schedule:
            raise CheckpointError(
                "persisted offline dataset task schedule digest does not match the recipe"
            )
        shutil.copytree(dataset, self.paths.offline_dataset)
        shutil.copytree(cache, self.paths.teacher_cache, dirs_exist_ok=True)
        self._offline_collection_checkpoint_digest = dataset_checkpoint_digest
        self._load_offline_dataset(expected_digest=str(manifest.get("dataset_digest") or ""))
        self.task_cursor = len(self._offline_samples or [])
        self.events.emit(
            "persisted_offline_dataset_attached",
            dataset_digest=self.offline_dataset_digest,
            source=source.name,
        )

    def _load_offline_dataset(self, *, expected_digest: str) -> None:
        from miniverl.losses.bucketed import bucketed_teacher_entropy
        from miniverl.losses.chunked import BucketedTargetProvider, VerlTopKTargetProvider
        from miniverl.teachers.base import TeacherScoreResult
        from miniverl.training.offline_dataset import load_offline_dataset

        cache = self._open_cache()
        manifest, trajectories = load_offline_dataset(
            self.paths,
            cache=cache,
            expected_digest=expected_digest,
        )
        if manifest.get("tokenizer_identity") != getattr(self.tokenizer, "identity", {}):
            raise CheckpointError("offline dataset tokenizer identity changed")
        if manifest.get("teacher_identity") != self._teacher_identity():
            raise CheckpointError("offline dataset teacher identity changed")
        expected_schedule = self.config.offline_kd.task_schedule_digest
        if expected_schedule and manifest.get("task_schedule_digest") != expected_schedule:
            raise CheckpointError("offline dataset task schedule digest changed")

        privileged = self.config.models.teacher.mode is TeacherContextMode.PRIVILEGED_CONTEXT
        task_by_id = {task.task_id: task for split in self.splits.values() for task in split}
        samples: list[TrainSample] = []
        for trajectory in trajectories:
            selection = select_positions(
                trajectory,
                self.config.selection,
                run_seed=self.config.run.seed,
            )
            teacher_view = None
            if privileged:
                assert self.runner is not None
                task = task_by_id.get(trajectory.task_id)
                if task is None:
                    raise CheckpointError(
                        f"offline trajectory references unknown task {trajectory.task_id!r}"
                    )
                teacher_view = self.runner.privileged_render(trajectory, task)
            alignment = build_alignment_map(
                trajectory,
                selection.positions,
                selection.weights,
                teacher=teacher_view,
            )
            batch = cache.read(trajectory.trajectory_id, device=self.plan.device)
            if batch.positions.tolist() != alignment.teacher_prediction_positions:
                raise CheckpointError(
                    f"offline target positions changed for {trajectory.trajectory_id!r}"
                )
            if batch.target_token_ids.tolist() != alignment.target_token_ids:
                raise CheckpointError(
                    f"offline target token IDs changed for {trajectory.trajectory_id!r}"
                )
            if batch.span_types != alignment.span_types:
                raise CheckpointError(
                    f"offline span order changed for {trajectory.trajectory_id!r}"
                )
            provider = (
                VerlTopKTargetProvider(
                    topk_indices=batch.topk_indices,
                    topk_log_probs=batch.topk_log_probs,
                    log_prob_min_clamp=self.config.loss.log_prob_min_clamp,
                    loss_max_clamp=self.config.loss.loss_max_clamp,
                )
                if self.config.loss.mode is LossMode.VERL_FORWARD_KL_TOPK
                else BucketedTargetProvider(
                    topk_indices=batch.topk_indices,
                    topk_log_probs=batch.topk_log_probs,
                    tail_log_prob=batch.tail_log_prob,
                    divergence_name=self.config.loss.divergence.value,
                    temperature=self.config.loss.temperature,
                    scale_by_temperature_squared=(self.config.loss.scale_by_temperature_squared),
                    jsd_beta=self.config.loss.jsd_beta,
                    tail_epsilon=self.config.loss.tail_epsilon,
                )
            )
            teacher_score = TeacherScoreResult(
                trajectory_id=trajectory.trajectory_id,
                policy_version=trajectory.policy_version,
                shape="cached",
                provider=provider,
                target_token_ids=batch.target_token_ids,
                weights=batch.weights,
                span_types=list(batch.span_types),
                teacher_entropy=bucketed_teacher_entropy(
                    batch.topk_log_probs,
                    batch.tail_log_prob,
                    tail_epsilon=self.config.loss.tail_epsilon,
                ),
                num_positions=int(batch.positions.numel()),
                cacheable=batch,
                metrics={"selected_positions": float(batch.positions.numel())},
            )
            samples.append(
                TrainSample(
                    trajectory=trajectory,
                    alignment=alignment,
                    selection=selection,
                    teacher=teacher_score,
                )
            )
        selected = manifest.get("selected_positions", [])
        rebuilt = [
            {
                "trajectory_id": sample.trajectory.trajectory_id,
                "count": len(sample.alignment.student_prediction_positions),
                "positions": list(sample.alignment.student_prediction_positions),
            }
            for sample in samples
        ]
        if rebuilt != selected:
            raise CheckpointError("offline dataset selection no longer matches its manifest")
        self._offline_samples = samples
        self.offline_dataset_digest = expected_digest or str(manifest["dataset_digest"])
        self.events.emit(
            "offline_dataset_loaded",
            dataset_digest=self.offline_dataset_digest,
            trajectories=len(samples),
        )

    def _build_samples(self, trajectories: list[Trajectory]) -> list[TrainSample]:
        """Select positions, align, and (for KD modes) score with the teacher."""
        config = self.config
        samples: list[TrainSample] = []
        selections: list[SelectionStats] = []
        privileged = config.models.teacher.mode is TeacherContextMode.PRIVILEGED_CONTEXT
        task_by_id = {t.task_id: t for split in self.splits.values() for t in split}

        for traj in trajectories:
            selection = select_positions(traj, config.selection, run_seed=config.run.seed)
            selections.append(selection.stats)
            if selection.stats.gate_decision is not None:
                selected = set(selection.positions)
                self.events.emit(
                    "alignment_gate_decision",
                    trajectory_id=traj.trajectory_id,
                    task_id=traj.task_id,
                    gate_version=selection.stats.gate_version,
                    gate_signal=selection.stats.gate_signal,
                    decision=selection.stats.gate_decision,
                    spans=[
                        {
                            "span_type": span.span_type.value,
                            "start": span.start,
                            "end": span.end,
                            "selected_positions": sum(
                                1
                                for position in range(span.start, span.end)
                                if position in selected
                            ),
                        }
                        for span in traj.spans
                        if span.is_critical
                    ],
                )
            if not selection.positions:
                continue
            teacher_view: Trajectory | None = None
            if privileged and self.scorer is not None:
                assert self.runner is not None
                task = task_by_id.get(traj.task_id)
                if task is not None:
                    teacher_view = self.runner.privileged_render(traj, task)
            alignment = build_alignment_map(
                traj, selection.positions, selection.weights, teacher=teacher_view
            )
            samples.append(TrainSample(trajectory=traj, alignment=alignment, selection=selection))

        self._last_selection_stats = selections

        if self.scorer is None or config.run.mode is TrainingMode.SFT:
            return samples

        cache = self._open_cache()
        for sample in samples:
            teacher_view = None
            if privileged:
                assert self.runner is not None
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

    def _loss_by_span_type(
        self,
        sample: TrainSample,
        per_token_objective: Any,
    ) -> dict[str, list[float]]:
        totals: dict[str, list[float]] = {}
        weights = sample.alignment.token_weights
        for value, weight, span in zip(
            per_token_objective.tolist(),
            weights,
            sample.alignment.span_types,
            strict=True,
        ):
            entry = totals.setdefault(span, [0.0, 0.0])
            entry[0] += value * weight
            entry[1] += weight
        return totals

    def _compute_group_gradients(
        self,
        group: list[TrainSample],
        chunk_size: int,
    ) -> dict[str, Any]:
        """Retryable forward/backward phase; never mutates optimizer parameters."""
        import torch

        from miniverl.losses.chunked import chunked_selected_position_loss
        from miniverl.training.batching import (
            build_padded_trajectory_batch,
            concatenate_target_providers,
            deterministic_padded_token_batches,
            normalize_trajectory_weights,
        )

        config = self.config
        optimizer = self.optimizer
        if optimizer is None:
            raise LifecycleError("training gradients are unavailable in evaluation-only mode")
        optimizer.zero_grad(set_to_none=True)
        self.student.set_train(True)
        device = self.student.device
        loss_total = 0.0
        positions_total = 0
        entropy_sum = 0.0
        entropy_count = 0
        span_losses: dict[str, list[float]] = {}
        divergence_total = 0.0
        sampled_nll_total = 0.0
        oracle_ce_total = 0.0
        divergence_available = False
        sampled_nll_available = False
        oracle_ce_available = False
        verl_student_mass: list[Any] = []
        verl_teacher_mass: list[Any] = []
        verl_overlap_count: list[Any] = []
        verl_overlap_advantage: list[Any] = []
        verl_pg_estimators: list[Any] = []
        verl_pg_advantages: list[Any] = []
        verl_pg_ratios: list[Any] = []
        verl_pg_metric_rows: list[dict[str, float]] = []
        group_scale = 1.0 / max(len(group), 1)
        token_mean = config.loss.aggregation is LossAggregation.TOKEN_MEAN
        group_weight_total = sum(sum(sample.alignment.token_weights) for sample in group)
        requested_batch_size = config.train.trajectory_batch_size
        physical_batch_size = (
            len(group) if requested_batch_size == "auto" else int(requested_batch_size)
        )
        physical_batch_size = max(1, min(physical_batch_size, max(len(group), 1)))
        batch_indices = deterministic_padded_token_batches(
            [len(sample.trajectory.token_ids) for sample in group],
            batch_size=physical_batch_size,
            max_padded_tokens=config.train.max_update_padded_tokens,
            sort_by_length=config.train.length_bucketing,
        )

        for index_group in batch_indices:
            samples = [group[index] for index in index_group]
            alignments = [sample.alignment for sample in samples]
            batch = build_padded_trajectory_batch(
                token_ids=[sample.trajectory.token_ids for sample in samples],
                selected_positions=[
                    alignment.student_prediction_positions for alignment in alignments
                ],
                pad_token_id=self.tokenizer.pad_token_id,
                device=device,
            )
            hidden = self.student.hidden_states_at_batch(batch, with_grad=True)
            weight_rows = [
                torch.tensor(alignment.token_weights, dtype=torch.float32, device=device)
                for alignment in alignments
            ]
            targets = torch.cat(
                [
                    torch.tensor(
                        alignment.target_token_ids,
                        dtype=torch.long,
                        device=device,
                    )
                    for alignment in alignments
                ]
            )
            providers = [
                sample.teacher.provider if sample.teacher is not None else None
                for sample in samples
            ]
            if all(provider is None for provider in providers):
                provider = None
            elif all(provider is not None for provider in providers):
                provider = concatenate_target_providers(
                    [provider for provider in providers if provider is not None],
                    [alignment.num_positions for alignment in alignments],
                )
            else:
                raise LifecycleError(
                    "one padded trajectory batch cannot mix SFT and distillation targets"
                )
            ce_weight = 1.0 if provider is None else config.loss.sampled_token_nll_weight
            microbatch_scale = 1.0 if token_mean else len(samples) * group_scale
            effective_weights = (
                torch.cat(weight_rows) if token_mean else normalize_trajectory_weights(weight_rows)
            )
            weight_normalizer = group_weight_total if token_mean else float(len(samples))
            output = chunked_selected_position_loss(
                hidden_states=hidden,
                lm_head=self.student.project,
                weights=effective_weights,
                weight_normalizer=weight_normalizer,
                provider=provider,
                target_token_ids=targets,
                ce_weight=ce_weight,
                chunk_size=chunk_size,
                backward=True,
                loss_scale=microbatch_scale,
            )
            for target_provider in providers:
                diagnostics = getattr(target_provider, "diagnostics", None)
                if not isinstance(diagnostics, list):
                    continue
                for values in diagnostics:
                    if "student_mass" in values:
                        verl_student_mass.append(values["student_mass"])
                        verl_teacher_mass.append(values["teacher_mass"])
                        verl_overlap_count.append(values["overlap_count"])
                        verl_overlap_advantage.append(values["overlap_token_advantage"])
                    elif "estimator" in values:
                        verl_pg_estimators.append(values["estimator"])
                        verl_pg_advantages.append(values["advantages"])
                        verl_pg_ratios.append(values["ratio"])
                        verl_pg_metric_rows.append(
                            {
                                key: float(value)
                                for key, value in values.items()
                                if isinstance(value, (int, float))
                            }
                        )
                diagnostics.clear()
            loss_total += float(output.loss) * microbatch_scale
            positions_total += output.num_positions
            for sample_index, (sample, weight_tensor) in enumerate(
                zip(samples, weight_rows, strict=True)
            ):
                start = batch.selected_offsets[sample_index]
                end = batch.selected_offsets[sample_index + 1]
                for name, (numerator, denominator) in self._loss_by_span_type(
                    sample,
                    output.per_token_objective[start:end],
                ).items():
                    entry = span_losses.setdefault(name, [0.0, 0.0])
                    entry[0] += numerator
                    entry[1] += denominator
                tensor_denominator = (
                    torch.tensor(group_weight_total, device=device).clamp_min(1e-12)
                    if token_mean
                    else weight_tensor.sum().clamp_min(1e-12)
                )
                if output.per_token_divergence is not None:
                    divergence_available = True
                    divergence_total += float(
                        (output.per_token_divergence[start:end].to(device) * weight_tensor).sum()
                        / tensor_denominator
                    ) * (1.0 if token_mean else group_scale)
                if output.per_token_ce is not None:
                    component = float(
                        (output.per_token_ce[start:end].to(device) * weight_tensor).sum()
                        / tensor_denominator
                    ) * (1.0 if token_mean else group_scale)
                    if provider is None:
                        oracle_ce_available = True
                        oracle_ce_total += component
                    else:
                        sampled_nll_available = True
                        sampled_nll_total += component
                if sample.teacher is not None and sample.teacher.teacher_entropy.numel():
                    entropy_sum += float(sample.teacher.teacher_entropy.sum())
                    entropy_count += int(sample.teacher.teacher_entropy.numel())
            del hidden, output

        result = {
            "loss": loss_total,
            "selected_positions": positions_total,
            "trajectories_in_step": len(group),
            "physical_trajectory_batches": len(batch_indices),
            "padded_trajectory_batch_size": physical_batch_size,
            "loss_aggregation": config.loss.aggregation.value,
            "teacher_entropy_mean": (entropy_sum / entropy_count) if entropy_count else None,
            "divergence_loss": divergence_total if divergence_available else None,
            "sampled_token_nll_loss": (sampled_nll_total if sampled_nll_available else None),
            "oracle_ce_loss": oracle_ce_total if oracle_ce_available else None,
            "loss_by_span_type": {
                name: numerator / denominator if denominator else 0.0
                for name, (numerator, denominator) in sorted(span_losses.items())
            },
        }
        if verl_student_mass:
            student_mass = torch.cat(verl_student_mass).float()
            teacher_mass = torch.cat(verl_teacher_mass).float()
            overlap_count = torch.cat(verl_overlap_count).float()
            overlap_advantage = torch.cat(verl_overlap_advantage).float()
            overlap_positions = overlap_count > 0
            result["verl_forward_kl_topk"] = {
                "student_mass_mean": float(student_mass.mean()),
                "student_mass_min": float(student_mass.min()),
                "student_mass_max": float(student_mass.max()),
                "teacher_mass_mean": float(teacher_mass.mean()),
                "teacher_mass_min": float(teacher_mass.min()),
                "teacher_mass_max": float(teacher_mass.max()),
                "overlap_ratio": float(overlap_count.mean() / config.loss.top_k),
                "overlap_token_advantage": (
                    float(overlap_advantage[overlap_positions].mean())
                    if bool(overlap_positions.any())
                    else 0.0
                ),
            }
        if verl_pg_estimators:
            estimators = torch.cat(verl_pg_estimators).float()
            advantages = torch.cat(verl_pg_advantages).float()
            ratios = torch.cat(verl_pg_ratios).float()
            metric_names = sorted({key for row in verl_pg_metric_rows for key in row})
            result["verl_pg_k1"] = {
                "estimator_mean": float(estimators.mean()),
                "estimator_abs_mean": float(estimators.abs().mean()),
                "advantage_mean": float(advantages.mean()),
                "ratio_mean": float(ratios.mean()),
                **{
                    key: sum(row.get(key, 0.0) for row in verl_pg_metric_rows)
                    / len(verl_pg_metric_rows)
                    for key in metric_names
                },
            }
        return result

    def _commit_update(self) -> dict[str, float]:
        """Non-retryable optimizer commit; ``step`` is invoked at most once."""
        import torch

        optimizer = self.optimizer
        if optimizer is None:
            raise LifecycleError("optimizer commits are unavailable in evaluation-only mode")
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.student.trainable_parameters(),
                self.config.train.max_grad_norm,
            )
        )
        lr = self.schedule.lr_at(self.global_step)
        for group_params in optimizer.param_groups:
            group_params["lr"] = lr
        try:
            optimizer.step()
        except (RuntimeError, MemoryError) as exc:
            optimizer.zero_grad(set_to_none=True)
            if gpu.is_oom_error(exc):
                raise GpuMemoryError(
                    "CUDA ran out of memory inside optimizer.step; the update may have "
                    "partially allocated optimizer state and was not retried.",
                    hint=(
                        "reducing loss.chunk_size cannot fix optimizer-state allocation. "
                        "Use a smaller model or optimizer, QLoRA/quantization, or an "
                        "8-bit optimizer, then restart from the last complete checkpoint. "
                        f"Original error: {exc}"
                    ),
                ) from exc
            raise
        optimizer.zero_grad(set_to_none=True)
        return {"grad_norm": grad_norm, "lr": lr}

    def _optimize(self, samples: list[TrainSample], *, phase: str) -> list[dict[str, Any]]:
        accum = self.config.train.gradient_accumulation_steps
        records: list[dict[str, Any]] = []
        optimizer = self.optimizer
        if optimizer is None:
            raise LifecycleError("optimization is unavailable in evaluation-only mode")
        rollout_versions = {
            int(sample.trajectory.policy_version)
            for sample in samples
            if getattr(sample, "trajectory", None) is not None
        }
        if len(rollout_versions) > 1:
            raise LifecycleError(
                "one optimizer pass cannot mix trajectories from multiple parameter versions"
            )
        rollout_policy_version = (
            next(iter(rollout_versions)) if rollout_versions else self.parameter_version
        )
        if (
            self.config.run.mode is TrainingMode.OPD
            and self.config.train.opd_freshness is OPDFreshness.STRICT
            and rollout_policy_version != self.parameter_version
        ):
            raise LifecycleError(
                "strict OPD requires rollout policy version to equal the current parameter "
                f"version before update (rollout={rollout_policy_version}, "
                f"parameters={self.parameter_version})"
            )
        for start in range(0, len(samples), accum):
            group = samples[start : start + accum]
            if not group:
                continue
            if self.config.memory.reset_peak_stats_each_cycle:
                gpu.reset_peak_stats()
            started = time.perf_counter()
            retry_rng = capture_rng()

            def run(chunk: int, batch: list[TrainSample] = group) -> dict[str, Any]:
                return self._compute_group_gradients(batch, chunk)

            def note_retry(old_chunk: int, new_chunk: int) -> None:
                self.events.emit(
                    "oom_chunk_retry",
                    old_chunk=old_chunk,
                    new_chunk=new_chunk,
                    note="objective unchanged; only the projection chunk size shrank",
                )

            def clear_grads(snapshot: Any = retry_rng) -> None:
                optimizer.zero_grad(set_to_none=True)
                restore_rng(snapshot)

            record = run_with_oom_retry(
                run,
                plan=self.plan,
                memory=self.config.memory,
                on_retry=note_retry,
                cleanup=clear_grads,
            )
            record.update(self._commit_update())
            self.global_step += 1
            self.parameter_version += 1
            elapsed = max(time.perf_counter() - started, 1e-9)
            record.update(
                {
                    "phase": phase,
                    "cycle": self.cycle,
                    "rollout_iteration": self.cycle,
                    "step": self.global_step,
                    "global_optimizer_step": self.global_step,
                    "parameter_version": self.parameter_version,
                    "policy_version": self.policy_version,
                    "rollout_policy_version": rollout_policy_version,
                    "seconds": round(elapsed, 4),
                    "train_selected_tokens_per_second": record["selected_positions"] / elapsed,
                    "projection_chunk_size": self.plan.chunk_size,
                    "memory": gpu.snapshot().to_dict(),
                    "ts": utc_now(),
                }
            )
            self.metrics_log.write(record)
            records.append(record)
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
        """Run once from ``ready`` and enter one terminal lifecycle state."""
        if not self._operation_guard.acquire(blocking=False):
            with self._state_guard:
                state = self._state
            raise LifecycleError(
                f"cannot train: this OPDTrainer is {state.value}",
                hint="wait for the active trainer operation to finish",
            )
        try:
            with self._state_guard:
                if self._state is not TrainerState.READY:
                    raise LifecycleError(
                        f"cannot train: this OPDTrainer is {self._state.value}",
                        hint=(
                            "construct a fresh trainer and use explicit resume/checkpoint "
                            "semantics for additional work"
                        ),
                    )
                self._state = TrainerState.RUNNING
            try:
                self._transition_manifest_to_running()
                result = self._train_impl()
            except KeyboardInterrupt as exc:
                try:
                    self._finalize_manifest(status="interrupted", error=exc)
                except BaseException as manifest_error:
                    logger.warning(
                        "manifest finalization failed while preserving KeyboardInterrupt: %s",
                        manifest_error,
                    )
                with self._state_guard:
                    self._state = TrainerState.INTERRUPTED
                raise
            except BaseException as exc:
                try:
                    self._finalize_manifest(status="failed", error=exc)
                except BaseException as manifest_error:
                    logger.warning(
                        "manifest finalization failed while preserving %s: %s",
                        type(exc).__name__,
                        manifest_error,
                    )
                with self._state_guard:
                    self._state = TrainerState.FAILED
                raise
            try:
                self._finalize_manifest(status="completed", result=result)
            except BaseException:
                with self._state_guard:
                    self._state = TrainerState.FAILED
                raise
            with self._state_guard:
                self._state = TrainerState.COMPLETED
            return result
        finally:
            self._operation_guard.release()

    def _train_impl(self) -> TrainResult:
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
            opd_freshness=(
                config.train.opd_freshness.value if config.run.mode is TrainingMode.OPD else None
            ),
        )
        if (
            config.run.mode is TrainingMode.OPD
            and config.train.opd_freshness is OPDFreshness.REPLAY
        ):
            self.events.emit(
                "online_distillation_replay",
                steps_per_cycle=self.optimizer_steps_per_cycle,
                note=(
                    "multiple optimizer steps may consume one rollout batch. This run "
                    "is explicitly online distillation with replay, not genuine OPD."
                ),
            )

        if (
            config.run.mode is TrainingMode.OFFLINE_KD
            and config.offline_kd.trajectory_source is OfflineKDTrajectorySource.PERSISTED
            and self._offline_samples is None
        ):
            self._attach_persisted_offline_dataset()

        if not (config.run.mode is TrainingMode.OFFLINE_KD and self._offline_samples is not None):
            self._prepare_toy_teacher()

        baseline = None
        if self._resumed:
            # A resumed run must not repeat the cold start or re-measure a
            # "baseline" that no longer describes an untrained policy.
            self.events.emit(
                "resumed",
                start_cycle=self._start_cycle,
                global_step=self.global_step,
                policy_version=self.policy_version,
                note="skipping the baseline evaluation and the SFT cold start",
            )
        else:
            alignment_warmup = (
                config.alignment is not None
                and config.train.sft_warmup_cycles > 0
                and config.run.mode is not TrainingMode.SFT
            )
            if alignment_warmup:
                self._run_sft_warmup(config.train.sft_warmup_cycles)
            if config.eval.enabled and config.eval.baseline_enabled:
                baseline = self._evaluate_impl(tag="baseline")
            if (
                config.train.sft_warmup_cycles > 0
                and config.run.mode is not TrainingMode.SFT
                and not alignment_warmup
            ):
                self._run_sft_warmup(config.train.sft_warmup_cycles)

        last_records: list[dict[str, Any]] = []
        continuation_started = time.perf_counter()
        cumulative_selected = 0
        if self._resumed:
            cumulative_selected = sum(
                int((row.get("selection") or {}).get("selected_model_tokens") or 0)
                for row in read_jsonl(self.paths.metrics)
                if str(row.get("phase", "")).endswith("_cycle")
            )
        stop_criterion: dict[str, Any] = {
            "kind": "configured_cycles",
            "target": config.train.cycles,
            "actual": self._cycles_completed,
        }
        overshoot: dict[str, Any] = {
            "axis": "optimizer_steps",
            "target": config.train.cycles,
            "actual": self._cycles_completed,
            "value": 0,
        }
        for cycle in range(self._start_cycle, config.train.cycles):
            self.cycle = cycle
            last_records = self._run_cycle()
            self._cycles_completed = cycle + 1
            cumulative_selected += int(
                (self._last_cycle_metrics.get("selection") or {}).get("selected_model_tokens") or 0
            )
            elapsed_continuation = time.perf_counter() - continuation_started
            budget_reached = False
            if config.train.max_selected_training_tokens is not None and (
                cumulative_selected >= config.train.max_selected_training_tokens
            ):
                target = config.train.max_selected_training_tokens
                stop_criterion = {
                    "kind": "selected_training_tokens",
                    "target": target,
                    "actual": cumulative_selected,
                }
                overshoot = {
                    "axis": "selected_training_tokens",
                    "target": target,
                    "actual": cumulative_selected,
                    "value": cumulative_selected - target,
                }
                budget_reached = True
            elif config.train.max_wall_seconds is not None and (
                elapsed_continuation >= config.train.max_wall_seconds
            ):
                target_seconds = config.train.max_wall_seconds
                stop_criterion = {
                    "kind": "wall_seconds",
                    "target": target_seconds,
                    "actual": elapsed_continuation,
                }
                overshoot = {
                    "axis": "wall_seconds",
                    "target": target_seconds,
                    "actual": elapsed_continuation,
                    "value": elapsed_continuation - target_seconds,
                }
                budget_reached = True
            if budget_reached:
                self.events.emit(
                    "continuation_budget_reached",
                    stop_criterion=stop_criterion,
                    overshoot=overshoot,
                    optimizer_step=self.global_step,
                )
                break
            if (
                config.eval.enabled
                and config.train.eval_every_cycles
                and (cycle + 1) % config.train.eval_every_cycles == 0
                and cycle + 1 < config.train.cycles
            ):
                self._evaluate_impl(tag=f"cycle{cycle + 1}")
            if (
                config.train.save_every_cycles
                and (cycle + 1) % config.train.save_every_cycles == 0
                and cycle + 1 < config.train.cycles
            ):
                self._save_checkpoint_impl()

        if stop_criterion["kind"] == "configured_cycles":
            stop_criterion["actual"] = self._cycles_completed
            overshoot["actual"] = self._cycles_completed
            overshoot["value"] = max(0, self._cycles_completed - config.train.cycles)

        final_eval = self._evaluate_impl(tag="final") if config.eval.enabled else None
        self._save_checkpoint_impl(name="final")
        duration = time.perf_counter() - started

        result = TrainResult(
            run_id=self.run_id,
            run_dir=self.paths.root,
            mode=config.run.mode.value,
            cycles_completed=self._cycles_completed,
            global_step=self.global_step,
            policy_version=self.policy_version,
            parameter_version=self.parameter_version,
            rollout_policy_version=self._last_rollout_policy_version,
            duration_seconds=duration,
            stop_criterion=stop_criterion,
            overshoot=overshoot,
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
            global_optimizer_step=self.global_step,
            parameter_version=self.parameter_version,
            policy_version=self.policy_version,
            rollout_iteration=self._cycles_completed,
            rollout_policy_version=self._last_rollout_policy_version,
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
            self.events.emit("sft_warmup_cycle", cycle=index, trajectories=stats.rollouts)
        self.cycle = 0

    def _build_samples_ce_only(self, trajectories: list[Trajectory]) -> list[TrainSample]:
        samples: list[TrainSample] = []
        selections: list[SelectionStats] = []
        for traj in trajectories:
            selection = select_positions(traj, self.config.selection, run_seed=self.config.run.seed)
            selections.append(selection.stats)
            if not selection.positions:
                continue
            alignment = build_alignment_map(traj, selection.positions, selection.weights)
            samples.append(TrainSample(trajectory=traj, alignment=alignment, selection=selection))
        self._last_selection_stats = selections
        return samples

    def _run_cycle(self) -> list[dict[str, Any]]:
        config = self.config
        mode = config.run.mode
        cycle_started = time.perf_counter()
        rollout_policy_version = self.parameter_version
        self._last_selection_stats = []
        self._last_rollout_execution = None
        rollout_seconds = 0.0
        teacher_scoring_seconds = 0.0

        if mode is TrainingMode.OFFLINE_KD and self._offline_samples is not None:
            samples = self._offline_batch_for_cycle()
            if samples:
                rollout_policy_version = samples[0].trajectory.policy_version
            stats = RolloutStats()
            self.events.emit(
                "offline_kd_reuse",
                cycle=self.cycle,
                trajectories=len(samples),
                note="fixed teacher targets reused; this run is explicitly not on-policy",
            )
        else:
            collection_tasks = config.train.rollouts_per_cycle
            if mode is TrainingMode.OFFLINE_KD and config.offline_kd.collection_tasks is not None:
                collection_tasks = config.offline_kd.collection_tasks
            tasks = self._next_tasks(collection_tasks)
            oracle = mode is TrainingMode.SFT or (
                mode is TrainingMode.OFFLINE_KD
                and config.offline_kd.trajectory_source is OfflineKDTrajectorySource.ORACLE
            )
            rollout_started = time.perf_counter()
            trajectories, stats = self._collect(
                tasks,
                oracle=oracle,
                rollout_seed_base=(
                    config.offline_kd.collection_seed
                    if mode is TrainingMode.OFFLINE_KD
                    and config.offline_kd.trajectory_source
                    is OfflineKDTrajectorySource.FROZEN_STUDENT
                    else None
                ),
            )
            if trajectories:
                rollout_policy_version = trajectories[0].policy_version
            rollout_seconds = max(time.perf_counter() - rollout_started, 1e-9)

            if mode is TrainingMode.SFT or self.teacher is None:
                samples = self._build_samples_ce_only(trajectories)
            else:
                teacher_scoring_started = time.perf_counter()
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
                teacher_scoring_seconds = time.perf_counter() - teacher_scoring_started

            self.events.emit(
                "rollouts_collected",
                cycle=self.cycle,
                rollout_iteration=self.cycle,
                policy_version=self.policy_version,
                parameter_version=self.parameter_version,
                rollout_policy_version=rollout_policy_version,
                trajectories=stats.rollouts,
                success_rate=(
                    None
                    if self.prompt_dataset is not None
                    else round(stats.to_dict()["success_rate"], 4)
                ),
                reward_status=(
                    "not_applicable_pure_opd" if self.prompt_dataset is not None else "measured"
                ),
                generated_tokens=stats.generated_tokens,
                rollout_tokens_per_second=round(stats.generated_tokens / rollout_seconds, 2),
                rollout_seconds=round(rollout_seconds, 4),
                teacher_scoring_seconds=round(teacher_scoring_seconds, 4),
            )
            if mode is TrainingMode.OFFLINE_KD:
                self._offline_samples = samples
                self._persist_offline_dataset(samples)
                samples = self._offline_batch_for_cycle()

        if not samples:
            # A selector can legitimately find nothing -- `tool_and_final` on a
            # policy that has not yet learned to emit a tool call, for instance.
            # Say so instead of reporting a cycle that quietly did no work.
            self.events.emit(
                "cycle_skipped_no_selected_positions",
                cycle=self.cycle,
                rollout_iteration=self.cycle,
                policy_version=self.policy_version,
                parameter_version=self.parameter_version,
                rollout_policy_version=rollout_policy_version,
                selector=config.selection.selector.value,
                trajectories=stats.rollouts,
                note=(
                    "no trajectory contained a token this selector would supervise, so "
                    "the cycle performed zero optimizer steps"
                ),
            )

        try:
            records = self._optimize(samples, phase=config.run.mode.value)
            if config.report.enabled and self.cycle == config.train.cycles - 1:
                tokens = self._write_token_analysis(samples)
                if tokens:
                    self.events.emit("token_analysis_written", tokens=tokens)
        finally:
            # Exact resident targets can close over the teacher's projection
            # method and hidden tensors. OPD never reuses them across cycles, so
            # sever those references immediately after their final consumer.
            if mode is TrainingMode.OPD:
                for sample in samples:
                    sample.teacher = None
        selection_stats = aggregate_selection_stats(
            self._last_selection_stats or [sample.selection.stats for sample in samples]
        )
        selected_for_rate = selection_stats.get("selected_model_tokens")
        selected_count = (
            int(selected_for_rate) if isinstance(selected_for_rate, (int, float)) else 0
        )
        cycle_metrics: dict[str, Any] = {
            "phase": f"{config.run.mode.value}_cycle",
            "cycle": self.cycle,
            "rollout_iteration": self.cycle,
            "step": self.global_step,
            "global_optimizer_step": self.global_step,
            "parameter_version": self.parameter_version,
            "policy_version": self.policy_version,
            "rollout_policy_version": rollout_policy_version,
            "seconds": round(time.perf_counter() - cycle_started, 3),
            "rollouts": stats.to_dict(),
            "selection": selection_stats,
            "memory": gpu.snapshot().to_dict(),
            "ts": utc_now(),
        }
        # These phase timings describe the portable prompt-data runtime added
        # for the verl-style OPD path. Keep the established environment-backed
        # metric contract byte-stable so resume comparisons remain exact.
        if self.prompt_dataset is not None:
            cycle_metrics.update(
                {
                    "rollout_seconds": round(rollout_seconds, 4),
                    "teacher_scoring_seconds": round(teacher_scoring_seconds, 4),
                    "teacher_scored_positions_per_second": (
                        round(selected_count / teacher_scoring_seconds, 2)
                        if teacher_scoring_seconds > 0
                        else None
                    ),
                    "rollout_execution": self._last_rollout_execution,
                }
            )
        if self._cache is not None:
            cycle_metrics["cache"] = self._cache.stats().model_dump(mode="json")
        self.metrics_log.write(cycle_metrics)
        self._last_cycle_metrics = cycle_metrics

        self._last_rollout_policy_version = rollout_policy_version
        if mode is TrainingMode.OPD and self._cache is not None and config.cache.keep_cycles:
            versions = sorted(self._cache.index.policy_versions())
            if len(versions) > config.cache.keep_cycles:
                removed = self._cache.prune_before(versions[-config.cache.keep_cycles])
                if removed:
                    self.events.emit("cache_pruned", removed=removed)
        # A rollout iteration is complete only after optimization (including a
        # legitimate no-op) and cycle metrics have both finished. Keeping the
        # counter here also makes an explicit checkpoint taken between public
        # train() calls resume at the next iteration.
        self._cycles_completed = max(self._cycles_completed, self.cycle + 1)
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
                teacher_sampled_log_prob: list[float | None] = [None] * len(per_token)
                if sample.teacher is not None and sample.teacher.cacheable is not None:
                    cacheable = sample.teacher.cacheable
                    if cacheable.topk_indices is not None:
                        teacher_top = cacheable.topk_indices[:, 0].tolist()
                    if cacheable.teacher_sampled_token_log_probs is not None:
                        teacher_sampled_log_prob = (
                            cacheable.teacher_sampled_token_log_probs.tolist()
                        )
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
                            "teacher_sampled_token_log_prob": (
                                teacher_sampled_log_prob[i]
                                if i < len(teacher_sampled_log_prob)
                                else None
                            ),
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
        from miniverl.losses.chunked import (
            BucketedTargetProvider,
            VerlPGK1TargetProvider,
            VerlTopKTargetProvider,
        )

        cache = self._open_cache()
        config = self.config
        for sample in samples:
            if sample.teacher is None:
                continue
            batch = cache.read(
                sample.trajectory.trajectory_id,
                expect_policy_version=(
                    sample.trajectory.policy_version if config.cache.strict_policy_version else None
                ),
                expect_prompt_row_digest=sample.trajectory.metadata.get("row_digest"),
                expect_actor_response_token_ids=[
                    token_id
                    for token_id, generated in zip(
                        sample.trajectory.token_ids,
                        sample.trajectory.model_generated_mask,
                        strict=True,
                    )
                    if generated
                ],
                device=self.plan.device,
            )
            if config.loss.mode is LossMode.VERL_PG_K1:
                if (
                    batch.old_actor_log_probs is None
                    or batch.teacher_sampled_token_log_probs is None
                ):
                    raise CheckpointError("PG-k1 cache is missing sampled-token log-probabilities")
                sample.teacher.provider = VerlPGK1TargetProvider(
                    target_token_ids=batch.target_token_ids,
                    old_actor_log_probs=batch.old_actor_log_probs,
                    teacher_sampled_token_log_probs=batch.teacher_sampled_token_log_probs,
                    clip_ratio=config.loss.clip_ratio,
                    clip_ratio_low=config.loss.clip_ratio_low,
                    clip_ratio_high=config.loss.clip_ratio_high,
                    clip_ratio_c=config.loss.clip_ratio_c,
                    loss_max_clamp=config.loss.loss_max_clamp,
                )
            elif config.loss.mode is LossMode.VERL_FORWARD_KL_TOPK:
                sample.teacher.provider = VerlTopKTargetProvider(
                    topk_indices=batch.topk_indices,
                    topk_log_probs=batch.topk_log_probs,
                    log_prob_min_clamp=config.loss.log_prob_min_clamp,
                    loss_max_clamp=config.loss.loss_max_clamp,
                )
            else:
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
        tasks: list[Any] | None = None,
        tag: str = "eval",
        write: bool = True,
    ) -> dict[str, Any]:
        """Run an evaluation only when no training operation owns the model."""
        self._ensure_state(
            "evaluate",
            TrainerState.READY,
            TrainerState.COMPLETED,
        )
        if not self._operation_guard.acquire(blocking=False):
            with self._state_guard:
                state = self._state
            raise LifecycleError(
                f"cannot evaluate: this OPDTrainer is {state.value}",
                hint="wait for the active trainer operation to finish",
            )
        try:
            self._ensure_state(
                "evaluate",
                TrainerState.READY,
                TrainerState.COMPLETED,
            )
            return self._evaluate_impl(split=split, tasks=tasks, tag=tag, write=write)
        finally:
            self._operation_guard.release()

    def _evaluate_impl(
        self,
        *,
        split: str | None = None,
        tasks: list[Any] | None = None,
        tag: str = "eval",
        write: bool = True,
    ) -> dict[str, Any]:
        """Deterministic greedy evaluation owned by the current operation."""
        config = self.config
        chosen_split = split or config.eval.split
        if self.prompt_dataset is not None:
            return self._evaluate_prompt_impl(
                chosen_split=chosen_split,
                prompts=tasks,
                tag=tag,
                write=write,
            )
        pool = tasks if tasks is not None else self.splits.get(chosen_split, [])
        limit = config.effective_eval_tasks if tasks is None else len(pool)
        task_offset = config.eval.task_offset if tasks is None else 0
        pool = pool[task_offset : task_offset + limit]
        if not pool:
            return {
                "tag": tag,
                "tasks": 0,
                "success_rate": 0.0,
                "global_step": self.global_step,
                "global_optimizer_step": self.global_step,
                "policy_version": self.policy_version,
                "parameter_version": self.parameter_version,
                "rollout_iteration": self._cycles_completed,
                "rollout_policy_version": self._last_rollout_policy_version,
                "note": "no eval tasks",
            }

        model = getattr(self.student, "model", None)
        was_training = getattr(model, "training", None)
        if not isinstance(was_training, bool):
            raise BackendError(
                "the student backend does not expose its current train/eval mode",
                hint="use a built-in miniVERL backend whose model has a boolean training flag",
            )
        self.student.set_train(False)
        try:
            gpu.reset_peak_stats()
            started = time.perf_counter()
            stats = RolloutStats()
            trajectories: list[Trajectory] = []
            by_difficulty: dict[str, list[int]] = {}
            assert self.runner is not None
            for offset, task in enumerate(pool):
                traj = self.runner.rollout(
                    task,
                    policy_version=self.policy_version,
                    seed=config.eval.seed + task_offset + offset,
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

            from miniverl.evaluation.diagnostics import lenient_diagnostic_success_rate

            payload = {
                "tag": tag,
                "split": chosen_split,
                "tasks": len(pool),
                "policy_version": self.policy_version,
                "parameter_version": self.parameter_version,
                "global_step": self.global_step,
                "global_optimizer_step": self.global_step,
                "rollout_iteration": self._cycles_completed,
                "rollout_policy_version": self._last_rollout_policy_version,
                "temperature": config.eval.temperature,
                "task_offset": task_offset,
                "seconds": round(elapsed, 3),
                "rollout_tokens_per_second": round(stats.generated_tokens / elapsed, 2),
                "success_by_difficulty": {
                    k: sum(v) / len(v) for k, v in sorted(by_difficulty.items())
                },
                "memory": gpu.snapshot().to_dict(),
                **stats.to_dict(),
                "lenient_diagnostic_success_rate": lenient_diagnostic_success_rate(trajectories),
                "protocol_token_accuracy": None,
                "policy_competence_measurement_status": {
                    "strict_task_success_rate": "measured_primary",
                    "lenient_diagnostic_success_rate": (
                        "measured_diagnostic_not_a_replacement_for_strict"
                    ),
                    "valid_tool_call_rate": (
                        "measured"
                        if stats.tool_calls + stats.invalid_tool_calls
                        else "not_observed"
                    ),
                    "parse_valid_tool_call_rate": (
                        "measured" if stats.emitted_tool_calls else "not_observed"
                    ),
                    "tool_execution_success_rate": (
                        "measured" if stats.parsed_tool_calls else "not_observed"
                    ),
                    "tool_execution_error_rate": (
                        "measured" if stats.parsed_tool_calls else "not_observed"
                    ),
                    "final_answer_format_validity_rate": "measured",
                    "protocol_token_accuracy": (
                        "not_applicable_free_running_trajectories_have_no_aligned_token_target"
                    ),
                },
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
        finally:
            self.student.set_train(was_training)

    def _evaluate_prompt_impl(
        self,
        *,
        chosen_split: str,
        prompts: list[Any] | None,
        tag: str,
        write: bool,
    ) -> dict[str, Any]:
        """Generate validation responses without inventing a reward or success metric."""
        from itertools import islice

        from miniverl.data.verl_parquet import render_prompt
        from miniverl.runtime.rollout import PromptDatasetRolloutRuntime

        config = self.config
        assert isinstance(config.source, VerlParquetSourceConfig)
        assert self.prompt_dataset is not None
        assert self.prompt_dataset_manifest is not None
        if chosen_split not in {"eval", "val"}:
            raise ConfigError(
                "Parquet prompt evaluation uses the validation files",
                hint="set eval.split=eval (the compatibility name for source.val_files)",
            )
        if prompts is None:
            available = self.prompt_dataset_manifest.rows["val"]
            limit = config.eval.tasks if config.eval.tasks is not None else available
            records = islice(
                self.prompt_dataset.iter_split("val", epoch=0),
                config.eval.task_offset,
                config.eval.task_offset + limit,
            )
            prompts = [render_prompt(record, self.tokenizer, config.source) for record in records]
        if not prompts:
            return {
                "tag": tag,
                "split": "val",
                "tasks": 0,
                "success_rate": None,
                "reward_status": "not_applicable_pure_opd",
                "note": "no validation prompts",
            }
        model = getattr(self.student, "model", None)
        was_training = getattr(model, "training", None)
        if not isinstance(was_training, bool):
            raise BackendError("the student backend does not expose train/eval mode")
        self.student.set_train(False)
        runtime = PromptDatasetRolloutRuntime(
            backend=self.student,
            source_config=config.source,
            rollout_config=config.rollout.model_copy(
                update={"temperature": config.eval.temperature}
            ),
        )
        try:
            gpu.reset_peak_stats()
            started = time.perf_counter()
            prepared = runtime.prepare_batch(prompts)
            generated = runtime.generate(
                prepared,
                policy_version=self.policy_version,
                seed=config.eval.seed + config.eval.task_offset,
            )
            trajectories = runtime.to_trajectories(
                prepared,
                generated,
                policy_version=self.policy_version,
            )
            elapsed = max(time.perf_counter() - started, 1e-9)
            append_trajectories(self.paths.eval_trajectories, trajectories)
            generated_tokens = sum(item.generated_token_count for item in trajectories)
            payload = {
                "tag": tag,
                "split": "val",
                "tasks": len(trajectories),
                "policy_version": self.policy_version,
                "parameter_version": self.parameter_version,
                "global_step": self.global_step,
                "global_optimizer_step": self.global_step,
                "rollout_iteration": self._cycles_completed,
                "rollout_policy_version": self._last_rollout_policy_version,
                "temperature": config.eval.temperature,
                "task_offset": config.eval.task_offset,
                "seconds": round(elapsed, 3),
                "generated_tokens": generated_tokens,
                "rollout_tokens_per_second": round(generated_tokens / elapsed, 2),
                "physical_batch_sizes": list(generated.physical_batch_sizes),
                "oom_downshifts": generated.oom_downshifts,
                "success_rate": None,
                "reward_status": "not_applicable_pure_opd",
                "measurement_status": {
                    "response_generation": "measured",
                    "task_reward": "not_configured",
                    "task_success": "not_measured",
                },
                "memory": gpu.snapshot().to_dict(),
            }
            if write:
                self.metrics_log.write({"phase": "eval", **payload, "ts": utc_now()})
            self.events.emit(
                "eval",
                tag=tag,
                tasks=len(trajectories),
                success_rate=None,
                reward_status="not_applicable_pure_opd",
            )
            return payload
        finally:
            runtime.close()
            self.student.set_train(was_training)

    # -- checkpointing -----------------------------------------------------------

    def _config_digest(self) -> str:
        return hashlib.sha256(self.config.training_identity_yaml().encode("utf-8")).hexdigest()

    def _resolved_config_digest(self) -> str:
        if self.paths.config_resolved.is_file():
            return hashlib.sha256(self.paths.config_resolved.read_bytes()).hexdigest()
        return self._config_digest()

    def _checkpoint_identity(self) -> dict[str, Any]:
        student = self.config.models.student
        return {
            "backend": self.config.models.backend.value,
            "student_model_id": student.model_id,
            "student_revision": student.revision,
            "tokenizer_identity": student.tokenizer_id or student.model_id,
            "tokenizer_revision": student.tokenizer_revision or student.revision,
            "lora": student.lora.model_dump(mode="json"),
            "execution_plan_digest": self.config.run.execution_plan_digest,
            "profile_identity": self.config.run.profile_identity,
        }

    def save_checkpoint(self, *, name: str | None = None) -> Path:
        """Write a resumable checkpoint when training does not own the model."""
        self._ensure_state(
            "save_checkpoint",
            TrainerState.READY,
            TrainerState.COMPLETED,
        )
        if not self._operation_guard.acquire(blocking=False):
            with self._state_guard:
                state = self._state
            raise LifecycleError(
                f"cannot save_checkpoint: this OPDTrainer is {state.value}",
                hint="wait for the active trainer operation to finish",
            )
        try:
            self._ensure_state(
                "save_checkpoint",
                TrainerState.READY,
                TrainerState.COMPLETED,
            )
            return self._save_checkpoint_impl(name=name)
        finally:
            self._operation_guard.release()

    def _save_checkpoint_impl(self, *, name: str | None = None) -> Path:
        """Write a checkpoint while the caller owns the trainer operation."""
        label = name or f"step-{self.global_step:06d}"
        target = self.paths.checkpoints / label
        state = CheckpointState(
            miniverl_version=__version__,
            global_step=self.global_step,
            policy_version=self.policy_version,
            parameter_version=self.parameter_version,
            cycle=self.cycle,
            rollout_iteration=self._cycles_completed,
            rollout_policy_version=self._last_rollout_policy_version,
            task_cursor=self.task_cursor,
            scheduler=self.schedule.state_dict(),
            config_digest=self._config_digest(),
            resolved_config_digest=self._resolved_config_digest(),
            execution_plan_digest=self.config.run.execution_plan_digest or "",
            profile_identity=self.config.run.profile_identity,
            offline_dataset_digest=self.offline_dataset_digest,
        )
        save_checkpoint(
            target,
            trainable_state=self.student.trainable_state_dict(),
            optimizer=self.optimizer,
            state=state,
            rng=capture_rng(),
            identity=self._checkpoint_identity(),
        )
        self.events.emit("checkpoint_saved", path=str(target), step=self.global_step)
        return target

    def load_from_checkpoint(self, directory: str | Path) -> CheckpointState:
        """Restore a checkpoint only while the trainer is ready."""
        self._ensure_state("load_from_checkpoint", TrainerState.READY)
        if not self._operation_guard.acquire(blocking=False):
            with self._state_guard:
                state = self._state
            raise LifecycleError(
                f"cannot load_from_checkpoint: this OPDTrainer is {state.value}",
                hint="wait for the active trainer operation to finish",
            )
        try:
            self._ensure_state("load_from_checkpoint", TrainerState.READY)
            return self._load_from_checkpoint_impl(directory)
        finally:
            self._operation_guard.release()

    def _load_from_checkpoint_impl(self, directory: str | Path) -> CheckpointState:
        """Restore a checkpoint while the caller owns the trainer operation."""
        from miniverl.training.checkpoint import validate_checkpoint

        validated = validate_checkpoint(directory)
        digest = self._config_digest()
        expected_plan_digest = self.config.run.execution_plan_digest or ""
        if validated.state.execution_plan_digest != expected_plan_digest:
            raise ConfigError(
                "the checkpoint was written by a different immutable execution plan",
                hint="resume using the exact plan.json recorded by the original run",
            )
        if validated.state.config_digest and validated.state.config_digest != digest:
            raise ConfigError(
                "the checkpoint was written by a different configuration",
                hint="resume with the run's config.original.yaml compatibility layer, "
                "not config.resolved.yaml or a modified recipe",
            )
        identity = self._checkpoint_identity()
        if validated.identity:
            mismatches = {
                key: (validated.identity.get(key), value)
                for key, value in identity.items()
                if validated.identity.get(key) != value
            }
            if mismatches:
                details = ", ".join(
                    f"{key}: checkpoint={actual!r}, current={expected!r}"
                    for key, (actual, expected) in sorted(mismatches.items())
                )
                raise ConfigError(f"checkpoint model/tokenizer identity mismatch ({details})")
        if self.config.run.mode is TrainingMode.OFFLINE_KD:
            if not validated.state.offline_dataset_digest:
                raise ConfigError(
                    "offline-KD checkpoint has no persisted dataset identity",
                    hint="restart this legacy run; exact offline-KD resume requires v0.2.1 artifacts",
                )
            self._load_offline_dataset(
                expected_digest=validated.state.offline_dataset_digest,
            )
        state = load_checkpoint(
            directory,
            backend=self.student,
            optimizer=self.optimizer,
            device=self.student.device,
            expected_config_digest=digest,
            expected_identity=identity if validated.identity else None,
        )
        self._apply_checkpoint_progress(state)
        self._start_cycle = self._cycles_completed
        self._resumed = True
        self._resumed_from = {
            "directory": Path(directory).name,
            "global_step": validated.state.global_step,
            "digest": validated.content_digest,
            "integrity": validated.integrity,
        }
        if state.scheduler:
            self.schedule = LearningRateSchedule.from_state_dict(state.scheduler)
        self.events.emit("checkpoint_loaded", path=str(directory), step=self.global_step)
        return state

    def close(self) -> None:
        """Destructively release every resource owned by this trainer.

        The first call attempts every cleanup stage even if one fails. Later
        calls are no-ops. Core provenance such as ``config`` and ``paths`` stays
        readable, while runtime objects are deliberately unusable.
        """
        if not self._operation_guard.acquire(blocking=False):
            raise LifecycleError(
                "cannot close while another trainer operation is active",
                hint=("wait for the active evaluate/checkpoint/load/train operation to finish"),
            )
        try:
            self._close_impl()
        finally:
            self._operation_guard.release()

    def _close_impl(self) -> None:
        """Release resources while the caller owns the trainer operation."""
        with self._state_guard:
            if self._state is TrainerState.CLOSED:
                return
            if self._state is TrainerState.RUNNING:
                raise LifecycleError(
                    "cannot close: this OPDTrainer is running",
                    hint="wait for train() to reach a terminal state before closing",
                )
            previous_state = self._state
            self._state = TrainerState.CLOSED
            self._closed = True
        failures: list[tuple[str, BaseException]] = []

        def cleanup(stage: str, action: Any) -> None:
            try:
                action()
            except BaseException as exc:
                failures.append((stage, exc))

        if previous_state is TrainerState.READY and not self._evaluation_only:

            def close_ready_manifest() -> None:
                import json

                if not self.paths.manifest.is_file():
                    return
                manifest = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
                if manifest.get("status") != "ready":
                    return
                manifest["status"] = "closed_before_training"
                manifest["closed_at"] = utc_now()
                write_json_atomic(self.paths.manifest, manifest)

            cleanup("ready manifest finalization", close_ready_manifest)

        cache = self._cache
        self._cache = None
        if cache is not None:
            cleanup("teacher cache flush", cache.flush)
        cache = None

        samples = self._offline_samples or []
        self._offline_samples = None
        for index in range(len(samples)):
            cleanup(
                f"teacher target release {index}",
                lambda owned_sample=samples[index]: setattr(owned_sample, "teacher", None),
            )
        samples.clear()

        # RolloutRunner and LocalTeacherScorer own strong backend references;
        # drop them before releasing the backends themselves.
        self.runner = None  # type: ignore[assignment]  # destructive close
        self.scorer = None
        self.role_graph = None  # type: ignore[assignment]  # destructive close

        optimizer = self.optimizer
        self.optimizer = None
        if optimizer is not None:
            owned_optimizer: Any = optimizer
            cleanup(
                "optimizer gradient clear",
                lambda: owned_optimizer.zero_grad(set_to_none=True),
            )

            def clear_optimizer() -> None:
                owned_optimizer.state.clear()
                for group in owned_optimizer.param_groups:
                    group["params"] = []
                owned_optimizer.param_groups.clear()

            cleanup("optimizer state clear", clear_optimizer)
        optimizer = None

        reference = self.reference
        self.reference = None
        if reference is not None:
            cleanup("reference release", reference.release)
        reference = None

        teacher = self.teacher
        self.teacher = None
        if teacher is not None:
            cleanup("teacher release", teacher.release)
        teacher = None

        student = self.student
        self.student = None
        if student is not None:
            cleanup("student release", student.release)
        student = None

        rollout_runtime = self.rollout_runtime
        self.rollout_runtime = None  # type: ignore[assignment]  # destructive close
        if rollout_runtime is not None:
            cleanup("rollout runtime close", rollout_runtime.close)
        del rollout_runtime

        environment = self.environment
        self.environment = None  # type: ignore[assignment]  # destructive close
        if environment is not None:
            closer = getattr(environment, "close", None)
            if callable(closer):
                cleanup("environment close", closer)
        del environment
        closer = None

        self.metrics_log = None  # type: ignore[assignment]  # destructive close
        self.events = None  # type: ignore[assignment]  # destructive close
        self._teacher_on_device = False
        cleanup("Python garbage collection", gc.collect)
        cleanup("CUDA allocator release", gpu.empty_cache)
        run_lock = self._run_lock
        self._run_lock = None
        if run_lock is not None:
            cleanup("run lock release", run_lock.release)

        if failures:
            details = "; ".join(f"{stage}: {type(exc).__name__}: {exc}" for stage, exc in failures)
            raise LifecycleError(
                f"trainer cleanup failed after all teardown stages were attempted: {details}",
                hint=(
                    "the trainer is closed and cannot be reused; inspect the first cleanup "
                    "error and start a fresh trainer"
                ),
            )

    def _ensure_state(self, operation: str, *allowed: TrainerState) -> None:
        with self._state_guard:
            state = self._state
        if state not in allowed:
            raise LifecycleError(
                f"cannot {operation}: this OPDTrainer is {state.value}",
                hint=(
                    "construct a fresh trainer with OPDTrainer.from_config(...) "
                    "or use an operation allowed by the current lifecycle state"
                ),
            )

    def __enter__(self) -> OPDTrainer:
        self._ensure_state("enter", TrainerState.READY)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        try:
            self.close()
        except LifecycleError as cleanup_error:
            if exc_type is None:
                raise
            logger.warning(
                "trainer cleanup also failed while preserving %s: %s",
                exc_type.__name__,
                cleanup_error,
            )


def _raise_config(message: str, hint: str | None = None) -> None:  # pragma: no cover - helper
    raise MiniVerlError(message, hint)
