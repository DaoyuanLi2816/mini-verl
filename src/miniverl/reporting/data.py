"""Assemble report data from run artifacts.

Torch-free on purpose: ``miniverl report`` reads only JSON, JSONL and YAML, so
it works on a run directory copied from a GPU machine to a laptop with nothing
but the base install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miniverl.errors import ReportError
from miniverl.schemas.trajectory import Trajectory
from miniverl.trajectory.io import iter_trajectories
from miniverl.utils.privacy import portable_payload, portable_text, portable_yaml
from miniverl.utils.runs import RunPaths, read_json, read_jsonl

__all__ = ["ReportData", "TrajectoryView"]


@dataclass
class TrajectoryView:
    """A trajectory rendered for display."""

    trajectory_id: str
    task_id: str
    termination_reason: str
    solved: bool | None
    reward: float | None
    turns: int
    tokens: int
    model_tokens: int
    critical_tokens: int
    spans: list[dict[str, Any]] = field(default_factory=list)
    tokens_by_span_type: dict[str, int] = field(default_factory=dict)
    expected: str | None = None
    predicted: str | None = None
    failure_category: str | None = None


@dataclass
class ReportData:
    """Everything the HTML/Markdown renderers need."""

    run_id: str
    run_dir: Path
    manifest: dict[str, Any]
    environment: dict[str, Any]
    resolved_config: str
    original_config: str
    summary: dict[str, Any]
    step_metrics: list[dict[str, Any]] = field(default_factory=list)
    cycle_metrics: list[dict[str, Any]] = field(default_factory=list)
    eval_metrics: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    trajectories: list[TrajectoryView] = field(default_factory=list)
    token_analysis: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    cache_stats: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None

    # -- loading ---------------------------------------------------------

    @classmethod
    def from_run(
        cls, run_dir: str | Path, *, max_trajectories: int = 5, max_tokens: int = 400
    ) -> ReportData:
        """Read a run directory into a report model."""
        paths = RunPaths.open(run_dir)
        manifest = portable_payload(read_json(paths.manifest))
        environment = (
            portable_payload(read_json(paths.environment)) if paths.environment.is_file() else {}
        )
        summary = portable_payload(read_json(paths.eval_json)) if paths.eval_json.is_file() else {}
        metrics = portable_payload(read_jsonl(paths.metrics))
        events = portable_payload(read_jsonl(paths.events))

        step_metrics = [
            m for m in metrics if m.get("phase") in {"sft", "offline_kd", "opd", "sft_warmup"}
        ]
        cycle_metrics = [m for m in metrics if str(m.get("phase", "")).endswith("_cycle")]
        eval_metrics = [m for m in metrics if m.get("phase") == "eval"]

        token_analysis = cls._load_token_analysis(paths.root / "token_analysis.jsonl", max_tokens)
        trajectories = cls._load_trajectories(
            paths, limit=max_trajectories, prefer_ids=set(token_analysis)
        )

        cache_stats = None
        if paths.teacher_cache.is_dir() and (paths.teacher_cache / "index.json").is_file():
            from miniverl.cache.stats import compute_stats

            try:
                cache_stats = portable_payload(
                    compute_stats(paths.teacher_cache, verify_checksums=True)
                )
            except Exception as exc:
                cache_stats = {"error": portable_text(str(exc))}

        benchmark = None
        if paths.benchmark_json.is_file():
            benchmark = portable_payload(read_json(paths.benchmark_json))

        validated_path = (
            paths.config_validated if paths.config_validated.is_file() else paths.config_original
        )

        return cls(
            run_id=str(manifest.get("run_id", paths.root.name)),
            run_dir=paths.root,
            manifest=manifest,
            environment=environment,
            resolved_config=(
                portable_yaml(paths.config_resolved.read_text(encoding="utf-8"))
                if paths.config_resolved.is_file()
                else ""
            ),
            original_config=(
                portable_yaml(validated_path.read_text(encoding="utf-8"))
                if validated_path.is_file()
                else ""
            ),
            summary=summary,
            step_metrics=step_metrics,
            cycle_metrics=cycle_metrics,
            eval_metrics=eval_metrics,
            events=events,
            trajectories=trajectories,
            token_analysis=token_analysis,
            cache_stats=cache_stats,
            benchmark=benchmark,
        )

    @staticmethod
    def _load_trajectories(
        paths: RunPaths, *, limit: int, prefer_ids: set[str] | None = None
    ) -> list[TrajectoryView]:
        """Load trajectories to display.

        Trajectories with per-token analysis come first (they are the ones whose
        token-level divergence view can be rendered); the rest of the budget is
        filled with the most recent evaluation rollouts, which reflect the
        trained policy.
        """
        if limit <= 0:
            return []
        wanted = prefer_ids or set()
        preferred: list[TrajectoryView] = []
        evaluation: list[TrajectoryView] = []
        training: list[TrajectoryView] = []
        for source, bucket in (
            (paths.trajectories, training),
            (paths.eval_trajectories, evaluation),
        ):
            if not source.is_file():
                continue
            for traj in iter_trajectories(source):
                view = ReportData._view_of(traj)
                if traj.trajectory_id in wanted:
                    preferred.append(view)
                else:
                    bucket.append(view)
        chosen = preferred[:limit]
        seen = {view.trajectory_id for view in chosen}
        # Evaluation rollouts reflect the trained policy, so they fill the budget
        # first; training rollouts are the fallback when there are no eval ones.
        # Within a bucket the *most recent* rollouts are taken, but they are then
        # presented in their original order so the report reads chronologically.
        for bucket in (evaluation, training):
            room = limit - len(chosen)
            if room <= 0:
                break
            fresh = [v for v in bucket if v.trajectory_id not in seen]
            for view in fresh[-room:]:
                seen.add(view.trajectory_id)
                chosen.append(view)
        return chosen

    @staticmethod
    def _view_of(traj: Trajectory) -> TrajectoryView:
        """Build a display view from a validated trajectory."""
        spans = [
            {
                "span_type": span.span_type.value,
                "start": span.start,
                "end": span.end,
                "turn_id": span.turn_id,
                "tokens": span.length,
                "model_generated": span.is_model_generated,
                "critical": span.is_critical,
                "tool_name": span.tool_name,
                "text": portable_text(span.text),
            }
            for span in traj.spans
        ]
        return TrajectoryView(
            trajectory_id=portable_text(traj.trajectory_id),
            task_id=portable_text(traj.task_id),
            termination_reason=traj.termination_reason.value,
            solved=(traj.verification.solved if traj.verification else None),
            reward=(traj.verification.reward if traj.verification else None),
            turns=len(traj.turns),
            tokens=traj.length,
            model_tokens=sum(traj.model_generated_mask),
            critical_tokens=sum(traj.critical_mask),
            spans=spans,
            tokens_by_span_type=traj.token_counts_by_span_type(),
            expected=(
                portable_text(traj.verification.expected)
                if traj.verification and traj.verification.expected is not None
                else None
            ),
            predicted=(
                portable_text(traj.verification.predicted)
                if traj.verification and traj.verification.predicted is not None
                else None
            ),
            failure_category=(traj.verification.failure_category if traj.verification else None),
        )

    @staticmethod
    def _load_token_analysis(path: Path, max_tokens: int) -> dict[str, list[dict[str, Any]]]:
        if not path.is_file():
            return {}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in read_jsonl(path):
            record = portable_payload(record)
            key = str(record.get("trajectory_id", "?"))
            bucket = grouped.setdefault(key, [])
            if len(bucket) < max_tokens:
                bucket.append(record)
        return grouped

    # -- derived views ----------------------------------------------------

    @property
    def mode(self) -> str:
        """Training mode of the run."""
        return str(self.manifest.get("mode", "unknown"))

    @property
    def is_on_policy(self) -> bool:
        """``True`` only for genuine OPD."""
        return self.mode == "opd"

    def loss_series(self) -> list[tuple[str, list[float], list[float]]]:
        """Loss curves grouped by training phase."""
        by_phase: dict[str, tuple[list[float], list[float]]] = {}
        for record in self.step_metrics:
            phase = str(record.get("phase", "train"))
            xs, ys = by_phase.setdefault(phase, ([], []))
            xs.append(float(record.get("step", len(xs))))
            ys.append(float(record.get("loss", 0.0)))
        return [(phase, xs, ys) for phase, (xs, ys) in sorted(by_phase.items())]

    def eval_series(self) -> list[tuple[str, list[float], list[float]]]:
        """Success rate against optimizer step."""
        xs = [float(m.get("global_step", 0)) for m in self.eval_metrics]
        ys = [float(m.get("success_rate", 0.0)) for m in self.eval_metrics]
        if not xs:
            return []
        return [("task success rate", xs, ys)]

    def failure_counts(self) -> list[tuple[str, float]]:
        """Failure taxonomy from the most recent evaluation."""
        if not self.eval_metrics:
            return []
        categories = self.eval_metrics[-1].get("failure_categories") or {}
        return sorted(((str(k), float(v)) for k, v in categories.items()), key=lambda kv: -kv[1])

    def termination_counts(self) -> list[tuple[str, float]]:
        """Termination reasons from the most recent evaluation."""
        if not self.eval_metrics:
            return []
        reasons = self.eval_metrics[-1].get("termination_reasons") or {}
        return sorted(((str(k), float(v)) for k, v in reasons.items()), key=lambda kv: -kv[1])

    def selection_counts(self) -> list[tuple[str, float]]:
        """Selected teacher positions by span type."""
        if not self.cycle_metrics:
            return []
        by_span = (self.cycle_metrics[-1].get("selection") or {}).get("selected_by_span_type") or {}
        return sorted(((str(k), float(v)) for k, v in by_span.items()), key=lambda kv: -kv[1])

    def baseline_comparison(self) -> list[dict[str, Any]]:
        """Baseline vs final evaluation table."""
        rows = []
        for key in ("baseline_eval", "eval"):
            payload = self.summary.get(key)
            if isinstance(payload, dict) and payload.get("tasks"):
                rows.append(
                    {
                        "label": "before training" if key == "baseline_eval" else "after training",
                        "tag": payload.get("tag"),
                        "policy_version": payload.get("policy_version"),
                        "global_step": payload.get("global_step"),
                        "tasks": payload.get("tasks"),
                        "success_rate": payload.get("success_rate"),
                        "avg_turns": payload.get("avg_turns"),
                        "parse_valid_tool_call_rate": payload.get(
                            "parse_valid_tool_call_rate",
                            payload.get("valid_tool_call_rate"),
                        ),
                        "tool_execution_success_rate": payload.get("tool_execution_success_rate"),
                        "generated_tokens_per_task": payload.get("generated_tokens_per_task"),
                        "rollout_tokens_per_second": payload.get("rollout_tokens_per_second"),
                    }
                )
        return rows

    def throughput(self) -> dict[str, Any]:
        """Throughput and memory highlights."""
        train_rates = [
            float(m["train_selected_tokens_per_second"])
            for m in self.step_metrics
            if m.get("train_selected_tokens_per_second")
        ]
        peaks_alloc = [
            float((m.get("memory") or {}).get("peak_allocated_bytes") or 0)
            for m in self.step_metrics
        ]
        peaks_reserved = [
            float((m.get("memory") or {}).get("peak_reserved_bytes") or 0)
            for m in self.step_metrics
        ]
        rollout_rates = [
            float(e["rollout_tokens_per_second"])
            for e in self.events
            if e.get("event") == "rollouts_collected" and e.get("rollout_tokens_per_second")
        ]
        # What matters is whether *this run* used CUDA, not whether the machine
        # happens to have a GPU. A CPU run on a GPU box must report "not
        # measured", never a reassuring 0.000 GiB.
        device = str(((self.manifest.get("models") or {}).get("device")) or "")
        cuda = device.startswith("cuda") and bool(
            (self.environment.get("gpu") or {}).get("available")
        )
        return {
            "cuda_available": cuda,
            "train_selected_tokens_per_second_mean": (
                sum(train_rates) / len(train_rates) if train_rates else None
            ),
            "rollout_tokens_per_second_mean": (
                sum(rollout_rates) / len(rollout_rates) if rollout_rates else None
            ),
            "peak_allocated_gib": (max(peaks_alloc) / 1024**3) if cuda and peaks_alloc else None,
            "peak_reserved_gib": (max(peaks_reserved) / 1024**3)
            if cuda and peaks_reserved
            else None,
            "optimizer_steps": len(self.step_metrics),
            "wall_clock_seconds": self.summary.get("duration_seconds"),
        }

    def validate(self) -> None:
        """Fail loudly if the run directory is unusable for a report."""
        if not self.manifest:
            raise ReportError(
                f"{self.run_dir} has an empty manifest.json",
                hint="the run may have crashed during startup; check events.jsonl",
            )
