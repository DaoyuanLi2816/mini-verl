"""End-to-end, provenance-first alignment workflow orchestration."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from miniverl.alignment.card import render_alignment_card
from miniverl.alignment.evaluation import alignment_metrics
from miniverl.alignment.schema import AlignmentConfig, AlignmentMethod
from miniverl.errors import ConfigError
from miniverl.trajectory.io import read_trajectories
from miniverl.utils.runs import read_json, read_jsonl, write_json_atomic

if TYPE_CHECKING:  # pragma: no cover
    from miniverl.config.models import RunConfig
    from miniverl.training.trainer import OPDTrainer, TrainResult

__all__ = [
    "build_alignment_stage_plan",
    "load_alignment_starting_checkpoint",
    "load_alignment_method_adapter",
    "publish_alignment_artifacts",
    "run_alignment",
]


def build_alignment_stage_plan(
    alignment: AlignmentConfig,
    *,
    sft_warmup_cycles: int,
) -> dict[str, Any]:
    """Return the six explicit stages recorded by every alignment run."""
    if alignment.starting_sft_checkpoint is not None:
        checkpoint_source = "imported_checkpoint"
        checkpoint_id = Path(alignment.starting_sft_checkpoint).name
    else:
        checkpoint_source = "embedded_sft_warmup"
        checkpoint_id = f"{sft_warmup_cycles}_cycles"
    return {
        "schema_version": 1,
        "method": alignment.method.value,
        "teacher_mode": alignment.teacher_mode.value if alignment.teacher_mode else None,
        "stages": [
            {"name": "base_model", "source": "recipe_model_identity"},
            {
                "name": "sft_checkpoint",
                "source": checkpoint_source,
                "id": checkpoint_id,
                "sha256": alignment.starting_sft_checkpoint_sha256,
            },
            {
                "name": "teacher_reference_construction",
                "teacher_mode": (alignment.teacher_mode.value if alignment.teacher_mode else None),
                "reference": (
                    alignment.reference.model_dump(mode="json") if alignment.reference else None
                ),
                "dpo": alignment.dpo.model_dump(mode="json") if alignment.dpo else None,
            },
            {
                "name": "alignment",
                "method": alignment.method.value,
                "gate": alignment.gate.model_dump(mode="json") if alignment.gate else None,
            },
            {
                "name": "evaluation",
                "adapters": list(alignment.evaluation_adapters),
                "primary": "deterministic_exact_verifier",
            },
            {"name": "alignment_card", "formats": ["json", "markdown"]},
        ],
    }


def load_alignment_starting_checkpoint(trainer: OPDTrainer) -> dict[str, Any] | None:
    """Load only starting adapter weights, never optimizer, progress or RNG state."""
    alignment = trainer.config.alignment
    if alignment is None or alignment.starting_sft_checkpoint is None:
        return None
    from miniverl.training.checkpoint import load_checkpoint, validate_checkpoint

    checkpoint = Path(alignment.starting_sft_checkpoint)
    validated = validate_checkpoint(checkpoint)
    declared = alignment.starting_sft_checkpoint_sha256
    if declared is not None and validated.content_digest != declared:
        raise ConfigError(
            "starting SFT checkpoint digest does not match alignment configuration",
            hint=(
                f"declared {declared}, validated {validated.content_digest}; do not continue "
                "from an unregistered checkpoint"
            ),
        )
    identity = trainer._checkpoint_identity()
    load_checkpoint(
        checkpoint,
        backend=trainer.student,
        optimizer=None,
        device=trainer.student.device,
        include_optimizer=False,
        include_rng=False,
        expected_identity=identity if validated.identity else None,
    )
    payload = {
        "id": checkpoint.name,
        "sha256": validated.content_digest,
        "integrity": validated.integrity,
        "global_step_ignored": validated.state.global_step,
    }
    trainer.events.emit(
        "alignment_starting_checkpoint_loaded",
        checkpoint_id=payload["id"],
        sha256=payload["sha256"],
        integrity=payload["integrity"],
        global_step_ignored=payload["global_step_ignored"],
    )
    return payload


def load_alignment_method_adapter(trainer: OPDTrainer) -> dict[str, Any] | None:
    """Load a pinned TRL DPO PEFT adapter into the already constructed actor role."""
    alignment = trainer.config.alignment
    if alignment is None or alignment.dpo is None:
        return None
    if not alignment.dpo_adapter_path:  # pragma: no cover - schema guard
        raise ConfigError("DPO evaluation requires alignment.dpo_adapter_path")
    adapter_dir = Path(alignment.dpo_adapter_path)
    weights = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not weights.is_file() or not config_path.is_file():
        raise ConfigError(
            f"DPO adapter is incomplete: {adapter_dir}",
            hint="expected adapter_config.json and adapter_model.safetensors",
        )
    actual = _sha256(weights)
    expected = alignment.dpo.adapter.sha256
    if actual != expected:
        raise ConfigError(
            "DPO adapter weights digest does not match alignment.dpo.adapter",
            hint=f"declared {expected}, measured {actual}",
        )
    model = getattr(trainer.student, "model", None)
    if model is None:
        raise ConfigError("DPO adapter evaluation requires a PEFT-backed HF student")
    from safetensors.torch import load_file

    from miniverl.utils.lazy import require_peft

    state = load_file(str(weights), device="cpu")
    adapter_name = (
        "student" if trainer.config.models.runtime.value == "shared_backbone" else "default"
    )
    result = require_peft("Loading the pinned DPO baseline adapter").set_peft_model_state_dict(
        model,
        state,
        adapter_name=adapter_name,
    )
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    missing = [
        key
        for key in list(getattr(result, "missing_keys", []) or [])
        if "lora_" in key and f".{adapter_name}." in key
    ]
    if unexpected or missing:
        raise ConfigError(
            "DPO adapter parameters do not match the configured actor LoRA",
            hint=f"unexpected={unexpected[:1]}, missing={missing[:1]}",
        )
    payload = {
        "id": alignment.dpo.adapter.id,
        "revision": alignment.dpo.adapter.revision,
        "weights_sha256": actual,
        "trl_version": alignment.dpo.trl_version,
    }
    trainer.events.emit(
        "alignment_dpo_adapter_loaded",
        adapter_id=payload["id"],
        revision=payload["revision"],
        weights_sha256=payload["weights_sha256"],
        trl_version=payload["trl_version"],
    )
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tagged(rows: list[Any], tag: str) -> list[Any]:
    marker = f":{tag}:v"
    return [row for row in rows if marker in row.trajectory_id]


def _decision_distribution(rows: list[Any]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for row in rows:
        predicted = str((row.verification.predicted if row.verification else "") or "UNKNOWN")
        key = predicted.strip().upper() or "UNKNOWN"
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())} if total else {}


def _jsd(left: Mapping[str, float], right: Mapping[str, float]) -> float | None:
    if not left or not right:
        return None
    keys = set(left) | set(right)
    midpoint = {key: (left.get(key, 0.0) + right.get(key, 0.0)) / 2 for key in keys}

    def kl(values: Mapping[str, float]) -> float:
        return sum(
            value * math.log2(value / midpoint[key])
            for key, value in values.items()
            if value > 0 and midpoint[key] > 0
        )

    return (kl(left) + kl(right)) / 2


def _teacher_identity(config: RunConfig) -> dict[str, Any] | None:
    if config.alignment is None or config.alignment.teacher_mode is None:
        return None
    adapter = config.models.teacher.adapter
    return {
        "id": config.models.teacher.model_id,
        "revision": config.models.teacher.revision,
        "mode": config.alignment.teacher_mode.value,
        "adapter_revision": adapter.revision if adapter is not None else None,
    }


def _reference_identity(config: RunConfig) -> dict[str, Any] | None:
    reference = config.alignment.reference if config.alignment else None
    return reference.model_dump(mode="json") if reference is not None else None


def _training_measurements(
    trainer: OPDTrainer,
    method: AlignmentMethod,
) -> tuple[float, int | None, float | None, int | None]:
    """Aggregate continuation cost and actual teacher-query work from JSONL evidence."""
    records = read_jsonl(trainer.paths.metrics)
    cycle_rows = [row for row in records if str(row.get("phase", "")).endswith("_cycle")]
    continuation_seconds = sum(float(row.get("seconds") or 0.0) for row in cycle_rows)
    peak_vram = max(
        (int((row.get("memory") or {}).get("peak_allocated_bytes") or 0) for row in records),
        default=0,
    )

    queried: int | None = None
    query_ratio: float | None = None
    teacher_methods = {
        AlignmentMethod.OFFLINE_DISTILLATION,
        AlignmentMethod.STANDARD_OPD,
        AlignmentMethod.VERIFIER_GATED_OPD,
    }
    if method in teacher_methods:
        query_rows = cycle_rows
        if method is AlignmentMethod.OFFLINE_DISTILLATION:
            # Offline targets are constructed once and then reused. Repeated
            # optimizer passes over the cache are not additional teacher calls.
            query_rows = cycle_rows[:1]
        selections = [row.get("selection") or {} for row in query_rows]
        queried = sum(int(row.get("selected_model_tokens") or 0) for row in selections)
        candidates = sum(int(row.get("total_model_tokens") or 0) for row in selections)
        query_ratio = queried / candidates if candidates else None

    return continuation_seconds, queried, query_ratio, peak_vram or None


def publish_alignment_artifacts(
    trainer: OPDTrainer,
    result: TrainResult,
    *,
    starting_checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create the hash-bound Alignment Card and attach it to the run manifest."""
    config = trainer.config
    alignment = config.alignment
    if alignment is None:  # pragma: no cover - guarded by the CLI and workflow
        raise ConfigError("alignment workflow requires an alignment section")

    rows = read_trajectories(trainer.paths.eval_trajectories)
    baseline_rows = _tagged(rows, "baseline")
    final_rows = _tagged(rows, "final")
    baseline_metrics = alignment_metrics(baseline_rows)
    metrics = alignment_metrics(final_rows)
    continuation_seconds, queried, query_ratio, peak = _training_measurements(
        trainer,
        alignment.method,
    )
    shift = _jsd(
        _decision_distribution(baseline_rows),
        _decision_distribution(final_rows),
    )
    metrics = metrics.model_copy(
        update={
            "teacher_queried_positions": queried,
            "teacher_query_ratio": query_ratio,
            "gpu_seconds": continuation_seconds if trainer.plan.device.startswith("cuda") else None,
            "peak_vram_bytes": int(peak) if peak is not None else None,
            "decision_distribution_shift_jsd": shift,
        }
    )

    artifact_hashes = {
        "eval": _sha256(trainer.paths.eval_json),
        "eval_trajectories": _sha256(trainer.paths.eval_trajectories),
    }
    starting = starting_checkpoint or {
        "id": f"embedded-sft-warmup-{config.train.sft_warmup_cycles}",
        "sha256": None,
        "integrity": "same_run_stage",
    }
    cost = {
        "wall_seconds": result.duration_seconds,
        "gpu_seconds": metrics.gpu_seconds,
        "peak_vram_bytes": metrics.peak_vram_bytes,
        "optimizer_updates": result.global_step,
    }
    card_path = trainer.paths.root / "alignment-card.md"
    card = render_alignment_card(
        card_path,
        method=alignment.method,
        starting_checkpoint=starting,
        teacher=_teacher_identity(config),
        reference=_reference_identity(config),
        policy=alignment.policy.model_dump(mode="json"),
        metrics=metrics,
        cost=cost,
        teacher_query_ratio=query_ratio,
        artifact_hashes=artifact_hashes,
        limitations=alignment.limitations,
    )
    payload = {
        "schema_version": 1,
        "workflow": build_alignment_stage_plan(
            alignment,
            sft_warmup_cycles=config.train.sft_warmup_cycles,
        ),
        "starting_checkpoint": starting,
        "baseline_metrics": baseline_metrics.model_dump(mode="json"),
        "final_metrics": metrics.model_dump(mode="json"),
        "card": {"markdown": card_path.name, "json": "alignment-card.json", **card},
    }
    alignment_path = trainer.paths.root / "alignment.json"
    write_json_atomic(alignment_path, payload)
    manifest = read_json(trainer.paths.manifest)
    if not isinstance(manifest, dict):  # pragma: no cover
        raise ConfigError("run manifest is not a JSON object")
    manifest["alignment_workflow"] = payload["workflow"]
    manifest["alignment_result"] = {
        "file": alignment_path.name,
        "sha256": _sha256(alignment_path),
        "card_sha256": card["card_sha256"],
    }
    write_json_atomic(trainer.paths.manifest, manifest)
    trainer.events.emit(
        "alignment_card_written",
        card=card_path.name,
        card_sha256=card["card_sha256"],
    )
    return payload


def run_alignment(
    config: RunConfig,
    *,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    local_files_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the configured alignment stages and return the published artifact summary."""
    if config.alignment is None:
        raise ConfigError("miniverl align requires a recipe with an alignment section")
    from miniverl.training.trainer import OPDTrainer

    with OPDTrainer.from_config(
        config,
        output_dir=output_dir,
        run_id=run_id,
        local_files_only=local_files_only,
        overwrite=overwrite,
    ) as trainer:
        starting = load_alignment_starting_checkpoint(trainer)
        method_adapter = load_alignment_method_adapter(trainer)
        result = trainer.train()
        alignment = publish_alignment_artifacts(
            trainer,
            result,
            starting_checkpoint=starting,
        )
        return {
            **result.to_dict(),
            "run_dir": str(trainer.paths.root),
            "alignment": alignment,
            "method_adapter": method_adapter,
        }
