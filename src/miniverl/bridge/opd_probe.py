"""Bounded CUDA calibration for immutable OPD execution plans."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from miniverl import __version__
from miniverl.config.models import RunConfig, VerlParquetSourceConfig
from miniverl.errors import ConfigError
from miniverl.utils.runs import canonical_json, write_json_atomic

__all__ = ["run_hardware_probe"]


def _digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _device_identity() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    try:
        driver = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.splitlines()[index]
            .strip()
        )
    except (FileNotFoundError, IndexError, OSError, subprocess.SubprocessError):
        driver = "unavailable"
    return {
        "name": properties.name,
        "uuid": str(uuid) if uuid is not None else None,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "driver": driver,
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
    }


def _identity(native: RunConfig, *, plan_digest: str, device: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(native.source, VerlParquetSourceConfig):
        raise ConfigError("OPD hardware probe requires a verl Parquet source")
    return {
        "schema_version": 1,
        "miniverl_version": __version__,
        "plan_digest": plan_digest,
        "device": device,
        "student": {
            "model_id": native.models.student.model_id,
            "revision": native.models.student.revision,
            "tokenizer_id": native.models.student.tokenizer_id,
            "tokenizer_revision": native.models.student.tokenizer_revision,
            "quantization": native.models.student.quantization.value,
            "dtype": native.models.student.dtype.value,
            "lora": native.models.student.lora.model_dump(mode="json"),
        },
        "teacher": {
            "model_id": native.models.teacher.model_id,
            "revision": native.models.teacher.revision,
            "tokenizer_id": native.models.teacher.tokenizer_id,
            "tokenizer_revision": native.models.teacher.tokenizer_revision,
            "quantization": native.models.teacher.quantization.value,
            "dtype": native.models.teacher.dtype.value,
            "adapter": (
                native.models.teacher.adapter.model_dump(mode="json")
                if native.models.teacher.adapter is not None
                else None
            ),
        },
        "token_bounds": {
            "max_prompt": native.source.max_prompt_length,
            "max_response": native.source.max_response_length,
            "max_total": native.rollout.max_total_tokens,
        },
        "top_k": native.loss.top_k,
    }


def _memory() -> dict[str, Any]:
    from miniverl.utils import gpu

    return gpu.snapshot().to_dict()


def _measure_probe(
    native: RunConfig,
    *,
    identity: dict[str, Any],
    offline: bool,
) -> dict[str, Any]:
    """Load roles sequentially and exercise inference/backward without an optimizer."""
    import torch

    from miniverl.models.factory import build_student, build_teacher, build_tokenizer
    from miniverl.utils import gpu

    started = time.perf_counter()
    baseline = _memory()
    tokenizer = build_tokenizer(native, local_files_only=offline)
    if not isinstance(native.source, VerlParquetSourceConfig):  # pragma: no cover - typed caller
        raise ConfigError("OPD hardware probe requires a verl Parquet source")
    prompt = "<|im_start|>user\nCompute 2 + 2.<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt)
    if len(ids) > native.source.max_prompt_length:
        ids = ids[-native.source.max_prompt_length :]
    if len(ids) < 2:
        raise ConfigError("probe prompt produced fewer than two tokens")
    phases: dict[str, Any] = {}
    failed: list[dict[str, Any]] = []
    student = None
    teacher = None
    hidden: Any = None
    logits: Any = None
    loss: Any = None
    outputs: Any = None
    rollout_ok = 0
    try:
        gpu.empty_cache()
        gpu.reset_peak_stats()
        phase_started = time.perf_counter()
        student = build_student(native, tokenizer, device="cuda", local_files_only=offline)
        phases["actor_static"] = {
            "seconds": time.perf_counter() - phase_started,
            "memory": _memory(),
        }
        for batch in sorted({1, 2, 4, native.rollout.prompt_batch_size}):
            try:
                gpu.reset_peak_stats()
                phase_started = time.perf_counter()
                outputs = student.generate_batch(
                    [ids] * batch,
                    max_new_tokens=min(2, native.rollout.max_new_tokens_per_turn),
                    temperature=0.0,
                )
                rollout_ok = batch
                phases[f"rollout_batch_{batch}"] = {
                    "seconds": time.perf_counter() - phase_started,
                    "generated_tokens": sum(len(item.token_ids) for item in outputs),
                    "memory": _memory(),
                }
            except BaseException as exc:
                if not gpu.is_oom_error(exc):
                    raise
                failed.append({"phase": "rollout", "batch_size": batch, "reason": "cuda_oom"})
                gpu.empty_cache()
                break
        positions = list(range(max(0, len(ids) - 2), len(ids) - 1))
        gpu.reset_peak_stats()
        phase_started = time.perf_counter()
        hidden = student.hidden_states_at(ids, positions, with_grad=True)
        logits = student.project(hidden).float()
        loss = logits.square().mean()
        loss.backward()
        phases["selected_position_backward"] = {
            "seconds": time.perf_counter() - phase_started,
            "positions": len(positions),
            "memory": _memory(),
        }
    finally:
        hidden = None
        logits = None
        loss = None
        outputs = None
        if student is not None:
            student.release()
            student = None
        gpu.empty_cache()
    try:
        gpu.reset_peak_stats()
        phase_started = time.perf_counter()
        teacher = build_teacher(native, tokenizer, device="cuda", local_files_only=offline)
        phases["teacher_static"] = {
            "seconds": time.perf_counter() - phase_started,
            "memory": _memory(),
        }
        gpu.reset_peak_stats()
        phase_started = time.perf_counter()
        with torch.no_grad():
            hidden = teacher.hidden_states_at(ids, [len(ids) - 2], with_grad=False)
            logits = teacher.project(hidden).float()
            torch.topk(logits, k=min(native.loss.top_k, logits.shape[-1]), dim=-1)
        phases["teacher_score_batch_1"] = {
            "seconds": time.perf_counter() - phase_started,
            "positions": 1,
            "memory": _memory(),
        }
    finally:
        hidden = None
        logits = None
        if teacher is not None:
            teacher.release()
            teacher = None
        gpu.empty_cache()
    released = _memory()
    release_allowance = 64 * 1024**2
    if released["allocated_bytes"] > baseline["allocated_bytes"] + release_allowance:
        raise ConfigError(
            "hardware probe did not release role memory back near its CUDA baseline",
            hint="close other references and retry; no training checkpoint was written",
        )
    return {
        "status": "measured",
        "identity": identity,
        "tokenizer_identity": getattr(tokenizer, "identity", {}),
        "measurements": {
            "phases": phases,
            "baseline": baseline,
            "after_release": released,
            "duration_seconds": round(time.perf_counter() - started, 4),
            "parameter_updates": 0,
            "checkpoint_published": False,
        },
        "recommendations": {
            "rollout_batch_size": max(1, rollout_ok),
            "teacher_score_batch_size": 1,
            "update_trajectory_batch_size": 1,
            "headroom_gib": native.memory.auto_swap_vram_headroom_gb,
            "basis": "bounded_probe",
        },
        "failed_candidates": failed,
    }


def run_hardware_probe(
    native: RunConfig,
    *,
    plan_digest: str,
    cache_dir: str | Path,
    offline: bool,
    force: bool = False,
) -> dict[str, Any]:
    """Return an exact-identity cached probe or perform one bounded measurement."""
    device = _device_identity()
    if device is None:
        raise ConfigError(
            "--probe requires one visible CUDA device",
            hint="run weight-free planning without --probe on CPU",
        )
    identity = _identity(native, plan_digest=plan_digest, device=device)
    key = _digest(identity)
    root = Path(cache_dir)
    target = root / f"{key}.json"
    if target.is_file() and not force:
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read cached hardware probe {target}: {exc}") from exc
        if payload.get("identity") != identity or payload.get("identity_digest") != key:
            raise ConfigError("hardware probe cache identity does not match its filename")
        if payload.get("probe_digest") != _digest(
            {k: v for k, v in payload.items() if k not in {"probe_digest", "cache"}}
        ):
            raise ConfigError("hardware probe cache digest mismatch")
        payload["cache"] = {"reused": True, "path": str(target)}
        return payload
    measured = _measure_probe(native, identity=identity, offline=offline)
    if measured.get("measurements", {}).get("parameter_updates") != 0:
        raise ConfigError("hardware probe updated parameters; refusing the result")
    payload = {**measured, "identity_digest": key}
    payload["probe_digest"] = _digest(payload)
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, payload)
    payload["cache"] = {"reused": False, "path": str(target)}
    return payload
