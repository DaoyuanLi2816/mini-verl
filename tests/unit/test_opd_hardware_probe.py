from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniverl.bridge.opd_runtime import compile_native_run_config
from miniverl.bridge.opd_v08 import load_verl_opd_v08_source
from miniverl.errors import ConfigError


def _native():
    return compile_native_run_config(load_verl_opd_v08_source("builtin:qwen3-0.6b-1.7b-opd"))


def test_probe_fails_before_model_loading_without_cuda(monkeypatch, tmp_path: Path) -> None:
    from miniverl.bridge import opd_probe

    monkeypatch.setattr(opd_probe, "_device_identity", lambda: None)
    with pytest.raises(ConfigError, match="CUDA"):
        opd_probe.run_hardware_probe(
            _native(), plan_digest="a" * 64, cache_dir=tmp_path, offline=True
        )


def test_probe_cache_is_identity_bound_and_reused(monkeypatch, tmp_path: Path) -> None:
    from miniverl.bridge import opd_probe

    device = {
        "name": "NVIDIA GeForce RTX 4080",
        "uuid": "GPU-test",
        "compute_capability": [8, 9],
        "total_memory_bytes": 16 * 1024**3,
        "driver": "test-driver",
        "torch": "test-torch",
        "cuda_runtime": "test-cuda",
    }
    monkeypatch.setattr(opd_probe, "_device_identity", lambda: device)
    calls = 0

    def measure(native, *, identity, offline):
        nonlocal calls
        calls += 1
        return {
            "status": "measured",
            "identity": identity,
            "measurements": {"parameter_updates": 0},
            "recommendations": {"rollout_batch_size": 1},
            "failed_candidates": [],
        }

    monkeypatch.setattr(opd_probe, "_measure_probe", measure)
    first = opd_probe.run_hardware_probe(
        _native(), plan_digest="a" * 64, cache_dir=tmp_path, offline=True
    )
    second = opd_probe.run_hardware_probe(
        _native(), plan_digest="a" * 64, cache_dir=tmp_path, offline=True
    )
    assert calls == 1
    assert first["cache"]["reused"] is False
    assert second["cache"]["reused"] is True
    assert second["probe_digest"] == first["probe_digest"]
    cached = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert cached["identity"]["plan_digest"] == "a" * 64


def test_probe_cache_miss_when_plan_or_token_bounds_change(monkeypatch, tmp_path: Path) -> None:
    from miniverl.bridge import opd_probe

    monkeypatch.setattr(
        opd_probe,
        "_device_identity",
        lambda: {
            "name": "GPU",
            "uuid": "GPU-x",
            "compute_capability": [8, 9],
            "total_memory_bytes": 1,
            "driver": "d",
            "torch": "t",
            "cuda_runtime": "c",
        },
    )
    calls = 0

    def measure(native, *, identity, offline):
        nonlocal calls
        calls += 1
        return {
            "status": "measured",
            "identity": identity,
            "measurements": {"parameter_updates": 0},
            "recommendations": {},
            "failed_candidates": [],
        }

    monkeypatch.setattr(opd_probe, "_measure_probe", measure)
    opd_probe.run_hardware_probe(_native(), plan_digest="a" * 64, cache_dir=tmp_path, offline=True)
    changed = _native()
    changed.source.max_response_length += 1
    opd_probe.run_hardware_probe(changed, plan_digest="b" * 64, cache_dir=tmp_path, offline=True)
    assert calls == 2


def test_probe_result_must_prove_zero_parameter_updates(monkeypatch, tmp_path: Path) -> None:
    from miniverl.bridge import opd_probe

    monkeypatch.setattr(
        opd_probe,
        "_device_identity",
        lambda: {
            "name": "GPU",
            "uuid": "GPU-x",
            "compute_capability": [8, 9],
            "total_memory_bytes": 1,
            "driver": "d",
            "torch": "t",
            "cuda_runtime": "c",
        },
    )
    monkeypatch.setattr(
        opd_probe,
        "_measure_probe",
        lambda *args, **kwargs: {
            "status": "measured",
            "identity": kwargs["identity"],
            "measurements": {"parameter_updates": 1},
            "recommendations": {},
            "failed_candidates": [],
        },
    )
    with pytest.raises(ConfigError, match="updated parameters"):
        opd_probe.run_hardware_probe(
            _native(), plan_digest="a" * 64, cache_dir=tmp_path, offline=True
        )
