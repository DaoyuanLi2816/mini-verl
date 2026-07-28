"""CUDA regression tests for destructive trainer teardown."""

from __future__ import annotations

import gc
import json

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.torch]

_PROBE_ELEMENTS = 8_000_000  # 32 MB at float32: well above the cleanup tolerance.
_ABSOLUTE_TOLERANCE = 2 * 1024**2


def _config(tmp_path):  # type: ignore[no-untyped-def]
    from tests.integration.test_toy_pipeline import _config as toy_config

    return toy_config(
        tmp_path,
        models={"device": "cuda"},
        train={"cycles": 1, "rollouts_per_cycle": 1, "gradient_accumulation_steps": 1},
        eval={"enabled": False},
        report={"enabled": False},
    )


def _clean_cuda(torch):  # type: ignore[no-untyped-def]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())


def _tolerance(baseline: int) -> int:
    # CUDA libraries may retain small process-global workspaces; 2 MiB plus 2%
    # of the starting footprint is tight enough to catch the 32 MB live probe.
    return max(_ABSOLUTE_TOLERANCE, int(baseline * 0.02))


def test_sequential_trainers_return_allocated_memory_to_baseline(tmp_path) -> None:
    import torch

    from miniverl.trainer import OPDTrainer

    baseline_allocated, baseline_reserved = _clean_cuda(torch)
    tolerance = _tolerance(baseline_allocated)

    first = OPDTrainer.from_config(_config(tmp_path / "first"), run_id="first")
    first.student._lifecycle_probe = torch.empty(
        _PROBE_ELEMENTS, dtype=torch.float32, device="cuda"
    )
    torch.cuda.synchronize()
    first_live = int(torch.cuda.memory_allocated())
    assert first_live >= baseline_allocated + _PROBE_ELEMENTS * 4

    first.close()
    torch.cuda.synchronize()
    after_first_allocated = int(torch.cuda.memory_allocated())
    after_first_reserved = int(torch.cuda.memory_reserved())
    assert after_first_allocated <= baseline_allocated + tolerance
    assert after_first_reserved <= baseline_reserved + tolerance
    del first

    before_second_allocated, _ = _clean_cuda(torch)
    assert before_second_allocated <= baseline_allocated + tolerance
    second = OPDTrainer.from_config(_config(tmp_path / "second"), run_id="second")
    second.student._lifecycle_probe = torch.empty(
        _PROBE_ELEMENTS, dtype=torch.float32, device="cuda"
    )
    torch.cuda.synchronize()
    second_live = int(torch.cuda.memory_allocated())
    assert second_live <= first_live + tolerance

    second.close()
    torch.cuda.synchronize()
    after_second_allocated = int(torch.cuda.memory_allocated())
    after_second_reserved = int(torch.cuda.memory_reserved())
    assert after_second_allocated <= baseline_allocated + tolerance
    assert after_second_reserved <= baseline_reserved + tolerance
    print(
        json.dumps(
            {
                "baseline_allocated": baseline_allocated,
                "baseline_reserved": baseline_reserved,
                "first_live_allocated": first_live,
                "after_first_allocated": after_first_allocated,
                "after_first_reserved": after_first_reserved,
                "before_second_allocated": before_second_allocated,
                "second_live_allocated": second_live,
                "after_second_allocated": after_second_allocated,
                "after_second_reserved": after_second_reserved,
                "tolerance": tolerance,
            },
            sort_keys=True,
        )
    )
