"""The documented temperature sweep is executable and interpretation-safe."""

from __future__ import annotations

import math

from tests.conftest import requires_torch

pytestmark = [requires_torch]


def test_temperature_gradient_sweep_covers_objectives_and_regimes() -> None:
    from scripts.temperature_gradient_sweep import run_sweep

    rows = run_sweep(vocab_size=64)
    assert len(rows) == 2 * 3 * 4 * 2
    assert {row["scenario"] for row in rows} == {
        "near_uniform",
        "sharply_peaked_mismatched",
    }
    assert {row["divergence"] for row in rows} == {
        "forward_kl",
        "reverse_kl",
        "jsd",
    }
    assert all(
        math.isfinite(row["loss"])
        and math.isfinite(row["mean_abs_logit_gradient"])
        and row["mean_abs_logit_gradient"] > 0.0
        for row in rows
    )

    forward_high_t = [
        row["mean_abs_logit_gradient"]
        for row in rows
        if row["scenario"] == "near_uniform"
        and row["divergence"] == "forward_kl"
        and row["scale_by_temperature_squared"]
    ]
    assert max(forward_high_t) / min(forward_high_t) < 1.01
