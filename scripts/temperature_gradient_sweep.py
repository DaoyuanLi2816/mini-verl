"""Measure logit-gradient norms across divergences, temperatures and regimes.

The ``T**2`` derivation is asymptotic and specific to forward KL near the
uniform high-temperature regime.  This experiment deliberately also reports
reverse KL, JSD and sharply peaked logits, where the factor is only a heuristic.

    python scripts/temperature_gradient_sweep.py
    python scripts/temperature_gradient_sweep.py --json
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def run_sweep(*, vocab_size: int = 512) -> list[dict[str, Any]]:
    """Return a deterministic grid of mean absolute student-logit gradients."""
    import torch

    from miniverl.losses.exact import exact_divergence

    generator = torch.Generator().manual_seed(20260727)
    near_teacher = torch.randn(1, vocab_size, generator=generator) * 0.05
    near_student = torch.randn(1, vocab_size, generator=generator) * 0.05
    sharp_teacher = torch.full((1, vocab_size), -8.0)
    sharp_student = torch.full((1, vocab_size), -8.0)
    sharp_teacher[0, 3] = 9.0
    sharp_student[0, 11] = 8.0
    scenarios = {
        "near_uniform": (near_teacher, near_student),
        "sharply_peaked_mismatched": (sharp_teacher, sharp_student),
    }

    rows: list[dict[str, Any]] = []
    for scenario, (teacher, initial_student) in scenarios.items():
        for divergence in ("forward_kl", "reverse_kl", "jsd"):
            for temperature in (1.0, 2.0, 4.0, 8.0):
                for scaled in (False, True):
                    student = initial_student.clone().requires_grad_(True)
                    loss = exact_divergence(
                        teacher,
                        student,
                        divergence=divergence,
                        temperature=temperature,
                        scale_by_temperature_squared=scaled,
                    ).sum()
                    loss.backward()
                    assert student.grad is not None
                    rows.append(
                        {
                            "scenario": scenario,
                            "divergence": divergence,
                            "temperature": temperature,
                            "scale_by_temperature_squared": scaled,
                            "loss": float(loss.detach()),
                            "mean_abs_logit_gradient": float(student.grad.abs().mean()),
                            "l2_logit_gradient": float(student.grad.norm()),
                        }
                    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the full JSON grid")
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args(argv)
    rows = run_sweep(vocab_size=args.vocab_size)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print("scenario divergence T T2 mean_abs_grad l2_grad")
    for row in rows:
        print(
            f"{row['scenario']} {row['divergence']} {row['temperature']:.0f} "
            f"{int(row['scale_by_temperature_squared'])} "
            f"{row['mean_abs_logit_gradient']:.6e} "
            f"{row['l2_logit_gradient']:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
