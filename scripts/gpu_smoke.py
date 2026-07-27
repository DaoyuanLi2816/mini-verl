"""Minimal GPU smoke test: load the real pair, run one OPD cycle, measure.

Run this before committing to the full recipe budgets:

    python scripts/gpu_smoke.py --output runs/gpu-smoke

It downloads the pinned Qwen3 pair on first use (~5.6 GB), then reports peak
VRAM and throughput so the recipe can be sized from measurements instead of
guesses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miniverl.config import RunConfig
from miniverl.utils import gpu


def main() -> int:
    """Run a one-cycle OPD smoke test and print measurements as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="recipes/qwen_consumer_gpu_calc.yaml")
    parser.add_argument("--output", default="runs/gpu-smoke")
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--eval-tasks", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    if not gpu.cuda_available():
        print(json.dumps({"status": "not_run", "reason": "no CUDA device"}, indent=2))
        return 2

    payload = RunConfig.from_yaml(args.recipe).model_dump(mode="python")
    payload["run"]["name"] = "gpu-smoke"
    payload["train"].update(
        cycles=args.cycles,
        rollouts_per_cycle=args.rollouts,
        gradient_accumulation_steps=args.rollouts,
        sft_warmup_cycles=args.warmup,
        eval_every_cycles=0,
        save_every_cycles=0,
    )
    payload["rollout"]["max_new_tokens_per_turn"] = args.max_new_tokens
    payload["environment"].update(train_tasks=16, eval_tasks=args.eval_tasks, test_tasks=2)
    payload["eval"]["tasks"] = args.eval_tasks
    config = RunConfig.model_validate(payload)

    from miniverl.trainer import OPDTrainer

    started = time.perf_counter()
    trainer = OPDTrainer.from_config(config, output_dir=args.output, run_id="smoke")
    load_seconds = time.perf_counter() - started
    try:
        result = trainer.train()
    finally:
        trainer.close()

    snapshot = gpu.snapshot()
    report = {
        "status": "measured",
        "run_dir": str(trainer.paths.root),
        "load_seconds": round(load_seconds, 2),
        "train_seconds": round(result.duration_seconds, 2),
        "optimizer_steps": result.global_step,
        "memory_strategy": trainer.plan.strategy.value,
        "memory_reason": trainer.plan.reason,
        "projection_chunk_size": trainer.plan.chunk_size,
        "oom_retries_used": trainer.plan.oom_retries_used,
        "peak_allocated_gib": round(snapshot.peak_allocated_gib, 3),
        "peak_reserved_gib": round(snapshot.peak_reserved_gib, 3),
        "baseline_success_rate": (result.baseline_eval or {}).get("success_rate"),
        "final_success_rate": (result.eval or {}).get("success_rate"),
        "eval_rollout_tokens_per_second": (result.eval or {}).get("rollout_tokens_per_second"),
        "student_trainable_params": trainer.student.capabilities.num_trainable_parameters,
        "student_total_params": trainer.student.capabilities.num_parameters,
        "teacher_total_params": (
            trainer.teacher.capabilities.num_parameters if trainer.teacher else None
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
