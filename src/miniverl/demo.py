"""The no-network toy demo.

``miniverl demo`` runs the *real* pipeline -- student rollouts with tool
execution, teacher scoring of exactly those states, a compressed top-k cache
with provenance checks, and a masked reverse-KL update on assistant tokens only
-- at a size that finishes on a laptop CPU in well under a minute.

The configuration is embedded rather than read from disk so the demo works from
a wheel with no repository checked out.  It is intentionally smaller than
``recipes/toy_cpu.yaml``: the demo proves the *machinery* in well under a
minute, the recipe spends a few minutes and produces a learning curve.

What the demo does and does not show
------------------------------------
It shows that the pipeline runs end to end and writes every artifact, and that
the provenance and cache guarantees hold on real data.  It does **not** show
that the student learns the task: at this budget the toy student learns the
tool-call *format* and not the arithmetic copying, so the success rate normally
stays at zero.

That is measured, not assumed.  On the calculator ``easy`` split with 256
training tasks and 600 supervised steps the same toy student reaches **81.2%**
with ``run.seed: 1234`` and **0.0%** with ``run.seed: 20260727`` -- a ~200k
parameter model at this scale either acquires the copy behaviour or it does
not, depending on initialization.  Capability numbers therefore come from the
GPU recipe, never from here.  See ``docs/limitations.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniverl.config.models import RunConfig

__all__ = ["demo_config", "DEMO_CONFIG"]

DEMO_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "run": {
        "name": "miniverl-demo",
        "mode": "opd",
        "seed": 1234,
        "deterministic": True,
        "notes": "Embedded no-network toy demo. Real pipeline, tiny models.",
    },
    "models": {
        "backend": "toy",
        "device": "cpu",
        "student": {
            "model_id": "toy-student",
            "dtype": "float32",
            "lora": {"enabled": False},
            "toy": {
                "hidden_size": 96,
                "num_layers": 3,
                "num_heads": 4,
                "intermediate_size": 192,
                "max_position_embeddings": 768,
            },
        },
        "teacher": {
            "model_id": "toy-teacher",
            "dtype": "float32",
            "mode": "standard",
            "toy_pretrain_steps": 260,
            "toy_pretrain_lr": 0.003,
            "toy": {
                "hidden_size": 160,
                "num_layers": 4,
                "num_heads": 4,
                "intermediate_size": 320,
                "max_position_embeddings": 768,
            },
        },
    },
    "environment": {
        "name": "calculator",
        "difficulty": "easy",
        "params": {"prompt_style": "compact"},
        "train_tasks": 128,
        "eval_tasks": 8,
        "test_tasks": 8,
        "split_seed": 7,
    },
    "rollout": {
        "max_turns": 3,
        "max_new_tokens_per_turn": 40,
        "max_total_tokens": 512,
        "temperature": 1.0,
        "max_parse_errors": 2,
        "max_repeated_calls": 2,
    },
    "selection": {"selector": "all_model_tokens"},
    "loss": {
        "mode": "bucketed_topk_tail",
        "divergence": "reverse_kl",
        "temperature": 1.0,
        "top_k": 16,
        "chunk_size": 128,
    },
    "train": {
        "cycles": 12,
        "rollouts_per_cycle": 8,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.003,
        "lr_schedule": "cosine",
        "sft_warmup_cycles": 120,
        "eval_every_cycles": 0,
    },
    "memory": {"strategy": "resident"},
    "cache": {"entries_per_shard": 8, "dtype": "float32", "keep_cycles": 2},
    "eval": {"enabled": True, "split": "eval", "temperature": 0.0, "seed": 0},
    "report": {"enabled": True, "max_trajectories": 3, "max_tokens_per_trajectory": 200},
}


def demo_config(*, fast: bool = False, output_dir: str | Path = "runs") -> RunConfig:
    """Build the embedded demo configuration.

    ``fast`` shrinks it further for CI smoke tests: the pipeline is identical,
    only the step counts change.
    """
    import copy

    payload = copy.deepcopy(DEMO_CONFIG)
    payload["run"]["output_dir"] = str(output_dir)
    if fast:
        payload["models"]["student"]["toy"].update(
            hidden_size=48, num_layers=2, intermediate_size=96
        )
        payload["models"]["teacher"]["toy"].update(
            hidden_size=64, num_layers=2, intermediate_size=128
        )
        payload["models"]["teacher"]["toy_pretrain_steps"] = 4
        payload["environment"].update(train_tasks=8, eval_tasks=2, test_tasks=2)
        payload["rollout"].update(max_new_tokens_per_turn=12, max_turns=2, max_total_tokens=400)
        payload["train"].update(
            cycles=2, rollouts_per_cycle=2, gradient_accumulation_steps=2, sft_warmup_cycles=2
        )
        payload["cache"]["entries_per_shard"] = 2
        payload["report"]["max_trajectories"] = 1
    return RunConfig.model_validate(payload)
