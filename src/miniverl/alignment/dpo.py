"""Pinned, path-independent TRL DPO baseline configuration."""

from __future__ import annotations

import hashlib
from typing import Any

from miniverl.utils.runs import canonical_json

__all__ = ["TRL_VERSION", "build_dpo_training_config", "dpo_config_digest"]

TRL_VERSION = "1.8.0"


def build_dpo_training_config(
    *,
    seed: int,
    max_steps: int,
    learning_rate: float,
    beta: float,
) -> dict[str, Any]:
    """Return the exact public config with a portable output placeholder."""
    return {
        "output_dir": "<OUTPUT>",
        "max_steps": max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": learning_rate,
        "beta": beta,
        "loss_type": "sigmoid",
        "max_length": 1024,
        "gradient_checkpointing": True,
        "bf16": True,
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "warmup_steps": 1,
        "max_grad_norm": 1.0,
        "save_strategy": "no",
        "logging_steps": 1,
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "full_determinism": True,
        "dataset_num_proc": 1,
        "remove_unused_columns": True,
    }


def dpo_config_digest(config: dict[str, Any]) -> str:
    """Hash the portable exact config written to the DPO manifest."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
