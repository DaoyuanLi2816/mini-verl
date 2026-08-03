#!/usr/bin/env python3
"""Train the pinned TRL DPO baseline on deterministic sandbox preferences."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any

from miniverl.alignment import build_tool_policy_preferences, preference_dataset_digest
from miniverl.utils.runs import canonical_json, utc_now, write_json_atomic

TRL_VERSION = "1.8.0"
BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starting-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-tasks", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    installed = importlib.metadata.version("trl")
    if installed != TRL_VERSION:
        raise RuntimeError(f"this baseline requires trl=={TRL_VERSION}, found {installed}")
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (args.starting_adapter / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"starting adapter is incomplete ({', '.join(missing)}): {args.starting_adapter}"
        )
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {args.output}")
        shutil.rmtree(args.output)

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("the formal DPO baseline requires one CUDA GPU")
    rows = build_tool_policy_preferences(
        count=args.train_tasks,
        seed=20260802,
        split="train",
    )
    dataset_digest = preference_dataset_digest(rows)
    dataset = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(
        base,
        str(args.starting_adapter),
        is_trainable=True,
        local_files_only=True,
    )
    exact_config: dict[str, Any] = {
        "output_dir": str(args.output),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
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
        "seed": args.seed,
        "data_seed": args.seed,
        "full_determinism": True,
        "dataset_num_proc": 1,
        "remove_unused_columns": True,
    }
    config_digest = hashlib.sha256(canonical_json(exact_config).encode("utf-8")).hexdigest()
    training_args = DPOConfig(**exact_config)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    output = trainer.train()
    trainer.save_model(str(args.output))
    weights = args.output / "adapter_model.safetensors"
    adapter_config = args.output / "adapter_config.json"
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "method": "dpo",
        "trl_version": installed,
        "base_model": {"id": BASE_MODEL, "revision": BASE_REVISION},
        "reference": {
            "kind": "implicit_initial_policy",
            "starting_adapter": args.starting_adapter.name,
            "adapter_config_sha256": _sha256(args.starting_adapter / "adapter_config.json"),
            "adapter_weights_sha256": _sha256(args.starting_adapter / "adapter_model.safetensors"),
        },
        "dataset": {
            "id": "miniverl-tool-policy-preferences",
            "revision": "v1",
            "rows": len(rows),
            "sha256": dataset_digest,
            "contains_real_actions": False,
        },
        "config": exact_config,
        "exact_config_sha256": config_digest,
        "seed": args.seed,
        "train_metrics": dict(output.metrics),
        "adapter": {
            "config_sha256": _sha256(adapter_config),
            "weights_sha256": _sha256(weights),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(0),
        },
    }
    write_json_atomic(args.output / "dpo_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
