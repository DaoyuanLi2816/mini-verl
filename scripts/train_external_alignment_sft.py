"""Train the bounded SFT run that produces the starting-checkpoint candidates.

Phase D of the v0.7.0 study needs a *non-saturated* starting policy. Rather than
searching for one, this trains a single bounded run and keeps checkpoints at a
committed set of update counts; the selection gate then takes the first
candidate in that order which clears it, using the eval split only.

One run, one trajectory through parameter space. Training separate models per
candidate would make the candidates incomparable, and picking the best-scoring
one would be selection on the outcome.

    python scripts/train_external_alignment_sft.py --out artifacts/v07-sft-candidates

Reports peak VRAM so the 14.5 GiB gate can be checked before anything else runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

CANDIDATE_UPDATES = (0, 4, 8, 16)
STUDENT_MODEL = "Qwen/Qwen3-0.6B"
STUDENT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

MAX_PROMPT_CHARACTERS = 1200
MAX_RESPONSE_CHARACTERS = 600
MAX_SEQUENCE_TOKENS = 768
EXAMPLES_PER_UPDATE = 8
SEED = 20260808


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-rows", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def _build_dataset(source_rows: int) -> list[Any]:
    from datasets import load_dataset

    from miniverl.alignment_external.training_data import (
        HH_RLHF_REVISION,
        build_examples,
    )

    raw = load_dataset("Anthropic/hh-rlhf", split="train", revision=HH_RLHF_REVISION)
    rows = [raw[index] for index in range(min(source_rows, raw.num_rows))]
    examples = list(
        build_examples(
            rows,
            max_prompt_characters=MAX_PROMPT_CHARACTERS,
            max_response_characters=MAX_RESPONSE_CHARACTERS,
        )
    )
    return [example for example in examples if example.split == "train"]


def main() -> int:
    args = _parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from miniverl.alignment_external.training_data import summarize
    from miniverl.utils.seeding import seed_everything

    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_examples = _build_dataset(args.source_rows)
    needed = max(CANDIDATE_UPDATES) * EXAMPLES_PER_UPDATE
    if len(train_examples) < needed:
        raise SystemExit(f"need {needed} training examples, have {len(train_examples)}")
    train_examples = train_examples[:needed]
    print(f"training examples: {len(train_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_MODEL, revision=STUDENT_REVISION, local_files_only=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        revision=STUDENT_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )

    def _encode(example: Any) -> tuple[list[int], list[int]]:
        """Prompt tokens are masked out; loss covers the response only."""
        prompt_text = f"{example.prompt}\n\nAssistant: "
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(example.chosen + tokenizer.eos_token, add_special_tokens=False)[
            "input_ids"
        ]
        ids = (prompt_ids + response_ids)[:MAX_SEQUENCE_TOKENS]
        labels = ([-100] * len(prompt_ids) + response_ids)[:MAX_SEQUENCE_TOKENS]
        return ids, labels

    def _save(update: int) -> dict[str, Any]:
        target = out / f"update-{update:03d}"
        model.save_pretrained(str(target))
        tokenizer.save_pretrained(str(target))
        digest = __import__("hashlib").sha256()
        for path in sorted(target.rglob("*")):
            if path.is_file():
                digest.update(path.name.encode())
                digest.update(path.read_bytes())
        record = {
            "update": update,
            "path": str(target),
            "adapter_digest": digest.hexdigest(),
        }
        print(f"saved candidate at update {update}: {record['adapter_digest'][:16]}")
        return record

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    candidates = [_save(0)]
    cursor = 0
    for update in range(1, max(CANDIDATE_UPDATES) + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = train_examples[cursor : cursor + EXAMPLES_PER_UPDATE]
        cursor += EXAMPLES_PER_UPDATE
        for example in batch:
            ids, labels = _encode(example)
            input_ids = torch.tensor([ids], device="cuda")
            label_ids = torch.tensor([labels], device="cuda")
            loss = model(input_ids=input_ids, labels=label_ids).loss / len(batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if update in CANDIDATE_UPDATES:
            candidates.append(_save(update))

    elapsed = time.time() - started
    manifest = {
        "schema_version": 1,
        "student_model": STUDENT_MODEL,
        "student_revision": STUDENT_REVISION,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "examples_per_update": EXAMPLES_PER_UPDATE,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "candidate_order": list(CANDIDATE_UPDATES),
        "candidates": candidates,
        "training_data": summarize(train_examples),
        "gpu_seconds": round(elapsed, 1),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }
    (out / "candidates.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"\ndone in {elapsed:.0f}s  peak reserved {manifest['peak_reserved_gib']} GiB (gate 14.5)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
