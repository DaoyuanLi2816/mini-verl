"""Preregistration amendment 2: the fallback starting-checkpoint lineage.

The primary lineage continues Qwen3-0.6B on HH-RLHF and scored 0/64 on JSONNav
retained tool utility at every candidate. HH-RLHF is a conversational
preference corpus, so the policy never learned the tool protocol; the failure is
about missing tool competence, not about saturation, which is exactly the case
amendment 2 was written for before any of those numbers existed.

This runs the identical continuation procedure from a different anchor: the
public pre-v0.7 tool-policy SFT adapter, which already has tool-protocol
competence to retain. Same HH-RLHF data, same candidate order 0/4/8/16, same
gate. Only the starting point differs.

The anchor was published before v0.7 and was not chosen from any external eval
outcome, which is what keeps this a declared contingency rather than a search
for a lineage that passes.

    python scripts/train_external_alignment_fallback_sft.py \
        --out artifacts/v07-sft-candidates-fallback
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

CANDIDATE_UPDATES = (0, 4, 8, 16)
STUDENT_MODEL = "Qwen/Qwen3-0.6B"
STUDENT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

#: The fallback anchor, fixed by amendment 2.
ANCHOR_REPOSITORY = "DaoyuanLi/mini-verl-qwen3-0.6b-tool-policy-sft"
ANCHOR_REVISION = "7b98164f73e493c51f2ed3fca3169fea078f47f0"

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


def _train_examples(source_rows: int) -> list[Any]:
    """The same HH-RLHF train split the primary lineage used."""
    from datasets import load_dataset

    from miniverl.alignment_external.training_data import HH_RLHF_REVISION, build_examples

    raw = load_dataset("Anthropic/hh-rlhf", split="train", revision=HH_RLHF_REVISION)
    rows = [raw[index] for index in range(min(source_rows, raw.num_rows))]
    examples = build_examples(
        rows,
        max_prompt_characters=MAX_PROMPT_CHARACTERS,
        max_response_characters=MAX_RESPONSE_CHARACTERS,
    )
    return [example for example in examples if example.split == "train"]


def main() -> int:
    args = _parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from miniverl.alignment_external.training_data import summarize
    from miniverl.utils.seeding import seed_everything

    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    examples = _train_examples(args.source_rows)
    needed = max(CANDIDATE_UPDATES) * EXAMPLES_PER_UPDATE
    if len(examples) < needed:
        raise SystemExit(f"need {needed} training examples, have {len(examples)}")
    examples = examples[:needed]

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, revision=STUDENT_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        revision=STUDENT_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")  # type: ignore[arg-type]
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    # The one difference from the primary lineage: continue the published
    # tool-policy adapter instead of starting a fresh one. `is_trainable` keeps
    # the existing LoRA weights updatable rather than merging them frozen.
    model = PeftModel.from_pretrained(
        base, ANCHOR_REPOSITORY, revision=ANCHOR_REVISION, is_trainable=True
    )
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"anchor loaded: {ANCHOR_REPOSITORY}@{ANCHOR_REVISION[:8]}")
    print(f"trainable parameters: {trainable:,}")
    if trainable == 0:
        raise SystemExit("the anchor adapter loaded frozen; continuation would be a no-op")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )

    def _encode(example: Any) -> tuple[list[int], list[int]]:
        prompt_ids = tokenizer(f"{example.prompt}\n\nAssistant: ", add_special_tokens=False)[
            "input_ids"
        ]
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
        digest = hashlib.sha256()
        for path in sorted(target.rglob("*")):
            if path.is_file():
                digest.update(path.name.encode())
                digest.update(path.read_bytes())
        record: dict[str, Any] = {
            "update": update,
            "path": str(target),
            "adapter_digest": digest.hexdigest(),
        }
        print(f"saved fallback candidate at update {update}: {record['adapter_digest'][:16]}")
        return record

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    candidates = [_save(0)]
    cursor = 0
    for update in range(1, max(CANDIDATE_UPDATES) + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = examples[cursor : cursor + EXAMPLES_PER_UPDATE]
        cursor += EXAMPLES_PER_UPDATE
        for example in batch:
            ids, labels = _encode(example)
            loss = model(
                input_ids=torch.tensor([ids], device="cuda"),
                labels=torch.tensor([labels], device="cuda"),
            ).loss / len(batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if update in CANDIDATE_UPDATES:
            candidates.append(_save(update))

    elapsed = time.time() - started
    manifest = {
        "schema_version": 1,
        "lineage": "fallback",
        "amendment": 2,
        "anchor_repository": ANCHOR_REPOSITORY,
        "anchor_revision": ANCHOR_REVISION,
        "student_model": STUDENT_MODEL,
        "student_revision": STUDENT_REVISION,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "examples_per_update": EXAMPLES_PER_UPDATE,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "candidate_order": list(CANDIDATE_UPDATES),
        "candidates": candidates,
        "training_data": summarize(examples),
        "gpu_seconds": round(elapsed, 1),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "procedure_identical_to_primary": True,
        "only_difference": "the starting adapter",
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
