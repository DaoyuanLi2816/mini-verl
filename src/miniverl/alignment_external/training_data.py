"""Continuation training data, parsed from a licensed preference corpus.

All five continuation arms need the same prompts, or the comparison is between
datasets rather than between methods:

* continued alignment SFT trains on the chosen response as a hard target;
* DPO trains on the (chosen, rejected) pair;
* offline KD and both OPD variants use the same prompts and let the teacher
  supply the targets.

The source is ``Anthropic/hh-rlhf`` (MIT). It was chosen over the alternatives
on licence and disjointness: ``LLM-LAT/harmful-dataset`` declares no licence at
all, ``PKU-Alignment/PKU-SafeRLHF`` is CC-BY-NC and would attach a
non-commercial restriction to the whole study, and neither RewardBench nor any
other endpoint draws from hh-rlhf.

Splitting is deterministic and derived from the row digest, not from row order,
so the train/eval boundary cannot move when upstream reorders rows.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "HH_RLHF_DATASET",
    "HH_RLHF_REVISION",
    "PreferenceExample",
    "assign_split",
    "parse_transcript",
    "build_examples",
]

HH_RLHF_DATASET = "Anthropic/hh-rlhf"
HH_RLHF_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"

#: hh-rlhf transcripts are one string with these turn markers.
_TURN = re.compile(r"\n\n(Human|Assistant):\s?")


@dataclass(frozen=True, slots=True)
class PreferenceExample:
    """One prompt with its preferred and dispreferred final response."""

    example_id: str
    prompt: str
    chosen: str
    rejected: str
    turns: int
    split: str

    def digest(self) -> str:
        payload = f"{self.prompt}\x00{self.chosen}\x00{self.rejected}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_transcript(transcript: str) -> tuple[list[tuple[str, str]], str] | None:
    """Split an hh-rlhf transcript into turns plus the final assistant reply.

    Returns ``None`` when the transcript does not end with an assistant turn or
    has no exchange at all -- a malformed row is dropped rather than repaired,
    because guessing what a truncated transcript meant would silently change
    the training distribution.
    """
    parts = _TURN.split(transcript)
    if len(parts) < 3:
        return None
    # `parts[0]` is whatever preceded the first marker; hh-rlhf leaves it empty.
    turns: list[tuple[str, str]] = []
    for role, text in zip(parts[1::2], parts[2::2], strict=True):
        turns.append((role, text.strip()))
    if not turns or turns[-1][0] != "Assistant":
        return None
    final = turns[-1][1]
    if not final:
        return None
    return turns[:-1], final


def assign_split(example_id: str, *, eval_fraction: float = 0.2, seed: int = 20260808) -> str:
    """Deterministic train/eval assignment from the example's own identity.

    Hashing the id rather than slicing by position means the boundary survives
    upstream reordering, and re-running the preparation puts every example back
    in the split it was in before.
    """
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be strictly between 0 and 1")
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return "eval" if bucket < eval_fraction else "train"


def build_examples(
    rows: Sequence[dict[str, Any]],
    *,
    max_prompt_characters: int,
    max_response_characters: int,
    eval_fraction: float = 0.2,
    seed: int = 20260808,
) -> Iterator[PreferenceExample]:
    """Turn raw hh-rlhf rows into bounded examples, dropping what does not fit.

    Length bounds are applied by *rejecting* an example, never by truncating it.
    A truncated preference pair is a different pair, and a truncated prompt can
    remove the very context that made the preferred answer preferred.
    """
    for index, row in enumerate(rows):
        chosen_parsed = parse_transcript(str(row.get("chosen", "")))
        rejected_parsed = parse_transcript(str(row.get("rejected", "")))
        if chosen_parsed is None or rejected_parsed is None:
            continue
        chosen_turns, chosen_final = chosen_parsed
        rejected_turns, rejected_final = rejected_parsed
        # The pair only means something if both branches share the same prompt.
        if chosen_turns != rejected_turns:
            continue
        if chosen_final == rejected_final:
            continue

        prompt = "\n\n".join(f"{role}: {text}" for role, text in chosen_turns)
        if not prompt:
            continue
        if len(prompt) > max_prompt_characters:
            continue
        if max(len(chosen_final), len(rejected_final)) > max_response_characters:
            continue

        example_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        yield PreferenceExample(
            example_id=example_id,
            prompt=prompt,
            chosen=chosen_final,
            rejected=rejected_final,
            turns=len(chosen_turns),
            split=assign_split(example_id, eval_fraction=eval_fraction, seed=seed),
        )


def summarize(examples: Sequence[PreferenceExample]) -> dict[str, Any]:
    """Counts and digests for the preregistration record."""
    by_split: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        by_split.setdefault(example.split, []).append(example)

    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda item: item.example_id):
        digest.update(example.digest().encode("ascii"))

    return {
        "dataset": HH_RLHF_DATASET,
        "revision": HH_RLHF_REVISION,
        "license": "mit",
        "total": len(examples),
        "by_split": {name: len(items) for name, items in sorted(by_split.items())},
        "split_rule": "sha256(seed:example_id) bucketed; independent of row order",
        "content_digest": digest.hexdigest(),
        "length_policy": "over-long examples are dropped, never truncated",
    }
