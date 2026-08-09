"""Pairwise preference scoring for the RewardBench endpoint.

A pairwise ranker is asked which of two responses is better. It is asked twice,
with the two responses in both orders, because these models have a documented
position bias and a single-order result silently bakes it in.

Three rules follow from that, and none of them is optional:

* both orders are evaluated wherever the ranker supports it;
* a pair the ranker orders inconsistently is a **tie**, not a coin flip. No
  winner is forced;
* the position-disagreement rate is published beside every win rate, so a
  reader can see how much of the signal is the ranker's own bias.

A PairRM preference is a model's preference. It is never described as human
preference, and the qualification numbers against the RewardBench labels are
published with the result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "PAIRWISE_OUTCOMES",
    "PairwiseJudge",
    "resolve_pair",
    "summarize_preference",
]

#: ``tie`` covers both a genuine tie and an order-inconsistent pair. They are
#: counted separately in the summary so the second is visible.
PAIRWISE_OUTCOMES = ("a", "b", "tie")


def resolve_pair(forward: str, reversed_: str) -> tuple[str, bool]:
    """Combine the two orderings into one outcome.

    ``forward`` is the winner with (A, B) presented in that order; ``reversed_``
    is the winner with (B, A). Both are expressed in terms of the original A/B
    labels. Returns ``(outcome, disagreed)``.
    """
    if forward not in PAIRWISE_OUTCOMES or reversed_ not in PAIRWISE_OUTCOMES:
        raise ValueError(f"unknown pairwise outcome {forward!r}/{reversed_!r}")
    if forward == reversed_:
        return forward, False
    # The ranker changed its mind when the responses swapped places. That is
    # position bias, and the honest outcome is no preference.
    return "tie", True


class PairwiseJudge:
    """Pinned local pairwise ranker. Never contacts the network."""

    def __init__(self, model_id: str, revision: str, *, device: str = "cuda") -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self._ranker: Any = None

    def load(self) -> PairwiseJudge:
        """Load the pinned revision from local files only."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._ranker = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._ranker.eval()
        self._ranker.to(self.device)
        return self

    @property
    def loaded(self) -> bool:
        return self._ranker is not None

    def identity(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "network_access": "never; local_files_only=True and trust_remote_code=False",
            "both_orders": True,
        }

    def _score(self, prompt: str, first: str, second: str) -> float:
        """Higher means the first response is preferred."""
        import torch

        encoded = self._tokenizer(
            f"{prompt}\n\n[RESPONSE A]\n{first}\n\n[RESPONSE B]\n{second}",
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)
        with torch.no_grad():
            logits = self._ranker(**encoded).logits[0]
        return float(logits[0]) if logits.numel() == 1 else float(logits[0] - logits[-1])

    def compare(self, prompt: str, response_a: str, response_b: str) -> dict[str, Any]:
        """Compare a pair in both orders and resolve to one outcome."""
        if not self.loaded:
            raise RuntimeError("PairwiseJudge.load() must be called before compare()")
        forward_margin = self._score(prompt, response_a, response_b)
        # Swapped: a positive margin now favours B.
        reverse_margin = self._score(prompt, response_b, response_a)

        forward = "a" if forward_margin > 0 else "b" if forward_margin < 0 else "tie"
        reversed_ = "b" if reverse_margin > 0 else "a" if reverse_margin < 0 else "tie"
        outcome, disagreed = resolve_pair(forward, reversed_)
        return {
            "outcome": outcome,
            "order_disagreement": disagreed,
            "forward_winner": forward,
            "reversed_winner": reversed_,
            "forward_margin": forward_margin,
            "reverse_margin": reverse_margin,
        }


def summarize_preference(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate pairwise outcomes into a win rate plus the bias it rests on."""
    wins = losses = ties = disagreements = 0
    by_subset: dict[str, dict[str, int]] = {}

    for record in records:
        outcome = str(record["outcome"])
        if outcome not in PAIRWISE_OUTCOMES:
            raise ValueError(f"unknown pairwise outcome {outcome!r}")
        disagreed = bool(record.get("order_disagreement", False))
        subset = str(record.get("subset", "unspecified"))
        bucket = by_subset.setdefault(
            subset, {"a": 0, "b": 0, "tie": 0, "order_disagreement": 0, "total": 0}
        )
        bucket[outcome] += 1
        bucket["total"] += 1
        bucket["order_disagreement"] += int(disagreed)

        disagreements += int(disagreed)
        if outcome == "a":
            wins += 1
        elif outcome == "b":
            losses += 1
        else:
            ties += 1

    total = wins + losses + ties
    decided = wins + losses
    return {
        "judge": "pairrm",
        "judge_kind": "pairwise_model",
        "pairs_total": total,
        "pairs_decided": decided,
        "ties": ties,
        # Ties stay out of the denominator; forcing them into a win rate would
        # convert the ranker's uncertainty into a preference it never expressed.
        "win_rate": (wins / decided) if decided else None,
        "wins": wins,
        "losses": losses,
        "order_disagreements": disagreements,
        "order_disagreement_rate": (disagreements / total) if total else None,
        "by_subset": {name: dict(counts) for name, counts in sorted(by_subset.items())},
        "scope": (
            "a pinned 0.4B pairwise ranker's preference over a fixed RewardBench "
            "subset, evaluated in both orders. This is a model preference, not a "
            "human preference."
        ),
    }


def qualify_judge(
    labelled: Sequence[Mapping[str, Any]],
    *,
    minimum_agreement: float,
    maximum_order_disagreement: float,
) -> dict[str, Any]:
    """Score the ranker against RewardBench's own chosen/rejected labels.

    Two floors, both from the preregistration: how often the ranker agrees with
    the benchmark label, and how often it contradicts itself under order
    reversal. Failing either disqualifies the endpoint.
    """
    total = len(labelled)
    disagreements = sum(1 for row in labelled if row.get("order_disagreement"))
    decided = [row for row in labelled if str(row["outcome"]) != "tie"]
    correct = sum(1 for row in decided if str(row["outcome"]) == str(row["expected"]))

    agreement = (correct / len(decided)) if decided else None
    order_rate = (disagreements / total) if total else None
    qualified = bool(
        agreement is not None
        and order_rate is not None
        and agreement >= minimum_agreement
        and order_rate <= maximum_order_disagreement
    )
    return {
        "calibration_pairs": total,
        "decided": len(decided),
        "agreement": agreement,
        "minimum_agreement": minimum_agreement,
        "order_disagreement_rate": order_rate,
        "maximum_order_disagreement": maximum_order_disagreement,
        "qualified": qualified,
        "note": (
            "a ranker below the agreement floor, or above the order-disagreement "
            "ceiling, disqualifies the endpoint, which then reports "
            "not_applicable instead of a win rate"
        ),
    }
