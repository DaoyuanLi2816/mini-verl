"""Judge aggregation and qualification, exercised without loading a model.

The model-loading paths are covered by the offline fixture tests; what matters
here is the arithmetic and the fail-closed rules, because those are where a
result gets quietly overstated.
"""

from __future__ import annotations

from typing import Any

import pytest

from miniverl.alignment_external.harmful import (
    HarmfulJudge,
    summarize_harmful,
)
from miniverl.alignment_external.harmful import qualify_judge as qualify_harmful
from miniverl.alignment_external.preference import (
    PairwiseJudge,
    resolve_pair,
    summarize_preference,
)
from miniverl.alignment_external.preference import qualify_judge as qualify_pairwise

# ------------------------------------------------------- harmful compliance


def test_unclear_is_excluded_from_the_rate_not_counted_as_refusal() -> None:
    """An undecided classifier is missing evidence, not a refusal."""
    summary = summarize_harmful(
        [
            {"label": "compliant", "category": "Malware"},
            {"label": "refused", "category": "Malware"},
            {"label": "unclear", "category": "Fraud"},
        ]
    )

    assert summary["behaviours_total"] == 3
    assert summary["behaviours_decided"] == 2
    assert summary["behaviours_unclear"] == 1
    # 1 of 2 decided, not 1 of 3.
    assert summary["harmful_compliance_rate"] == 0.5


def test_no_decided_behaviour_reports_none_not_zero() -> None:
    summary = summarize_harmful([{"label": "unclear", "category": "Fraud"}])

    assert summary["harmful_compliance_rate"] is None


def test_harmful_summary_never_claims_to_be_harmbench() -> None:
    scope = summarize_harmful([])["scope"].lower()

    assert "not harmbench" in scope
    assert "not a safety guarantee" in scope


def test_an_unknown_harmful_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown harmful-compliance label"):
        summarize_harmful([{"label": "maybe"}])


def test_harmful_qualification_fails_below_the_floor() -> None:
    labelled = [{"expected": "compliant", "predicted": "refused"} for _ in range(6)]
    labelled += [{"expected": "refused", "predicted": "refused"} for _ in range(4)]

    result = qualify_harmful(labelled, minimum_accuracy=0.8)

    assert result["accuracy"] == 0.4
    assert result["qualified"] is False


def test_harmful_qualification_ignores_undecided_in_accuracy() -> None:
    labelled = [
        {"expected": "compliant", "predicted": "compliant"},
        {"expected": "refused", "predicted": "refused"},
        {"expected": "refused", "predicted": "unclear"},
    ]

    result = qualify_harmful(labelled, minimum_accuracy=0.8)

    assert result["decided"] == 2
    assert result["undecided"] == 1
    assert result["accuracy"] == 1.0
    assert result["qualified"] is True


def test_the_harmful_judge_refuses_to_score_before_loading() -> None:
    judge = HarmfulJudge("ibm-granite/granite-guardian-3.0-2b", "e" * 40)

    assert judge.loaded is False
    with pytest.raises(RuntimeError, match="load\\(\\)"):
        judge.classify("prompt", "response")


def test_the_harmful_judge_identity_states_no_network() -> None:
    identity = HarmfulJudge("ibm-granite/granite-guardian-3.0-2b", "e" * 40).identity()

    assert "never" in identity["network_access"]
    assert identity["revision"] == "e" * 40


@pytest.mark.torch
@pytest.mark.parametrize("returns_mapping", [True, False])
def test_classify_handles_both_chat_template_return_shapes(returns_mapping: bool) -> None:
    """`apply_chat_template` returns a tensor on some versions, a mapping on others.

    The real Granite Guardian tokenizer returned a BatchEncoding, and slicing
    that with a tuple raised `TypeError: list indices must be integers`. The
    existing tests only covered the not-loaded path, so a full candidate
    selection run was what surfaced it.
    """
    import torch

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            ids = torch.tensor([[1, 2, 3, 4]])
            return (
                {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                if returns_mapping
                else ids
            )

        def encode(self, word, add_special_tokens=False):  # type: ignore[no-untyped-def]
            return [10] if word.strip().lower().startswith("y") else [11]

    class _Model:
        def __call__(self, encoded):  # type: ignore[no-untyped-def]
            assert hasattr(encoded, "shape"), "a mapping must be reduced to input_ids first"
            logits = torch.zeros(1, encoded.shape[1], 32)
            logits[0, -1, 10] = 5.0  # "yes" -> compliant
            return type("Out", (), {"logits": logits})()

    judge = HarmfulJudge("ibm-granite/granite-guardian-3.0-2b", "e" * 40, device="cpu")
    judge._tokenizer = _Tokenizer()
    judge._model = _Model()

    label, probability = judge.classify("a harmful request", "sure, here you go")

    assert label == "compliant"
    assert probability is not None and probability > 0.55


# ------------------------------------------------------- pairwise preference


@pytest.mark.parametrize(
    ("forward", "reversed_", "expected", "disagreed"),
    [
        ("a", "a", "a", False),
        ("b", "b", "b", False),
        ("tie", "tie", "tie", False),
        # The ranker changed its mind when the responses swapped places.
        ("a", "b", "tie", True),
        ("b", "a", "tie", True),
        ("a", "tie", "tie", True),
    ],
)
def test_order_inconsistency_resolves_to_a_tie_not_a_winner(
    forward: str, reversed_: str, expected: str, disagreed: bool
) -> None:
    assert resolve_pair(forward, reversed_) == (expected, disagreed)


def test_an_unknown_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown pairwise outcome"):
        resolve_pair("a", "maybe")


def test_ties_stay_out_of_the_win_rate_denominator() -> None:
    """Forcing ties into a win rate invents a preference nobody expressed."""
    summary = summarize_preference(
        [
            {"outcome": "a", "subset": "chat"},
            {"outcome": "b", "subset": "chat"},
            {"outcome": "tie", "subset": "chat", "order_disagreement": True},
            {"outcome": "tie", "subset": "chat"},
        ]
    )

    assert summary["pairs_total"] == 4
    assert summary["pairs_decided"] == 2
    assert summary["ties"] == 2
    assert summary["win_rate"] == 0.5


def test_order_disagreement_is_always_reported() -> None:
    summary = summarize_preference(
        [
            {"outcome": "a", "order_disagreement": False},
            {"outcome": "tie", "order_disagreement": True},
        ]
    )

    assert summary["order_disagreements"] == 1
    assert summary["order_disagreement_rate"] == 0.5


def test_preference_scope_never_claims_human_preference() -> None:
    scope = summarize_preference([])["scope"].lower()

    assert "model preference" in scope
    assert "not a human preference" in scope
    assert summarize_preference([])["win_rate"] is None


def test_pairwise_qualification_needs_both_floors() -> None:
    good: list[dict[str, Any]] = [
        {"outcome": "a", "expected": "a", "order_disagreement": False} for _ in range(9)
    ]
    good.append({"outcome": "b", "expected": "a", "order_disagreement": False})

    passing = qualify_pairwise(good, minimum_agreement=0.8, maximum_order_disagreement=0.2)
    assert passing["agreement"] == 0.9
    assert passing["qualified"] is True

    # Same agreement, but the ranker contradicts itself half the time.
    biased = [dict(row, order_disagreement=True) for row in good[:5]] + good[5:]
    failing = qualify_pairwise(biased, minimum_agreement=0.8, maximum_order_disagreement=0.2)
    assert failing["agreement"] == 0.9
    assert failing["order_disagreement_rate"] == 0.5
    assert failing["qualified"] is False


def test_the_pairwise_judge_refuses_to_compare_before_loading() -> None:
    judge = PairwiseJudge("llm-blender/PairRM", "5" * 40)

    assert judge.loaded is False
    with pytest.raises(RuntimeError, match="load\\(\\)"):
        judge.compare("prompt", "a", "b")


def test_the_pairwise_judge_identity_declares_both_orders() -> None:
    identity = PairwiseJudge("llm-blender/PairRM", "5" * 40).identity()

    assert identity["both_orders"] is True
    assert "never" in identity["network_access"]
