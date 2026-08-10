"""The continuation training set is parsed, bounded and split deterministically."""

from __future__ import annotations

from typing import Any

import pytest

from miniverl.alignment_external.training_data import (
    assign_split,
    build_examples,
    parse_transcript,
    summarize,
)

PROMPT = "\n\nHuman: what is the capital of France?"
GOOD = "\n\nAssistant: Paris."
BAD = "\n\nAssistant: I refuse to say."


def _row(chosen: str = PROMPT + GOOD, rejected: str = PROMPT + BAD) -> dict[str, Any]:
    return {"chosen": chosen, "rejected": rejected}


def _build(rows: list[dict[str, Any]], **kwargs: Any) -> list:
    kwargs.setdefault("max_prompt_characters", 4000)
    kwargs.setdefault("max_response_characters", 2000)
    return list(build_examples(rows, **kwargs))


# --------------------------------------------------------------------- parsing


def test_a_transcript_splits_into_turns_and_a_final_reply() -> None:
    parsed = parse_transcript(
        "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: more?\n\nAssistant: sure"
    )

    assert parsed is not None
    turns, final = parsed
    assert turns == [("Human", "hi"), ("Assistant", "hello"), ("Human", "more?")]
    assert final == "sure"


@pytest.mark.parametrize(
    "transcript",
    [
        "",
        "no markers at all",
        "\n\nHuman: dangling question",  # ends on a human turn
        "\n\nHuman: hi\n\nAssistant:   ",  # empty final reply
    ],
)
def test_a_malformed_transcript_is_dropped_not_repaired(transcript: str) -> None:
    assert parse_transcript(transcript) is None


def test_a_pair_whose_branches_disagree_on_the_prompt_is_dropped() -> None:
    """The pair only means something if both branches answer the same thing."""
    rows = [_row(rejected="\n\nHuman: a different question\n\nAssistant: no")]

    assert _build(rows) == []


def test_an_identical_pair_is_dropped() -> None:
    assert _build([_row(rejected=PROMPT + GOOD)]) == []


# --------------------------------------------------------------------- bounds


def test_an_over_long_prompt_is_dropped_not_truncated() -> None:
    long_prompt = "\n\nHuman: " + "x" * 500

    assert (
        _build(
            [_row(chosen=long_prompt + GOOD, rejected=long_prompt + BAD)], max_prompt_characters=100
        )
        == []
    )


def test_an_over_long_response_is_dropped_not_truncated() -> None:
    long_good = "\n\nAssistant: " + "y" * 500

    assert _build([_row(chosen=PROMPT + long_good)], max_response_characters=100) == []


def test_a_bounded_example_survives_intact() -> None:
    built = _build([_row()])

    assert len(built) == 1
    example = built[0]
    assert example.prompt == "Human: what is the capital of France?"
    assert example.chosen == "Paris."
    assert example.rejected == "I refuse to say."
    assert example.turns == 1


# ---------------------------------------------------------------------- splits


def test_the_split_is_stable_for_the_same_identity() -> None:
    assert assign_split("abc123") == assign_split("abc123")


def test_the_split_does_not_depend_on_row_order() -> None:
    """Upstream reordering must not move the train/eval boundary."""
    rows = [
        _row(
            chosen=f"\n\nHuman: q{index}\n\nAssistant: yes",
            rejected=f"\n\nHuman: q{index}\n\nAssistant: no",
        )
        for index in range(40)
    ]

    forward = {e.example_id: e.split for e in _build(rows)}
    backward = {e.example_id: e.split for e in _build(list(reversed(rows)))}

    assert forward == backward
    assert len(forward) == 40


def test_both_splits_are_populated_at_the_configured_fraction() -> None:
    rows = [
        _row(
            chosen=f"\n\nHuman: q{index}\n\nAssistant: yes",
            rejected=f"\n\nHuman: q{index}\n\nAssistant: no",
        )
        for index in range(400)
    ]

    counts: dict[str, int] = {}
    for example in _build(rows, eval_fraction=0.25):
        counts[example.split] = counts.get(example.split, 0) + 1

    assert set(counts) == {"train", "eval"}
    # Hash bucketing, so approximate rather than exact.
    assert 0.18 < counts["eval"] / sum(counts.values()) < 0.32


def test_an_out_of_range_eval_fraction_is_refused() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="strictly between"):
            assign_split("abc", eval_fraction=bad)


# -------------------------------------------------------------------- summary


def test_the_summary_records_provenance_and_a_content_digest() -> None:
    examples = _build(
        [
            _row(
                chosen=f"\n\nHuman: q{index}\n\nAssistant: yes",
                rejected=f"\n\nHuman: q{index}\n\nAssistant: no",
            )
            for index in range(10)
        ]
    )

    summary = summarize(examples)

    assert summary["dataset"] == "Anthropic/hh-rlhf"
    assert summary["license"] == "mit"
    assert len(summary["revision"]) == 40
    assert summary["total"] == 10
    assert sum(summary["by_split"].values()) == 10
    assert len(summary["content_digest"]) == 64
    assert "never truncated" in summary["length_policy"]


def test_the_content_digest_is_order_independent() -> None:
    rows = [
        _row(
            chosen=f"\n\nHuman: q{index}\n\nAssistant: yes",
            rejected=f"\n\nHuman: q{index}\n\nAssistant: no",
        )
        for index in range(12)
    ]

    forward = summarize(_build(rows))["content_digest"]
    backward = summarize(list(reversed(_build(rows))))["content_digest"]

    assert forward == backward
