"""Every IFEval verifier is checked against a satisfying and a violating case.

This is an independent implementation, so it earns trust only from explicit
cases. Each instruction type gets both directions; an instruction with no
verifier must report `not_applicable` rather than `False`, because a missing
verifier is missing evidence.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from miniverl.alignment_external.ifeval import (
    IFEVAL_SUPPORTED_INSTRUCTIONS,
    evaluate_ifeval_response,
    loose_variants,
    verify_instruction,
)

# (instruction_id, kwargs, satisfying response, violating response)
CASES: list[tuple[str, dict[str, Any], str, str]] = [
    ("keywords:existence", {"keywords": ["peace", "river"]}, "peace by the river", "only peace"),
    ("keywords:forbidden_words", {"forbidden_words": ["bad"]}, "all good here", "this is bad"),
    (
        "keywords:frequency",
        {"keyword": "peace", "frequency": 2, "relation": "at least"},
        "peace and peace",
        "peace once",
    ),
    (
        "keywords:letter_frequency",
        {"letter": "l", "let_frequency": 3, "let_relation": "at least"},
        "lolly llama",
        "none here",
    ),
    (
        "length_constraints:number_words",
        {"num_words": 5, "relation": "at least"},
        "one two three four five six",
        "one two",
    ),
    (
        "length_constraints:number_sentences",
        {"num_sentences": 2, "relation": "at least"},
        "First one. Second one.",
        "Only one.",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "First para.\n***\nSecond para.",
        "Only one paragraph.",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "booster"},
        "First para.\n***\nbooster starts here.",
        "First para.\n***\nwrong start.",
    ),
    ("punctuation:no_comma", {}, "no commas at all", "yes, there is one"),
    ("change_case:english_capital", {}, "ALL CAPS HERE", "not all caps"),
    ("change_case:english_lowercase", {}, "all lower here", "Not All Lower"),
    (
        "change_case:capital_word_frequency",
        {"capital_frequency": 2, "capital_relation": "at least"},
        "THIS IS loud",
        "this is quiet",
    ),
    ("detectable_format:title", {}, "<<A Real Title>> and body", "no title here"),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 2},
        "*one* and *two* highlighted",
        "*only one*",
    ),
    (
        "detectable_format:number_bullet_lists",
        {"num_bullets": 2},
        "* first\n* second",
        "* only one",
    ),
    ("detectable_format:json_format", {}, '{"a": 1}', "not json at all"),
    (
        "detectable_format:multiple_sections",
        {"section_spliter": "SECTION", "num_sections": 2},
        "SECTION 1\nbody\nSECTION 2\nbody",
        "SECTION 1\nbody only",
    ),
    (
        "detectable_format:constrained_response",
        {},
        "My answer is yes.",
        "Perhaps, it depends.",
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "Dear [name], see [address].",
        "Dear [name] only.",
    ),
    (
        "detectable_content:postscript",
        {"postscript_marker": "P.S."},
        "Body.\nP.S. more",
        "Body only",
    ),
    ("combination:two_responses", {}, "first answer\n******\nsecond answer", "only one answer"),
    (
        "combination:repeat_prompt",
        {"prompt_to_repeat": "Repeat this exactly"},
        "Repeat this exactly\nthen answer",
        "I will not repeat it",
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Call me at 631-481-4867"},
        "Sure. Call me at 631-481-4867",
        "Sure. Goodbye",
    ),
    ("startend:quotation", {}, '"the whole thing is quoted"', "not quoted at all"),
]


@pytest.mark.parametrize(
    ("instruction_id", "kwargs", "good", "bad"), CASES, ids=[c[0] for c in CASES]
)
def test_verifier_accepts_and_rejects(
    instruction_id: str, kwargs: dict[str, Any], good: str, bad: str
) -> None:
    assert verify_instruction(instruction_id, good, kwargs) == (True, "evaluated")
    assert verify_instruction(instruction_id, bad, kwargs) == (False, "evaluated")


def test_every_dataset_instruction_type_has_a_case() -> None:
    """The table above must cover every implemented verifier except the one
    needing an optional dependency, which has its own tests below."""
    covered = {case[0] for case in CASES}
    implemented = set(IFEVAL_SUPPORTED_INSTRUCTIONS)
    assert implemented - covered == {"language:response_language"}


# ------------------------------------------------------- not_applicable paths


def test_an_unknown_instruction_is_not_applicable_not_false() -> None:
    satisfied, status = verify_instruction("made_up:instruction", "anything", {})

    assert satisfied is None
    assert status.startswith("not_applicable")


def test_a_verifier_that_cannot_run_is_not_applicable() -> None:
    # Missing required kwarg.
    satisfied, status = verify_instruction("keywords:frequency", "text", {})

    assert satisfied is None
    assert "not_applicable" in status


def test_language_detection_runs_when_langdetect_is_installed() -> None:
    """Sentence-length text, because langdetect is unreliable on fragments.

    `"hola amigo"` alone identifies as Somali. That is a real property of the
    detector and is recorded as a limitation of this endpoint rather than
    worked around.
    """
    pytest.importorskip("langdetect")

    spanish = "Hola, me llamo Juan y vivo en Madrid desde hace muchos anos."
    assert verify_instruction("language:response_language", spanish, {"language": "es"}) == (
        True,
        "evaluated",
    )
    assert verify_instruction("language:response_language", spanish, {"language": "de"}) == (
        False,
        "evaluated",
    )


@pytest.mark.parametrize("empty", ["", "   ", "...", "!!!"])
def test_a_featureless_response_fails_the_language_constraint(empty: str) -> None:
    """Found by running the real 541-row dataset, not by a synthetic case.

    langdetect raises when the text has no detectable features. A loose variant
    can be empty after the first and last lines are stripped, and an empty
    response does not answer in Arabic -- so this is a failed constraint, not
    an unrunnable verifier, and it must not propagate.
    """
    pytest.importorskip("langdetect")

    assert verify_instruction("language:response_language", empty, {"language": "ar"}) == (
        False,
        "evaluated",
    )


def test_a_short_response_survives_loose_variant_stripping() -> None:
    """The exact shape that crashed: loose retries empty out a 2-line answer."""
    pytest.importorskip("langdetect")

    result = evaluate_ifeval_response(
        "Intro line\nnope", ["language:response_language"], [{"language": "ar"}]
    )

    assert result["instructions"][0]["strict"] is False
    assert result["instructions"][0]["loose"] is False
    assert result["instructions_not_applicable"] == 0


def test_a_missing_optional_dependency_is_not_applicable(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langdetect":
            raise ImportError("No module named 'langdetect'", name="langdetect")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    satisfied, status = verify_instruction("language:response_language", "hola", {"language": "es"})

    assert satisfied is None
    assert "dependency missing" in status


# ------------------------------------------------------------- loose metric


def test_loose_variants_are_the_reference_eight() -> None:
    variants = loose_variants("intro\n*body*\noutro")

    assert len(variants) == 8
    assert "intro\n*body*\noutro" in variants
    assert "*body*\noutro" in variants  # first line removed
    assert "intro\n*body*" in variants  # last line removed
    assert "body" in variants  # both removed and asterisks stripped


def test_loose_passes_where_strict_fails_on_a_wrapper_line() -> None:
    """A preamble line is exactly what the loose metric forgives."""
    response = "Sure, here you go:\n" + '"the quoted answer"'

    result = evaluate_ifeval_response(response, ["startend:quotation"], [{}])

    assert result["instructions"][0]["strict"] is False
    assert result["instructions"][0]["loose"] is True
    assert result["strict_prompt_level"] is False
    assert result["loose_prompt_level"] is True


# ------------------------------------------------------------ example scoring


def test_all_instructions_satisfied_is_a_prompt_level_pass() -> None:
    result = evaluate_ifeval_response(
        "all lower and no commas",
        ["change_case:english_lowercase", "punctuation:no_comma"],
        [{}, {}],
    )

    assert result["strict_prompt_level"] is True
    assert result["strict_instructions_satisfied"] == 2
    assert result["instructions_not_applicable"] == 0


def test_an_unimplemented_instruction_blocks_a_prompt_level_claim() -> None:
    """A pass over a partial instruction set is not a pass."""
    result = evaluate_ifeval_response(
        "all lower here",
        ["change_case:english_lowercase", "made_up:instruction"],
        [{}, {}],
    )

    assert result["instructions_not_applicable"] == 1
    assert result["strict_prompt_level"] is None
    assert result["loose_prompt_level"] is None
    # The satisfied one still counts at instruction level.
    assert result["strict_instructions_satisfied"] == 1


def test_approximate_verifiers_are_named_in_the_result() -> None:
    result = evaluate_ifeval_response(
        "One sentence. Two sentences.",
        ["length_constraints:number_sentences"],
        [{"num_sentences": 2, "relation": "at least"}],
    )

    assert result["approximate_instruction_ids"] == ["length_constraints:number_sentences"]


def test_mismatched_instruction_and_kwarg_lengths_fail_loudly() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_ifeval_response("text", ["punctuation:no_comma"], [])


def test_json_format_tolerates_a_code_fence() -> None:
    assert (
        verify_instruction("detectable_format:json_format", '```json\n{"a": 1}\n```', {})[0] is True
    )
    assert json.loads('{"a": 1}') == {"a": 1}
