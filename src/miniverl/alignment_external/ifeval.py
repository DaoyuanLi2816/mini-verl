"""Independent deterministic verifier for the IFEval instruction set.

There is no pip-installable official IFEval scorer -- the ``ifeval`` name on
PyPI is an unrelated package that evaluates Python ``if`` statements -- so this
is an independent implementation of the 25 instruction types that appear in
``google/IFEval`` at the pinned revision, written from the instruction
semantics rather than copied from the reference source.

Two consequences are reported rather than hidden:

* an instruction type this module does not implement scores ``not_applicable``,
  never ``False``. A missing verifier is missing evidence, not a failure;
* ``fidelity`` marks each verifier ``exact`` or ``approximate``. The reference
  scorer uses ``nltk`` for sentence segmentation and ``langdetect`` for
  language identification; where this module substitutes something else, the
  results say so and the affected instruction ids are listed in the report.

Both IFEval metrics are produced. ``strict`` scores the response as generated.
``loose`` retries against the same eight response variants the reference uses
-- with the first line, the last line, both, and markdown asterisks removed --
and passes if any variant satisfies the instruction.
"""

from __future__ import annotations

import json
import re
import string
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = [
    "IFEVAL_SUPPORTED_INSTRUCTIONS",
    "evaluate_ifeval_response",
    "loose_variants",
    "verify_instruction",
]

_COMPARISONS: dict[str, Callable[[int, int], bool]] = {
    "at least": lambda actual, expected: actual >= expected,
    "less than": lambda actual, expected: actual < expected,
    "at most": lambda actual, expected: actual <= expected,
    "exactly": lambda actual, expected: actual == expected,
}

#: The reference scorer's fixed answer set for `detectable_format:constrained_response`.
_CONSTRAINED_RESPONSES = ("My answer is yes.", "My answer is no.", "My answer is maybe.")

_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
_WORD = re.compile(r"\b\w+\b")
_BULLET = re.compile(r"^\s*[*-]\s+\S", re.MULTILINE)
_HIGHLIGHT = re.compile(r"\*[^\*\n]+\*")
_TITLE = re.compile(r"<<[^\n]+>>")
_PLACEHOLDER = re.compile(r"\[[^\[\]\n]*\]")


def _relation(value: int, expected: int, relation: str) -> bool:
    compare = _COMPARISONS.get(str(relation).strip().lower())
    if compare is None:
        raise ValueError(f"unknown IFEval relation {relation!r}")
    return compare(value, expected)


def _count_words(text: str) -> int:
    return len(_WORD.findall(text))


def _count_sentences(text: str) -> int:
    """Sentence count.

    Approximate: the reference scorer segments with ``nltk``. This counts
    terminal punctuation runs followed by whitespace or end of string, which
    agrees on ordinary prose and diverges on abbreviations such as "Dr.".
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return len(_SENTENCE_END.findall(stripped)) or 1


def _paragraphs(text: str) -> list[str]:
    """Paragraphs as the reference splits them: on a `***` divider line."""
    parts = re.split(r"\n\s*\*\*\*\s*\n", text.strip())
    return [part for part in (item.strip() for item in parts) if part]


# --------------------------------------------------------------- verifiers


def _keywords_existence(response: str, kwargs: Mapping[str, Any]) -> bool:
    lowered = response.lower()
    return all(str(word).lower() in lowered for word in kwargs["keywords"])


def _keywords_forbidden(response: str, kwargs: Mapping[str, Any]) -> bool:
    lowered = response.lower()
    return all(str(word).lower() not in lowered for word in kwargs["forbidden_words"])


def _keywords_frequency(response: str, kwargs: Mapping[str, Any]) -> bool:
    keyword = str(kwargs["keyword"]).lower()
    count = len(re.findall(re.escape(keyword), response.lower()))
    return _relation(count, int(kwargs["frequency"]), str(kwargs["relation"]))


def _letter_frequency(response: str, kwargs: Mapping[str, Any]) -> bool:
    letter = str(kwargs["letter"]).lower()
    count = response.lower().count(letter)
    return _relation(count, int(kwargs["let_frequency"]), str(kwargs["let_relation"]))


def _number_words(response: str, kwargs: Mapping[str, Any]) -> bool:
    return _relation(_count_words(response), int(kwargs["num_words"]), str(kwargs["relation"]))


def _number_sentences(response: str, kwargs: Mapping[str, Any]) -> bool:
    return _relation(
        _count_sentences(response), int(kwargs["num_sentences"]), str(kwargs["relation"])
    )


def _number_paragraphs(response: str, kwargs: Mapping[str, Any]) -> bool:
    return len(_paragraphs(response)) == int(kwargs["num_paragraphs"])


def _nth_paragraph_first_word(response: str, kwargs: Mapping[str, Any]) -> bool:
    paragraphs = _paragraphs(response)
    if len(paragraphs) != int(kwargs["num_paragraphs"]):
        return False
    index = int(kwargs["nth_paragraph"]) - 1
    if not 0 <= index < len(paragraphs):
        return False
    first = paragraphs[index].split()
    if not first:
        return False
    return first[0].strip(string.punctuation).lower() == str(kwargs["first_word"]).lower()


def _no_comma(response: str, _kwargs: Mapping[str, Any]) -> bool:
    return "," not in response


def _english_capital(response: str, _kwargs: Mapping[str, Any]) -> bool:
    return response == response.upper()


def _english_lowercase(response: str, _kwargs: Mapping[str, Any]) -> bool:
    return response == response.lower()


def _capital_word_frequency(response: str, kwargs: Mapping[str, Any]) -> bool:
    words = _WORD.findall(response)
    count = sum(1 for word in words if word.isupper() and word.isalpha())
    return _relation(count, int(kwargs["capital_frequency"]), str(kwargs["capital_relation"]))


def _title(response: str, _kwargs: Mapping[str, Any]) -> bool:
    return bool(_TITLE.search(response))


def _number_highlighted(response: str, kwargs: Mapping[str, Any]) -> bool:
    highlights = [item for item in _HIGHLIGHT.findall(response) if item.strip("*").strip()]
    return len(highlights) >= int(kwargs["num_highlights"])


def _number_bullets(response: str, kwargs: Mapping[str, Any]) -> bool:
    return len(_BULLET.findall(response)) == int(kwargs["num_bullets"])


def _number_placeholders(response: str, kwargs: Mapping[str, Any]) -> bool:
    return len(_PLACEHOLDER.findall(response)) >= int(kwargs["num_placeholders"])


def _postscript(response: str, kwargs: Mapping[str, Any]) -> bool:
    marker = str(kwargs["postscript_marker"])
    # The reference matches the marker case-insensitively anywhere after the
    # body, tolerating the "P.P.S" / "P.P.S." spelling difference.
    pattern = re.escape(marker).replace(r"\.", r"\.?")
    return bool(re.search(pattern, response, flags=re.IGNORECASE))


def _json_format(response: str, _kwargs: Mapping[str, Any]) -> bool:
    text = response.strip()
    for fence in ("```json", "```JSON", "```"):
        if text.startswith(fence):
            text = text[len(fence) :]
            break
    if text.endswith("```"):
        text = text[: -len("```")]
    try:
        json.loads(text.strip())
    except (ValueError, TypeError):
        return False
    return True


def _multiple_sections(response: str, kwargs: Mapping[str, Any]) -> bool:
    splitter = str(kwargs["section_spliter"])
    matches = re.findall(rf"{re.escape(splitter)}\s*\d+", response)
    return len(matches) >= int(kwargs["num_sections"])


def _constrained_response(response: str, _kwargs: Mapping[str, Any]) -> bool:
    stripped = response.strip()
    return any(option in stripped for option in _CONSTRAINED_RESPONSES)


def _two_responses(response: str, _kwargs: Mapping[str, Any]) -> bool:
    parts = [part.strip() for part in response.split("******")]
    return len(parts) == 2 and all(parts)


def _repeat_prompt(response: str, kwargs: Mapping[str, Any]) -> bool:
    expected = str(kwargs["prompt_to_repeat"]).strip()
    return response.strip().startswith(expected)


def _end_checker(response: str, kwargs: Mapping[str, Any]) -> bool:
    end_phrase = str(kwargs["end_phrase"]).strip().lower()
    return response.strip().lower().endswith(end_phrase)


def _quotation(response: str, _kwargs: Mapping[str, Any]) -> bool:
    stripped = response.strip()
    return len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"')


def _response_language(response: str, kwargs: Mapping[str, Any]) -> bool:
    """Language identification, which needs `langdetect` to be meaningful."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    # Deterministic output; langdetect is randomised by default.
    DetectorFactory.seed = 0
    try:
        return detect(response) == str(kwargs["language"])
    except LangDetectException:
        # No detectable features -- an empty or punctuation-only response.
        # That does not satisfy "answer in language X", so it is a failed
        # constraint rather than an unrunnable verifier. This fires on the
        # loose variants, where stripping the first and last line can empty
        # a short response.
        return False


#: instruction id -> (verifier, fidelity). ``approximate`` means this module
#: substitutes something for a reference dependency and may disagree at the
#: margin; the difference is reported, never silently absorbed.
IFEVAL_SUPPORTED_INSTRUCTIONS: dict[str, tuple[Callable[..., bool], str]] = {
    "keywords:existence": (_keywords_existence, "exact"),
    "keywords:forbidden_words": (_keywords_forbidden, "exact"),
    "keywords:frequency": (_keywords_frequency, "exact"),
    "keywords:letter_frequency": (_letter_frequency, "exact"),
    "length_constraints:number_words": (_number_words, "exact"),
    "length_constraints:number_sentences": (_number_sentences, "approximate"),
    "length_constraints:number_paragraphs": (_number_paragraphs, "exact"),
    "length_constraints:nth_paragraph_first_word": (_nth_paragraph_first_word, "exact"),
    "punctuation:no_comma": (_no_comma, "exact"),
    "change_case:english_capital": (_english_capital, "exact"),
    "change_case:english_lowercase": (_english_lowercase, "exact"),
    "change_case:capital_word_frequency": (_capital_word_frequency, "exact"),
    "detectable_format:title": (_title, "exact"),
    "detectable_format:number_highlighted_sections": (_number_highlighted, "exact"),
    "detectable_format:number_bullet_lists": (_number_bullets, "exact"),
    "detectable_format:json_format": (_json_format, "exact"),
    "detectable_format:multiple_sections": (_multiple_sections, "exact"),
    "detectable_format:constrained_response": (_constrained_response, "exact"),
    "detectable_content:number_placeholders": (_number_placeholders, "exact"),
    "detectable_content:postscript": (_postscript, "exact"),
    "combination:two_responses": (_two_responses, "exact"),
    "combination:repeat_prompt": (_repeat_prompt, "exact"),
    "startend:end_checker": (_end_checker, "exact"),
    "startend:quotation": (_quotation, "exact"),
    "language:response_language": (_response_language, "approximate"),
}


def loose_variants(response: str) -> list[str]:
    """The eight response variants the reference loose metric tries."""
    lines = response.split("\n")
    without_first = "\n".join(lines[1:]).strip()
    without_last = "\n".join(lines[:-1]).strip()
    without_both = "\n".join(lines[1:-1]).strip()
    base = [response, without_first, without_last, without_both]
    return base + [item.replace("*", "") for item in base]


def verify_instruction(
    instruction_id: str, response: str, kwargs: Mapping[str, Any]
) -> tuple[bool | None, str]:
    """Return ``(satisfied, status)`` for one instruction.

    ``satisfied`` is ``None`` when the instruction type has no verifier or the
    verifier could not run, and ``status`` names why. It is never coerced to
    ``False``.
    """
    entry = IFEVAL_SUPPORTED_INSTRUCTIONS.get(instruction_id)
    if entry is None:
        return None, "not_applicable: no verifier for this instruction type"
    verifier, _fidelity = entry
    supplied = {key: value for key, value in kwargs.items() if value is not None}
    try:
        return bool(verifier(response, supplied)), "evaluated"
    except ImportError as exc:
        return None, f"not_applicable: verifier dependency missing ({exc.name})"
    except (KeyError, ValueError, TypeError) as exc:
        return None, f"not_applicable: verifier could not run ({type(exc).__name__}: {exc})"


def evaluate_ifeval_response(
    response: str,
    instruction_ids: Sequence[str],
    instruction_kwargs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score one IFEval example under both the strict and loose metrics."""
    if len(instruction_ids) != len(instruction_kwargs):
        raise ValueError("instruction_ids and instruction_kwargs must be the same length")

    variants = loose_variants(response)
    per_instruction: list[dict[str, Any]] = []
    for instruction_id, kwargs in zip(instruction_ids, instruction_kwargs, strict=True):
        strict, status = verify_instruction(instruction_id, response, kwargs)
        if strict is None:
            loose: bool | None = None
        elif strict:
            loose = True
        else:
            loose = any(
                verify_instruction(instruction_id, variant, kwargs)[0] is True
                for variant in variants
            )
        entry = IFEVAL_SUPPORTED_INSTRUCTIONS.get(instruction_id)
        per_instruction.append(
            {
                "instruction_id": instruction_id,
                "strict": strict,
                "loose": loose,
                "status": status,
                "fidelity": entry[1] if entry else "unimplemented",
            }
        )

    evaluated = [item for item in per_instruction if item["strict"] is not None]
    inapplicable = [item for item in per_instruction if item["strict"] is None]
    return {
        "instructions": per_instruction,
        "instructions_total": len(per_instruction),
        "instructions_evaluated": len(evaluated),
        "instructions_not_applicable": len(inapplicable),
        # Prompt-level: every *evaluated* instruction satisfied. An example with
        # an unimplemented instruction cannot claim a prompt-level pass, so it
        # reports None instead of a pass over a partial set.
        "strict_prompt_level": (
            None if inapplicable or not evaluated else all(item["strict"] for item in evaluated)
        ),
        "loose_prompt_level": (
            None if inapplicable or not evaluated else all(item["loose"] for item in evaluated)
        ),
        "strict_instructions_satisfied": sum(1 for item in evaluated if item["strict"]),
        "loose_instructions_satisfied": sum(1 for item in evaluated if item["loose"]),
        "approximate_instruction_ids": sorted(
            {
                item["instruction_id"]
                for item in per_instruction
                if item["fidelity"] == "approximate"
            }
        ),
    }
