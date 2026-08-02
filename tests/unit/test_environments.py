"""Environment contract: determinism, split disjointness, oracles and sandboxing.

This file protects the four promises the :mod:`miniverl.environments` package
makes to the rest of the trainer:

1. **Determinism.**  ``generate_task(index, seed, ...)`` is a pure function of
   its arguments, so a run can be replayed from its manifest, and a different
   seed really does move to a different problem.
2. **Disjoint splits.**  ``make_splits`` never lets a train prompt reappear in
   eval or test, at any difficulty, for any environment -- otherwise every
   reported eval number is contaminated.
3. **A correct oracle.**  Replaying ``oracle_actions`` through
   ``reset``/``step``/``verify`` solves the task with *no* failed intermediate
   step, and the last tool output really carries the answer.  A broken oracle
   would silently poison the SFT cold start and the privileged-context teacher.
4. **Enforced sandboxes.**  The calculator evaluates arithmetic without
   executing Python (no imports, calls, names, comprehensions, f-strings, and
   no unbounded exponentiation), and the SQLite environment is read-only by
   engine authorizer, not by hopeful string matching.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from miniverl.environments.base import (
    FailureCategory,
    OracleActionKind,
    Task,
    ToolCall,
    ToolEnvironment,
    VerificationResult,
    make_splits,
    unique_prompts,
)
from miniverl.environments.calculator import (
    MAX_EXPRESSION_CHARS,
    CalculatorEnvironment,
    normalize_number,
    safe_eval,
)
from miniverl.environments.jsonnav import (
    MAX_PATH_CHARS,
    MAX_RESULTS,
    parse_path,
    resolve_path,
)
from miniverl.environments.registry import (
    ENVIRONMENT_NAMES,
    available_environments,
    make_environment,
)
from miniverl.environments.sqlite_env import MAX_ROWS, MAX_SQL_CHARS
from miniverl.errors import ConfigError, ToolEnvironmentError

ENV_NAMES = ("calculator", "jsonnav", "sqlite")
DIFFICULTIES = ("easy", "medium", "hard")
ALL_CASES = [(name, difficulty) for name in ENV_NAMES for difficulty in DIFFICULTIES]
JSON_VALUE = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**100), max_value=10**100)
    | st.floats(allow_nan=False, allow_infinity=False, width=64)
    | st.text(max_size=80),
    lambda children: (
        st.lists(children, max_size=6) | st.dictionaries(st.text(max_size=20), children, max_size=6)
    ),
    max_leaves=20,
)


@contextmanager
def environment(name: str, **params: Any) -> Iterator[ToolEnvironment]:
    """Yield a fresh registered environment and release its episode resources."""
    env = make_environment(name, **params)
    try:
        yield env
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()


def replay_oracle(env: ToolEnvironment, task: Task) -> tuple[VerificationResult, str]:
    """Drive ``oracle_actions`` through the episode, asserting every step succeeded."""
    env.reset(task)
    last_result = ""
    verification: VerificationResult | None = None
    for action in env.oracle_actions(task):
        if action.kind is OracleActionKind.TOOL_CALL:
            assert action.tool_name is not None
            step = env.step(ToolCall(name=action.tool_name, arguments=dict(action.arguments)))
            assert step.ok, f"{task.task_id}: tool {action.tool_name} failed: {step.error}"
            assert step.error is None
            assert step.failure_category is None
            last_result = step.result
        else:
            assert action.answer is not None
            verification = env.verify(action.answer)
    assert verification is not None, f"{task.task_id}: oracle trace has no final action"
    return verification, last_result


def same_number_or_text(left: str, right: str) -> bool:
    """Compare two answer strings numerically when possible, else case-insensitively."""
    try:
        return abs(float(left) - float(right)) <= 1e-6 * max(1.0, abs(float(right)))
    except ValueError:
        return left.strip().casefold() == right.strip().casefold()


# -- A. determinism -------------------------------------------------------


@pytest.mark.parametrize(("name", "difficulty"), ALL_CASES)
def test_generate_task_is_deterministic(name: str, difficulty: str) -> None:
    """Two independent instances must produce byte-identical tasks."""
    with environment(name) as first, environment(name) as second:
        for index in range(6):
            a = first.generate_task(index, 1234, difficulty=difficulty, split="train")
            b = second.generate_task(index, 1234, difficulty=difficulty, split="train")
            assert a == b
            assert a.difficulty == difficulty
            assert a.split == "train"
            assert a.answer != ""
            # Regenerating on the same instance must not drift either.
            assert first.generate_task(index, 1234, difficulty=difficulty, split="train") == a


@pytest.mark.parametrize(("name", "difficulty"), ALL_CASES)
def test_different_seeds_give_different_prompts(name: str, difficulty: str) -> None:
    with environment(name) as env:
        prompts = [
            env.generate_task(0, seed, difficulty=difficulty, split="train").prompt
            for seed in range(1, 9)
        ]
    assert len(set(prompts)) == len(prompts)


@pytest.mark.parametrize(("name", "difficulty"), ALL_CASES)
def test_split_name_does_not_change_the_problem(name: str, difficulty: str) -> None:
    """``split`` labels a task; it must not silently change what is being asked."""
    with environment(name) as env:
        train = env.generate_task(5, 77, difficulty=difficulty, split="train")
        test = env.generate_task(5, 77, difficulty=difficulty, split="test")
    assert train.prompt == test.prompt
    assert train.answer == test.answer
    assert train.task_id != test.task_id


# -- B. splits ------------------------------------------------------------


@pytest.mark.parametrize(("name", "difficulty"), ALL_CASES)
def test_make_splits_sizes_are_exact_and_prompts_disjoint(name: str, difficulty: str) -> None:
    counts = {"train": 12, "eval": 6, "test": 5}
    with environment(name) as env:
        splits = make_splits(env, counts=counts, seed=3, difficulty=difficulty)
    assert sorted(splits) == ["eval", "test", "train"]
    for split, wanted in counts.items():
        assert len(splits[split]) == wanted
        assert unique_prompts(splits[split]) == wanted
        assert all(task.split == split for task in splits[split])
        assert all(task.difficulty == difficulty for task in splits[split])
    prompts = {split: {task.prompt for task in tasks} for split, tasks in splits.items()}
    assert not prompts["train"] & prompts["eval"]
    assert not prompts["train"] & prompts["test"]
    assert not prompts["eval"] & prompts["test"]


@pytest.mark.parametrize(("name", "difficulty"), ALL_CASES)
def test_make_splits_is_reproducible(name: str, difficulty: str) -> None:
    counts = {"train": 7, "eval": 4, "test": 4}
    with environment(name) as env:
        first = make_splits(env, counts=counts, seed=99, difficulty=difficulty)
    with environment(name) as env:
        second = make_splits(env, counts=counts, seed=99, difficulty=difficulty)
    assert first == second


@pytest.mark.parametrize("name", ENV_NAMES)
def test_make_splits_honours_a_zero_count(name: str) -> None:
    with environment(name) as env:
        splits = make_splits(env, counts={"train": 3, "eval": 0}, seed=5)
    assert len(splits["train"]) == 3
    assert splits["eval"] == []
    assert splits["test"] == []


# -- C. oracles -----------------------------------------------------------


@pytest.mark.parametrize(("name", "difficulty"), ALL_CASES)
def test_oracle_solves_every_task(name: str, difficulty: str) -> None:
    """The reference trace must solve 100% of tasks with no failed step."""
    with environment(name) as env:
        for index in range(10):
            task = env.generate_task(index, 2026, difficulty=difficulty, split="train")
            verification, last_result = replay_oracle(env, task)
            assert verification.solved, f"{task.task_id}: {verification.detail}"
            assert verification.reward == pytest.approx(1.0)
            assert verification.failure_category is FailureCategory.SOLVED
            assert last_result != ""


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_calculator_oracle_last_tool_output_is_the_answer(difficulty: str) -> None:
    """The final tool result must already be gradeable, not just the pre-baked answer."""
    with environment("calculator") as env:
        for index in range(10):
            task = env.generate_task(index, 8, difficulty=difficulty, split="train")
            _, last_result = replay_oracle(env, task)
            assert env.verify(last_result).solved, f"{task.task_id}: {last_result!r}"


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_jsonnav_oracle_last_tool_output_is_the_answer(difficulty: str) -> None:
    with environment("jsonnav") as env:
        for index in range(10):
            task = env.generate_task(index, 8, difficulty=difficulty, split="train")
            _, last_result = replay_oracle(env, task)
            assert last_result == task.answer
            assert env.verify(last_result).solved


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_sqlite_oracle_query_returns_the_answer(difficulty: str) -> None:
    with environment("sqlite") as env:
        for index in range(10):
            task = env.generate_task(index, 8, difficulty=difficulty, split="train")
            _, last_result = replay_oracle(env, task)
            rows = json.loads(last_result)
            assert isinstance(rows, list) and rows
            values = [str(value) for value in rows[0].values()]
            assert any(same_number_or_text(value, task.answer) for value in values), (
                f"{task.task_id}: {rows[0]} does not contain {task.answer!r}"
            )


@pytest.mark.parametrize(("name", "difficulty"), ALL_CASES)
def test_a_wrong_answer_is_not_solved(name: str, difficulty: str) -> None:
    """The verifier must be able to say no; otherwise the oracle test proves nothing."""
    with environment(name) as env:
        task = env.generate_task(0, 4, difficulty=difficulty, split="train")
        env.reset(task)
        verification = env.verify("-987654321")
        assert not verification.solved
        assert verification.reward == pytest.approx(0.0)
        assert verification.failure_category is not FailureCategory.SOLVED


@pytest.mark.parametrize("name", ENV_NAMES)
def test_verify_before_reset_is_rejected(name: str) -> None:
    with environment(name) as env, pytest.raises(ToolEnvironmentError, match="before reset"):
        env.verify("1")


@pytest.mark.parametrize("answer", ["nan", "NaN", "inf", "Infinity", "-inf", "1e9999"])
def test_sqlite_non_finite_final_answers_are_malformed(answer: str) -> None:
    with environment("sqlite", protocol_version="v2") as env:
        task = env.generate_task(0, 7, difficulty="easy", split="eval")
        env.reset(task)
        result = env.verify(answer)
    assert result.failure_category is FailureCategory.MALFORMED_ANSWER
    assert not result.solved


VERIFY_EDGE_CASES = (
    "nan",
    "NaN",
    "inf",
    "Infinity",
    "-inf",
    "1e9999",
    "9" * 10_000,
    "",
    "   ",
    "日本語 λ 🚀",
    "the answer is probably four",
)


@pytest.mark.parametrize("name", ENV_NAMES)
@pytest.mark.parametrize("answer", VERIFY_EDGE_CASES)
@pytest.mark.parametrize("protocol_version", ["v1", "v2"])
def test_builtin_verifiers_are_total_for_adversarial_strings(
    name: str,
    answer: str,
    protocol_version: str,
) -> None:
    with environment(name, protocol_version=protocol_version) as env:
        task = env.generate_task(0, 11, difficulty="easy", split="eval")
        env.reset(task)
        result = env.verify(answer)
    assert isinstance(result, VerificationResult)


@settings(max_examples=100, deadline=None)
@given(answer=st.text(max_size=500))
@pytest.mark.parametrize("name", ENV_NAMES)
def test_builtin_verifiers_are_total_for_arbitrary_strings(name: str, answer: str) -> None:
    for protocol_version in ("v1", "v2"):
        with environment(name, protocol_version=protocol_version) as env:
            task = env.generate_task(0, 13, difficulty="easy", split="eval")
            env.reset(task)
            result = env.verify(answer)
        assert isinstance(result, VerificationResult)


# -- D. privileged context ------------------------------------------------


@pytest.mark.parametrize(("name", "difficulty"), ALL_CASES)
def test_privileged_context_is_non_empty_and_contains_the_answer(
    name: str, difficulty: str
) -> None:
    with environment(name) as env:
        for index in range(4):
            task = env.generate_task(index, 31, difficulty=difficulty, split="eval")
            context = env.privileged_context(task)
            assert isinstance(context, str)
            assert context.strip() != ""
            assert task.answer in context
            # A privileged hint that merely repeats the prompt would be useless.
            assert context != task.prompt


# -- E. calculator sandbox ------------------------------------------------

UNSAFE_EXPRESSIONS = [
    ("__import__('os').system('echo hi')", "Call is not allowed"),
    ("open('x')", "Call is not allowed"),
    ("os.system('x')", "Call is not allowed"),
    ("[i for i in range(3)]", "ListComp is not allowed"),
    ("lambda: 1", "Lambda is not allowed"),
    ("x", "Name is not allowed"),
    ("1 if 1 else 2", "IfExp is not allowed"),
    ("f'{1}'", "JoinedStr is not allowed"),
    ("1;2", "could not parse expression"),
    ("{}", "Dict is not allowed"),
    ("()", "Tuple is not allowed"),
    ("1 == 1", "Compare is not allowed"),
    ("9**9**9", "exceeds the limit"),
    ("1/0", "division by zero"),
    ("1%0", "division by zero"),
    ("10**20", "exceeds the limit"),
    ("1 and 2", "BoolOp is not allowed"),
    ("not 1", "unary operator Not is not allowed"),
    ("1 << 2", "operator LShift is not allowed"),
    ("[1, 2]", "List is not allowed"),
    ("'a'", "only numeric literals are allowed"),
    ("None", "only numeric literals are allowed"),
]


@pytest.mark.parametrize(("expression", "fragment"), UNSAFE_EXPRESSIONS)
def test_safe_eval_rejects_unsafe_expression(expression: str, fragment: str) -> None:
    with pytest.raises(ToolEnvironmentError, match=fragment):
        safe_eval(expression)


@pytest.mark.parametrize("expression", ["True", "False", "-True", "1 + True"])
def test_safe_eval_rejects_boolean_literals(expression: str) -> None:
    """``bool`` is a subclass of ``int``, so it needs an explicit rejection."""
    with pytest.raises(ToolEnvironmentError, match="got bool"):
        safe_eval(expression)


def test_safe_eval_rejects_an_over_long_expression() -> None:
    too_long = "1+" * MAX_EXPRESSION_CHARS
    assert len(too_long) > MAX_EXPRESSION_CHARS
    with pytest.raises(ToolEnvironmentError, match=f"over the {MAX_EXPRESSION_CHARS} character"):
        safe_eval(too_long)


def test_safe_eval_accepts_an_expression_at_the_length_limit() -> None:
    """The length guard must bite only past the documented cap."""
    pad = (MAX_EXPRESSION_CHARS - 2) // 2
    at_cap = "(" * pad + "12" + ")" * pad
    assert len(at_cap) == MAX_EXPRESSION_CHARS
    assert safe_eval(at_cap) == pytest.approx(12.0)


@pytest.mark.parametrize("expression", ["-" * 13 + "1", "1+" * 13 + "1"])
def test_safe_eval_rejects_a_too_deeply_nested_expression(expression: str) -> None:
    with pytest.raises(ToolEnvironmentError, match="nests deeper than"):
        safe_eval(expression)


@pytest.mark.parametrize(
    ("expression", "expected"), [("-" * 12 + "1", 1.0), ("1+" * 12 + "1", 13.0)]
)
def test_safe_eval_accepts_nesting_at_the_depth_limit(expression: str, expected: float) -> None:
    """The depth guard must bite only past the documented limit."""
    assert safe_eval(expression) == pytest.approx(expected)


VALID_EXPRESSIONS = [
    ("2*(3+4)", 14.0),
    ("1+2*3", 7.0),
    ("100 - 10 * 3", 70.0),
    ("2+3*4-5", 9.0),
    ("(1+2)*(3+4)", 21.0),
    ("-5+3", -2.0),
    ("+3", 3.0),
    ("-2**2", -4.0),
    ("2**3", 8.0),
    ("10/4", 2.5),
    ("7//2", 3.0),
    ("-7//2", -4.0),
    ("8//3*2", 4.0),
    ("7%3", 1.0),
    ("-7%3", 2.0),
    ("1e3", 1000.0),
]


@pytest.mark.parametrize(("expression", "expected"), VALID_EXPRESSIONS)
def test_safe_eval_computes_valid_expressions(expression: str, expected: float) -> None:
    value = safe_eval(expression)
    assert isinstance(value, float)
    assert value == pytest.approx(expected)


def test_calculator_tool_turns_an_unsafe_expression_into_a_tool_error() -> None:
    with environment("calculator") as env:
        env.reset(env.generate_task(0, 1, difficulty="easy", split="train"))
        step = env.step(ToolCall("calculator", {"expression": "__import__('os').system('x')"}))
    assert not step.ok
    assert step.failure_category is FailureCategory.TOOL_ERROR
    assert step.error is not None and "not allowed" in step.error


def test_calculator_tool_rejects_a_non_string_expression() -> None:
    with environment("calculator") as env:
        env.reset(env.generate_task(0, 1, difficulty="easy", split="train"))
        step = env.step(ToolCall("calculator", {"expression": 5}))
    assert not step.ok
    assert step.failure_category is FailureCategory.INVALID_TOOL_CALL


@pytest.mark.parametrize("name", ENV_NAMES)
def test_unknown_tool_is_categorised(name: str) -> None:
    with environment(name) as env:
        env.reset(env.generate_task(0, 1, difficulty="easy", split="train"))
        step = env.step(ToolCall("definitely_not_a_tool", {}))
    assert not step.ok
    assert step.failure_category is FailureCategory.UNKNOWN_TOOL
    assert step.error is not None and "definitely_not_a_tool" in step.error


# -- F. calculator conversion --------------------------------------------


@pytest.mark.parametrize(
    ("value", "from_unit", "to_unit", "expected"),
    [
        (5, "km", "mi", 3.106855),
        (10, "km", "mi", 6.21371),
        (100, "c", "f", 212.0),
        (0, "c", "f", 32.0),
        (212, "f", "c", 100.0),
        (98.6, "f", "c", 37.0),
        (7, "km", "km", 7.0),
        (3, "c", "c", 3.0),
    ],
)
def test_convert_values(value: float, from_unit: str, to_unit: str, expected: float) -> None:
    got = CalculatorEnvironment._convert(float(value), from_unit, to_unit)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_convert_rejects_an_unsupported_pair() -> None:
    with pytest.raises(ToolEnvironmentError, match="cannot convert"):
        CalculatorEnvironment._convert(5.0, "km", "kg")


def test_convert_tool_rejects_an_unsupported_pair() -> None:
    with environment("calculator") as env:
        env.reset(env.generate_task(0, 1, difficulty="easy", split="train"))
        step = env.step(ToolCall("convert", {"value": 5, "from_unit": "km", "to_unit": "kg"}))
    assert not step.ok
    assert step.failure_category is FailureCategory.TOOL_ERROR
    assert step.error is not None
    assert "km->mi" in step.error and "c->f" in step.error


def test_convert_tool_accepts_a_string_numeric_value() -> None:
    with environment("calculator") as env:
        env.reset(env.generate_task(0, 1, difficulty="easy", split="train"))
        text = env.step(ToolCall("convert", {"value": "5", "from_unit": "km", "to_unit": "mi"}))
        number = env.step(ToolCall("convert", {"value": 5, "from_unit": "km", "to_unit": "mi"}))
    assert text.ok
    assert text.result == number.result
    assert normalize_number(text.result) == pytest.approx(3.106855, abs=1e-4)


@pytest.mark.parametrize("value", [True, False])
def test_convert_tool_rejects_a_boolean_value(value: bool) -> None:
    """``bool`` must not sneak through the numeric check for ``value``."""
    with environment("calculator") as env:
        env.reset(env.generate_task(0, 1, difficulty="easy", split="train"))
        step = env.step(ToolCall("convert", {"value": value, "from_unit": "km", "to_unit": "mi"}))
    assert not step.ok
    assert step.failure_category is FailureCategory.INVALID_TOOL_CALL
    assert step.error is not None and "must be a number" in step.error


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), "nan", "inf", "-inf"],
)
def test_convert_tool_rejects_non_finite_values_without_crashing(value: object) -> None:
    with environment("calculator") as env:
        env.reset(env.generate_task(0, 1, difficulty="easy", split="train"))
        step = env.step(ToolCall("convert", {"value": value, "from_unit": "km", "to_unit": "mi"}))
    assert not step.ok
    assert step.failure_category in {
        FailureCategory.INVALID_TOOL_CALL,
        FailureCategory.TOOL_ERROR,
    }


def test_convert_tool_rejects_an_integer_too_large_for_float_without_crashing() -> None:
    with environment("calculator") as env:
        env.reset(env.generate_task(0, 1, difficulty="easy", split="train"))
        step = env.step(
            ToolCall("convert", {"value": 10**10_000, "from_unit": "km", "to_unit": "mi"})
        )
    assert not step.ok
    assert step.failure_category is FailureCategory.INVALID_TOOL_CALL


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "1e9999"])
def test_normalize_number_rejects_non_finite_values(text: str) -> None:
    assert normalize_number(text) is None


@pytest.mark.parametrize(
    ("answer", "solved", "category"),
    [
        ("14", True, FailureCategory.SOLVED),
        ("14.0", True, FailureCategory.SOLVED),
        ("1.4e1", True, FailureCategory.SOLVED),
        ("14 meters", True, FailureCategory.SOLVED),
        ("14 m", True, FailureCategory.SOLVED),
        ("14 miles", False, FailureCategory.WRONG_ANSWER),
        ("14 arbitrary text", False, FailureCategory.MALFORMED_ANSWER),
        ("nan", False, FailureCategory.MALFORMED_ANSWER),
        ("inf", False, FailureCategory.MALFORMED_ANSWER),
        ("-inf", False, FailureCategory.MALFORMED_ANSWER),
    ],
)
def test_calculator_verifier_v2_requires_a_complete_number_and_compatible_unit(
    answer: str,
    solved: bool,
    category: FailureCategory,
) -> None:
    env = CalculatorEnvironment(protocol_version="v2")
    env.reset(
        Task(
            task_id="conversion",
            prompt="Convert.",
            answer="14",
            metadata={"kind": "conversion", "from_unit": "ft", "to_unit": "m"},
        )
    )
    result = env.verify(answer)
    assert result.solved is solved
    assert result.failure_category is category


def test_calculator_verifier_v1_remains_historically_prefix_tolerant() -> None:
    env = CalculatorEnvironment(protocol_version="v1")
    env.reset(
        Task(task_id="legacy", prompt="Compute.", answer="14", metadata={"kind": "arithmetic"})
    )
    assert env.verify("14 historical trailing prose").solved


def test_convert_tool_identity_returns_the_input() -> None:
    with environment("calculator") as env:
        env.reset(env.generate_task(0, 1, difficulty="easy", split="train"))
        step = env.step(ToolCall("convert", {"value": 42, "from_unit": "kg", "to_unit": "kg"}))
    assert step.ok
    assert normalize_number(step.result) == pytest.approx(42.0)


# -- G. jsonnav paths -----------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a.b[2].c", ["a", "b", 2, "c"]),
        ("config", ["config"]),
        ("config.alpha.label", ["config", "alpha", "label"]),
        ("records[0].owner", ["records", 0, "owner"]),
        ("items[0][1]", ["items", 0, 1]),
        ("_private[10]", ["_private", 10]),
    ],
)
def test_parse_path_variants(path: str, expected: list[str | int]) -> None:
    assert parse_path(path) == expected


@pytest.mark.parametrize("path", ["", "   ", "$", "."])
def test_parse_path_empty_means_root(path: str) -> None:
    assert parse_path(path) == []


@pytest.mark.parametrize("path", ["1abc", "a-b", "a..b", "a[x]", "config[]", "a b", "a.[0]"])
def test_parse_path_rejects_an_invalid_segment(path: str) -> None:
    with pytest.raises(ToolEnvironmentError, match="invalid path segment"):
        parse_path(path)


def test_parse_path_rejects_an_over_long_path() -> None:
    with pytest.raises(ToolEnvironmentError, match=f"over the {MAX_PATH_CHARS} character limit"):
        parse_path("a" * (MAX_PATH_CHARS + 1))


@pytest.fixture
def document() -> dict[str, Any]:
    """A hand-written document, so path errors are checked against known keys."""
    return {
        "config": {"alpha": {"label": "alpha-1", "region": "north"}, "beta": {"label": "beta-2"}},
        "records": [{"id": "r1", "score": 3}, {"id": "r2", "score": 9}],
        "meta": {"version": 4, "pointer": "alpha"},
    }


def test_resolve_path_empty_path_returns_the_root(document: dict[str, Any]) -> None:
    assert resolve_path(document, "") is document
    assert resolve_path(document, "$") is document


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("config.alpha.label", "alpha-1"),
        ("config.beta.label", "beta-2"),
        ("records[1].id", "r2"),
        ("records[0].score", 3),
        ("meta.version", 4),
    ],
)
def test_resolve_path_reads_values(document: dict[str, Any], path: str, expected: Any) -> None:
    assert resolve_path(document, path) == expected


def test_resolve_path_missing_key_lists_the_available_keys(document: dict[str, Any]) -> None:
    with pytest.raises(ToolEnvironmentError) as excinfo:
        resolve_path(document, "config.zzz")
    message = excinfo.value.message
    assert "available keys" in message
    assert "alpha" in message
    assert "beta" in message


def test_resolve_path_rejects_indexing_a_non_list(document: dict[str, Any]) -> None:
    with pytest.raises(ToolEnvironmentError, match="is not a list, cannot index"):
        resolve_path(document, "config[0]")


def test_resolve_path_rejects_an_out_of_range_index(document: dict[str, Any]) -> None:
    with pytest.raises(ToolEnvironmentError, match="index 999 is out of range"):
        resolve_path(document, "records[999]")


def test_resolve_path_rejects_reading_a_key_off_a_scalar(document: dict[str, Any]) -> None:
    with pytest.raises(ToolEnvironmentError, match="is not an object, cannot read"):
        resolve_path(document, "meta.version.nope")


def test_find_caps_the_number_of_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """``find`` must never return an unbounded observation into the context window."""
    from miniverl.environments import jsonnav

    wide = {f"s{i}": {"dup": i, "other": i} for i in range(MAX_RESULTS + 5)}
    monkeypatch.setattr(jsonnav, "build_document", lambda seed: wide)
    with environment("jsonnav") as env:
        env.reset(Task(task_id="t", prompt="p", answer="a", metadata={"document_seed": 1}))
        step = env.step(ToolCall("find", {"key": "dup"}))
    assert step.ok
    matches = json.loads(step.result)
    assert len(matches) == MAX_RESULTS
    assert all(path.endswith(".dup") for path in matches)


def test_keys_on_a_scalar_is_an_error() -> None:
    with environment("jsonnav") as env:
        env.reset(env.generate_task(0, 5, difficulty="easy", split="train"))
        step = env.step(ToolCall("keys", {"path": "meta.version"}))
    assert not step.ok
    assert step.failure_category is FailureCategory.TOOL_ERROR
    assert step.error is not None and "scalar" in step.error


def test_keys_lists_object_keys_and_list_indices() -> None:
    with environment("jsonnav") as env:
        env.reset(env.generate_task(0, 5, difficulty="easy", split="train"))
        root = env.step(ToolCall("keys", {"path": ""}))
        records = env.step(ToolCall("keys", {"path": "records"}))
    assert root.ok
    assert json.loads(root.result) == ["config", "meta", "records"]
    assert records.ok
    indices = json.loads(records.result)
    assert indices and indices == [f"[{i}]" for i in range(len(indices))]


def test_find_rejects_an_empty_key() -> None:
    with environment("jsonnav") as env:
        env.reset(env.generate_task(0, 5, difficulty="easy", split="train"))
        step = env.step(ToolCall("find", {"key": ""}))
    assert not step.ok
    assert step.error is not None and "non-empty string" in step.error


# -- H. sqlite read-only enforcement -------------------------------------


@pytest.fixture
def sql_env() -> Iterator[ToolEnvironment]:
    """A reset SQLite environment on a seeded in-memory database."""
    with environment("sqlite") as env:
        env.reset(env.generate_task(0, 7, difficulty="easy", split="train"))
        yield env


WRITE_STATEMENTS = [
    "DROP TABLE orders",
    "INSERT INTO orders VALUES (99, 1, 5, 'x')",
    "UPDATE orders SET amount = 0",
    "DELETE FROM orders",
    "CREATE TABLE t (a INTEGER)",
    "ALTER TABLE orders RENAME TO o2",
    "ATTACH DATABASE 'evil.db' AS evil",
    "PRAGMA table_info(orders)",
    "VACUUM",
]


@pytest.mark.parametrize("sql", WRITE_STATEMENTS)
def test_write_statements_are_rejected(sql_env: ToolEnvironment, sql: str) -> None:
    step = sql_env.step(ToolCall("query", {"sql": sql}))
    assert not step.ok
    assert step.failure_category is FailureCategory.TOOL_ERROR
    assert step.error is not None and "only SELECT statements are allowed" in step.error
    # The database must still be intact afterwards.
    survivor = sql_env.step(ToolCall("query", {"sql": "SELECT count(*) AS n FROM orders"}))
    assert survivor.ok
    assert json.loads(survivor.result)[0]["n"] > 0


def test_a_write_smuggled_behind_a_cte_is_denied_by_the_authorizer(
    sql_env: ToolEnvironment,
) -> None:
    """``WITH ... INSERT`` passes the prefix check, so the engine must refuse it."""
    sql = "WITH x AS (SELECT 1 AS a) INSERT INTO orders SELECT 99, 1, 1, 'x' FROM x"
    step = sql_env.step(ToolCall("query", {"sql": sql}))
    assert not step.ok
    assert step.error is not None and "read-only" in step.error


def test_sqlite_master_is_not_readable(sql_env: ToolEnvironment) -> None:
    step = sql_env.step(ToolCall("query", {"sql": "SELECT * FROM sqlite_master"}))
    assert not step.ok
    assert step.failure_category is FailureCategory.TOOL_ERROR
    assert step.error is not None and "is not permitted" in step.error
    assert step.error is not None and "sqlite_master" in step.error


@pytest.mark.parametrize("sql", ["SELECT randomblob(4)", "SELECT load_extension('x')"])
def test_non_whitelisted_functions_are_rejected(sql_env: ToolEnvironment, sql: str) -> None:
    step = sql_env.step(ToolCall("query", {"sql": sql}))
    assert not step.ok
    assert step.error is not None and "is not permitted" in step.error


def test_two_statements_in_one_call_are_rejected(sql_env: ToolEnvironment) -> None:
    step = sql_env.step(ToolCall("query", {"sql": "SELECT 1 AS a; SELECT 2 AS b"}))
    assert not step.ok
    assert step.error is not None and "only one SQL statement per call" in step.error


def test_an_over_long_query_is_rejected(sql_env: ToolEnvironment) -> None:
    sql = "SELECT " + "1+" * MAX_SQL_CHARS + "1"
    step = sql_env.step(ToolCall("query", {"sql": sql}))
    assert not step.ok
    assert step.error is not None and f"over the {MAX_SQL_CHARS} character limit" in step.error


def test_an_empty_query_is_an_invalid_tool_call(sql_env: ToolEnvironment) -> None:
    step = sql_env.step(ToolCall("query", {"sql": "   "}))
    assert not step.ok
    assert step.failure_category is FailureCategory.INVALID_TOOL_CALL


def test_a_legitimate_aggregate_select_returns_parsed_json_rows(
    sql_env: ToolEnvironment,
) -> None:
    sql = "SELECT count(*) AS n, total(amount) AS amount_total FROM orders WHERE status = 'shipped'"
    step = sql_env.step(ToolCall("query", {"sql": sql}))
    assert step.ok
    assert step.error is None
    rows = json.loads(step.result)
    assert len(rows) == 1
    assert sorted(rows[0]) == ["amount_total", "n"]
    assert isinstance(rows[0]["n"], int)
    assert rows[0]["n"] >= 0
    if rows[0]["n"] == 0:
        assert rows[0]["amount_total"] == pytest.approx(0.0)
    else:
        assert rows[0]["amount_total"] > 0


def test_sqlite_non_finite_result_is_a_bounded_tool_error(sql_env: ToolEnvironment) -> None:
    step = sql_env.step(ToolCall("query", {"sql": "SELECT 1e400 AS too_large"}))
    assert not step.ok
    assert step.failure_category is FailureCategory.TOOL_ERROR
    assert step.error is not None and "finite JSON" in step.error


def test_a_join_select_returns_column_named_rows(sql_env: ToolEnvironment) -> None:
    sql = (
        "SELECT c.city AS city, total(o.amount) AS total FROM orders o "
        "JOIN customers c ON c.id = o.customer_id GROUP BY c.city ORDER BY total DESC LIMIT 3"
    )
    step = sql_env.step(ToolCall("query", {"sql": sql}))
    assert step.ok
    rows = json.loads(step.result)
    assert 1 <= len(rows) <= 3
    assert all(sorted(row) == ["city", "total"] for row in rows)


def test_the_row_cap_is_enforced_with_a_truncation_note(sql_env: ToolEnvironment) -> None:
    sql = "SELECT o.id AS oid FROM orders o, customers c"
    step = sql_env.step(ToolCall("query", {"sql": sql}))
    assert step.ok
    rows = json.loads(step.result)
    assert len(rows) == MAX_ROWS + 1
    assert rows[-1] == {"_note": f"result truncated to {MAX_ROWS} rows"}
    assert all(sorted(row) == ["oid"] for row in rows[:MAX_ROWS])


def test_the_schema_tool_returns_the_ddl(sql_env: ToolEnvironment) -> None:
    step = sql_env.step(ToolCall("schema", {}))
    assert step.ok
    assert "CREATE TABLE customers" in step.result
    assert "CREATE TABLE orders" in step.result


def test_sqlite_step_before_reset_is_rejected() -> None:
    with environment("sqlite") as env, pytest.raises(ToolEnvironmentError, match="before reset"):
        env.step(ToolCall("schema", {}))


@settings(max_examples=50, deadline=None)
@given(arguments=st.dictionaries(st.text(max_size=20), JSON_VALUE, max_size=8))
@pytest.mark.parametrize(
    ("environment_name", "tool_name"),
    [
        ("calculator", "calculator"),
        ("calculator", "convert"),
        ("jsonnav", "get"),
        ("jsonnav", "keys"),
        ("jsonnav", "find"),
        ("sqlite", "schema"),
        ("sqlite", "query"),
    ],
)
def test_every_environment_contains_arbitrary_json_compatible_arguments(
    environment_name: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    """Untrusted model arguments always become a bounded ``StepResult``."""
    with environment(environment_name) as env:
        task = env.generate_task(0, 7, difficulty="easy", split="train")
        env.reset(task)
        result = env.step(ToolCall(tool_name, arguments))
    assert isinstance(result.ok, bool)
    assert len(result.result) <= 20_000
    assert result.error is None or len(result.error) <= 2_000


# -- I. registry ----------------------------------------------------------


def test_available_environments_is_the_documented_set() -> None:
    assert available_environments() == ["calculator", "jsonnav", "sqlite", "sqlite_recovery"]
    assert ENVIRONMENT_NAMES == ("calculator", "jsonnav", "sqlite", "sqlite_recovery")


def test_unknown_environment_lists_the_available_names() -> None:
    with pytest.raises(ConfigError) as excinfo:
        make_environment("nope")
    error = excinfo.value
    assert "nope" in error.message
    assert error.hint is not None
    for name in ENV_NAMES:
        assert name in error.hint


@pytest.mark.parametrize("name", ENV_NAMES)
def test_make_environment_passes_params_through(name: str) -> None:
    with environment(name, tolerance=0.25) as env:
        assert env.name == name
        assert env.params["tolerance"] == pytest.approx(0.25)
        assert env.describe() == {
            "name": name,
            "params": {
                "tolerance": 0.25,
                "protocol_version": "v1",
                "verifier_version": "v1",
            },
        }


# -- J. prompt style -----------------------------------------------------


@pytest.mark.parametrize("name", ENV_NAMES)
def test_compact_prompt_is_strictly_shorter_than_full(name: str) -> None:
    with environment(name) as full_env, environment(name, prompt_style="compact") as compact_env:
        full = full_env.system_prompt()
        compact = compact_env.system_prompt()
        names = [spec.name for spec in full_env.tool_specs()]
    assert full_env.prompt_style == "full"
    assert compact_env.prompt_style == "compact"
    assert len(compact) < len(full)
    # Both styles must still name every tool the policy is allowed to call.
    for tool_name in names:
        assert tool_name in full
        assert tool_name in compact


@pytest.mark.parametrize("style", ["tiny", "FULL", "", "verbose"])
def test_invalid_prompt_style_is_rejected(style: str) -> None:
    with environment("calculator", prompt_style=style) as env:
        with pytest.raises(ValueError, match="prompt_style must be"):
            env.prompt_style
        with pytest.raises(ValueError, match="prompt_style must be"):
            env.system_prompt()


@pytest.mark.parametrize("name", ENV_NAMES)
def test_tool_specs_are_renderable_and_required_params_are_declared(name: str) -> None:
    with environment(name) as env:
        specs = env.tool_specs()
    assert specs
    for spec in specs:
        assert set(spec.required) <= set(spec.parameters)
        rendered = spec.render()
        assert spec.name in rendered
        assert "example" in rendered
        assert json.loads(rendered.split("example: ", 1)[1])["name"] == spec.name


@pytest.mark.parametrize("name", ENV_NAMES)
def test_user_prompt_is_the_task_prompt(name: str) -> None:
    with environment(name) as env:
        task = env.generate_task(0, 1, difficulty="easy", split="train")
        assert env.user_prompt(task) == task.prompt
