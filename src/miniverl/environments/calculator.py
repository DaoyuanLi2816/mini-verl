"""Calculator environment: arithmetic and unit conversion.

Safety
------
Expressions are evaluated by walking a parsed :mod:`ast`, not by
:func:`eval`.  The walker accepts a closed whitelist of node types -- numeric
constants, ``+ - * / // % **`` and unary sign -- and rejects everything else,
including names, attribute access, calls, subscripts, comprehensions and
f-strings.  Depth, operand magnitude and exponent size are all bounded, so
``9**9**9`` is a clean error rather than a hung process.
"""

from __future__ import annotations

import ast
import math
import random
from typing import Any

from miniverl.environments.base import (
    FailureCategory,
    Observation,
    OracleAction,
    OracleActionKind,
    StepResult,
    Task,
    ToolCall,
    ToolEnvironment,
    ToolSpec,
    VerificationResult,
)
from miniverl.errors import ToolEnvironmentError

__all__ = ["CalculatorEnvironment", "safe_eval", "UNIT_CONVERSIONS", "normalize_number"]

_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

MAX_DEPTH = 12
MAX_ABS_VALUE = 1e12
MAX_EXPONENT = 8
MAX_EXPRESSION_CHARS = 200

#: ``(from_unit, to_unit) -> multiplicative factor``.  Temperatures are affine
#: and are handled separately below.
UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("km", "mi"): 0.621371,
    ("mi", "km"): 1.609344,
    ("m", "ft"): 3.280839895,
    ("ft", "m"): 0.3048,
    ("kg", "lb"): 2.20462262,
    ("lb", "kg"): 0.45359237,
    ("g", "oz"): 0.0352739619,
    ("oz", "g"): 28.349523125,
    ("l", "gal"): 0.264172052,
    ("gal", "l"): 3.785411784,
}

_UNIT_NAMES = {
    "km": "kilometers",
    "mi": "miles",
    "m": "meters",
    "ft": "feet",
    "kg": "kilograms",
    "lb": "pounds",
    "g": "grams",
    "oz": "ounces",
    "l": "liters",
    "gal": "gallons",
    "c": "degrees Celsius",
    "f": "degrees Fahrenheit",
}


def normalize_number(text: str) -> float | None:
    """Parse a numeric answer, tolerating whitespace, commas and a unit suffix."""
    cleaned = text.strip().replace(",", "").replace("_", "")
    if not cleaned:
        return None
    head = cleaned.split()[0]
    try:
        return float(head)
    except ValueError:
        return None


def _check_value(value: float | int) -> float:
    if isinstance(value, complex):  # pragma: no cover - guarded by node whitelist
        raise ToolEnvironmentError("complex results are not supported")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ToolEnvironmentError("expression produced a non-finite value")
    if abs(value) > MAX_ABS_VALUE:
        raise ToolEnvironmentError(
            f"intermediate value {value:.3g} exceeds the magnitude limit {MAX_ABS_VALUE:.0e}"
        )
    return float(value)


def _eval_node(node: ast.AST, depth: int) -> float:
    if depth > MAX_DEPTH:
        raise ToolEnvironmentError(f"expression nests deeper than {MAX_DEPTH} levels")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolEnvironmentError(
                f"only numeric literals are allowed, got {type(node.value).__name__}"
            )
        return _check_value(node.value)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return _check_value(-_eval_node(node.operand, depth + 1))
        if isinstance(node.op, ast.UAdd):
            return _check_value(+_eval_node(node.operand, depth + 1))
        raise ToolEnvironmentError(f"unary operator {type(node.op).__name__} is not allowed")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ToolEnvironmentError(f"operator {type(node.op).__name__} is not allowed")
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ToolEnvironmentError(f"exponent {right} exceeds the limit {MAX_EXPONENT}")
            if abs(left) > 1e3:
                raise ToolEnvironmentError("base of ** is too large")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolEnvironmentError("division by zero")
        try:
            value = op(left, right)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise ToolEnvironmentError(f"arithmetic error: {exc}") from exc
        return _check_value(value)
    raise ToolEnvironmentError(
        f"{type(node).__name__} is not allowed in an expression; only numbers and "
        "the operators + - * / // % ** are supported"
    )


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression without executing Python.

    Raises :class:`~miniverl.errors.ToolEnvironmentError` for anything outside
    the whitelist -- names, calls, imports, attribute access, comprehensions --
    and for depth, magnitude or exponent violations.
    """
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolEnvironmentError(
            f"expression is {len(expression)} characters, over the "
            f"{MAX_EXPRESSION_CHARS} character limit"
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolEnvironmentError(f"could not parse expression: {exc.msg}") from exc
    return _eval_node(tree.body, 0)


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e12:
        return str(round(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


class CalculatorEnvironment(ToolEnvironment):
    """Arithmetic and unit-conversion tasks with an exact verifier."""

    name = "calculator"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._task: Task | None = None
        self._steps = 0
        self.tolerance = float(params.get("tolerance", 1e-4))

    # -- tools ----------------------------------------------------------

    def tool_specs(self) -> list[ToolSpec]:
        """The two calculator tools."""
        return [
            ToolSpec(
                name="calculator",
                description="Evaluate an arithmetic expression (+ - * / // % **).",
                parameters={"expression": "arithmetic expression, digits and operators only"},
                required=("expression",),
                example={"expression": "2*(3+4)"},
            ),
            ToolSpec(
                name="convert",
                description=(
                    "Convert a value between units. Supported: "
                    "km/mi, m/ft, kg/lb, g/oz, l/gal, c/f."
                ),
                parameters={
                    "value": "number to convert",
                    "from_unit": "source unit code",
                    "to_unit": "target unit code",
                },
                required=("value", "from_unit", "to_unit"),
                example={"value": 5, "from_unit": "km", "to_unit": "mi"},
            ),
        ]

    # -- episode --------------------------------------------------------

    def reset(self, task: Task) -> Observation:
        """Begin an episode."""
        self._task = task
        self._steps = 0
        return Observation(text=task.prompt, state_id="calc:0")

    def step(self, call: ToolCall) -> StepResult:
        """Run one tool."""
        self._steps += 1
        state_id = f"calc:{self._steps}"
        if call.name == "calculator":
            expression = call.arguments.get("expression")
            if not isinstance(expression, str):
                return StepResult(
                    ok=False,
                    error="'expression' must be a string",
                    state_id=state_id,
                    failure_category=FailureCategory.INVALID_TOOL_CALL,
                )
            try:
                value = safe_eval(expression)
            except ToolEnvironmentError as exc:
                return StepResult(
                    ok=False,
                    error=exc.message,
                    state_id=state_id,
                    failure_category=FailureCategory.TOOL_ERROR,
                )
            return StepResult(ok=True, result=_format_number(value), state_id=state_id)

        if call.name == "convert":
            raw_value = call.arguments.get("value")
            from_unit = str(call.arguments.get("from_unit", "")).lower()
            to_unit = str(call.arguments.get("to_unit", "")).lower()
            if isinstance(raw_value, str):
                parsed = normalize_number(raw_value)
            elif isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                parsed = float(raw_value)
            else:
                parsed = None
            if parsed is None:
                return StepResult(
                    ok=False,
                    error="'value' must be a number",
                    state_id=state_id,
                    failure_category=FailureCategory.INVALID_TOOL_CALL,
                )
            try:
                converted = self._convert(parsed, from_unit, to_unit)
            except ToolEnvironmentError as exc:
                return StepResult(
                    ok=False,
                    error=exc.message,
                    state_id=state_id,
                    failure_category=FailureCategory.TOOL_ERROR,
                )
            return StepResult(ok=True, result=_format_number(converted), state_id=state_id)

        return StepResult(
            ok=False,
            error=f"unknown tool {call.name!r}; available tools: calculator, convert",
            state_id=state_id,
            failure_category=FailureCategory.UNKNOWN_TOOL,
        )

    @staticmethod
    def _convert(value: float, from_unit: str, to_unit: str) -> float:
        if from_unit == to_unit:
            return value
        if (from_unit, to_unit) == ("c", "f"):
            return value * 9.0 / 5.0 + 32.0
        if (from_unit, to_unit) == ("f", "c"):
            return (value - 32.0) * 5.0 / 9.0
        factor = UNIT_CONVERSIONS.get((from_unit, to_unit))
        if factor is None:
            raise ToolEnvironmentError(
                f"cannot convert {from_unit!r} to {to_unit!r}; supported pairs: "
                + ", ".join(f"{a}->{b}" for a, b in sorted(UNIT_CONVERSIONS))
                + ", c->f, f->c"
            )
        return value * factor

    def verify(self, answer: str) -> VerificationResult:
        """Compare against the exact reference value."""
        if self._task is None:
            raise ToolEnvironmentError("verify() called before reset()")
        expected = self._task.answer
        predicted = answer.strip()
        expected_value = normalize_number(expected)
        predicted_value = normalize_number(predicted)
        if predicted_value is None:
            return VerificationResult(
                solved=False,
                reward=0.0,
                expected=expected,
                predicted=predicted,
                failure_category=FailureCategory.MALFORMED_ANSWER,
                detail="answer is not a number",
            )
        assert expected_value is not None
        scale = max(1.0, abs(expected_value))
        if abs(predicted_value - expected_value) <= self.tolerance * scale:
            return VerificationResult(
                solved=True, reward=1.0, expected=expected, predicted=predicted
            )
        return VerificationResult(
            solved=False,
            reward=0.0,
            expected=expected,
            predicted=predicted,
            failure_category=FailureCategory.WRONG_ANSWER,
            detail=f"expected {expected_value:g}, got {predicted_value:g}",
        )

    # -- tasks ----------------------------------------------------------

    def generate_task(self, index: int, seed: int, *, difficulty: str, split: str) -> Task:
        """Deterministically build one task."""
        rng = random.Random(f"calculator:{seed}:{difficulty}:{index}")
        kind = rng.random()
        if difficulty == "easy":
            return self._arithmetic_task(rng, index, split, difficulty, terms=2)
        if difficulty == "medium":
            if kind < 0.5:
                return self._arithmetic_task(rng, index, split, difficulty, terms=3)
            return self._conversion_task(rng, index, split, difficulty)
        if kind < 0.5:
            return self._arithmetic_task(rng, index, split, difficulty, terms=4)
        return self._chained_task(rng, index, split, difficulty)

    def _arithmetic_task(
        self, rng: random.Random, index: int, split: str, difficulty: str, *, terms: int
    ) -> Task:
        expression = self._random_expression(rng, terms)
        value = safe_eval(expression)
        return Task(
            task_id=f"calc-{split}-{index}",
            prompt=f"Compute {expression} and report the value.",
            answer=_format_number(value),
            difficulty=difficulty,
            split=split,
            metadata={"kind": "arithmetic", "expression": expression},
        )

    def _conversion_task(self, rng: random.Random, index: int, split: str, difficulty: str) -> Task:
        pairs = [*sorted(UNIT_CONVERSIONS), ("c", "f"), ("f", "c")]
        from_unit, to_unit = pairs[rng.randrange(len(pairs))]
        value = rng.randrange(2, 500)
        result = self._convert(float(value), from_unit, to_unit)
        return Task(
            task_id=f"calc-{split}-{index}",
            prompt=(
                f"Convert {value} {_UNIT_NAMES[from_unit]} to {_UNIT_NAMES[to_unit]} "
                "and report the value."
            ),
            answer=_format_number(result),
            difficulty=difficulty,
            split=split,
            metadata={
                "kind": "conversion",
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
            },
        )

    def _chained_task(self, rng: random.Random, index: int, split: str, difficulty: str) -> Task:
        expression = self._random_expression(rng, 2)
        intermediate = safe_eval(expression)
        from_unit, to_unit = ("km", "mi") if rng.random() < 0.5 else ("kg", "lb")
        result = self._convert(intermediate, from_unit, to_unit)
        return Task(
            task_id=f"calc-{split}-{index}",
            prompt=(
                f"Compute {expression}, then convert that many "
                f"{_UNIT_NAMES[from_unit]} to {_UNIT_NAMES[to_unit]} and report the value."
            ),
            answer=_format_number(result),
            difficulty=difficulty,
            split=split,
            metadata={
                "kind": "chained",
                "expression": expression,
                "intermediate": _format_number(intermediate),
                "from_unit": from_unit,
                "to_unit": to_unit,
            },
        )

    @staticmethod
    def _random_expression(rng: random.Random, terms: int) -> str:
        operators = ["+", "-", "*"]
        expression = str(rng.randrange(1, 20))
        for i in range(terms - 1):
            op = operators[rng.randrange(len(operators))]
            operand = rng.randrange(1, 13)
            if i == 0 and rng.random() < 0.5:
                expression = f"({expression} {op} {operand})"
            else:
                expression = f"{expression} {op} {operand}"
        return expression

    # -- oracle ----------------------------------------------------------

    def oracle_actions(self, task: Task) -> list[OracleAction]:
        """Reference tool sequence for ``task``."""
        kind = task.metadata.get("kind")
        if kind == "arithmetic":
            return [
                OracleAction(
                    kind=OracleActionKind.TOOL_CALL,
                    tool_name="calculator",
                    arguments={"expression": str(task.metadata["expression"])},
                ),
                OracleAction(kind=OracleActionKind.FINAL, answer=task.answer),
            ]
        if kind == "conversion":
            return [
                OracleAction(
                    kind=OracleActionKind.TOOL_CALL,
                    tool_name="convert",
                    arguments={
                        "value": task.metadata["value"],
                        "from_unit": task.metadata["from_unit"],
                        "to_unit": task.metadata["to_unit"],
                    },
                ),
                OracleAction(kind=OracleActionKind.FINAL, answer=task.answer),
            ]
        if kind == "chained":
            return [
                OracleAction(
                    kind=OracleActionKind.TOOL_CALL,
                    tool_name="calculator",
                    arguments={"expression": str(task.metadata["expression"])},
                ),
                OracleAction(
                    kind=OracleActionKind.TOOL_CALL,
                    tool_name="convert",
                    arguments={
                        "value": normalize_number(str(task.metadata["intermediate"])),
                        "from_unit": task.metadata["from_unit"],
                        "to_unit": task.metadata["to_unit"],
                    },
                ),
                OracleAction(kind=OracleActionKind.FINAL, answer=task.answer),
            ]
        raise ToolEnvironmentError(f"task {task.task_id} has unknown kind {kind!r}")

    def privileged_context(self, task: Task) -> str | None:
        """Oracle hint shown to the teacher only."""
        kind = task.metadata.get("kind")
        if kind == "chained":
            return (
                f"Verified reference: {task.metadata['expression']} = "
                f"{task.metadata['intermediate']}, and the final answer is {task.answer}."
            )
        return f"Verified reference answer: {task.answer}."
