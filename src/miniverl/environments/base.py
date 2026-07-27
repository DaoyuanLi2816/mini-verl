"""Tool-environment contract.

An environment owns four things and nothing else:

1. **Task generation** -- deterministic from a seed, with disjoint splits.
2. **Tools** -- a typed spec plus a safe executor.
3. **A verifier** -- exact, not model-graded.
4. **An oracle** -- the reference action sequence, used for SFT cold starts and
   for the ``privileged_context`` teacher mode.

Everything is local, seeded and network-free.  There is no LLM-as-judge
anywhere in the loop: a task is solved or it is not, and the failure is placed
in a named category.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ToolSpec",
    "Task",
    "Observation",
    "ToolCall",
    "StepResult",
    "VerificationResult",
    "OracleActionKind",
    "OracleAction",
    "ToolEnvironment",
    "FailureCategory",
    "make_splits",
]


class FailureCategory(str, Enum):
    """Why a rollout did not solve its task."""

    SOLVED = "solved"
    WRONG_ANSWER = "wrong_answer"
    NO_FINAL_ANSWER = "no_final_answer"
    MALFORMED_ANSWER = "malformed_answer"
    INVALID_TOOL_CALL = "invalid_tool_call"
    UNKNOWN_TOOL = "unknown_tool"
    TOOL_ERROR = "tool_error"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class ToolSpec:
    """Declarative description of one tool, rendered into the system prompt."""

    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    example: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """One-tool block for the system prompt."""
        import json

        params = ", ".join(
            f"{name} ({'required' if name in self.required else 'optional'}): {desc}"
            for name, desc in self.parameters.items()
        )
        example = json.dumps(
            {"name": self.name, "arguments": self.example}, ensure_ascii=False, sort_keys=True
        )
        return f"- {self.name}: {self.description}\n  parameters: {params}\n  example: {example}"


@dataclass(frozen=True)
class Task:
    """A single problem instance."""

    task_id: str
    prompt: str
    answer: str
    difficulty: str = "easy"
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """The environment's view after :meth:`ToolEnvironment.reset`."""

    text: str
    state_id: str


@dataclass(frozen=True)
class ToolCall:
    """A parsed request to run one tool."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class StepResult:
    """Outcome of one tool invocation."""

    ok: bool
    result: str = ""
    error: str | None = None
    state_id: str = "s0"
    failure_category: FailureCategory | None = None


@dataclass(frozen=True)
class VerificationResult:
    """Exact grading of a final answer."""

    solved: bool
    reward: float
    expected: str
    predicted: str
    failure_category: FailureCategory = FailureCategory.SOLVED
    detail: str | None = None


class OracleActionKind(str, Enum):
    """Kinds of step in an oracle trace."""

    TOOL_CALL = "tool_call"
    FINAL = "final"


@dataclass(frozen=True)
class OracleAction:
    """One reference action.  Rendered into text by the transcript builder."""

    kind: OracleActionKind
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None


class ToolEnvironment(ABC):
    """Base class for miniVERL's deterministic local environments."""

    #: Registry name, matching ``environment.name`` in a recipe.
    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    # -- task generation ------------------------------------------------

    @abstractmethod
    def generate_task(self, index: int, seed: int, *, difficulty: str, split: str) -> Task:
        """Build the ``index``-th task for ``(seed, difficulty, split)``."""
        ...

    # -- tools ----------------------------------------------------------

    @abstractmethod
    def tool_specs(self) -> list[ToolSpec]:
        """Tools available to the policy."""
        ...

    @abstractmethod
    def reset(self, task: Task) -> Observation:
        """Start an episode for ``task`` and return the initial observation."""
        ...

    @abstractmethod
    def step(self, call: ToolCall) -> StepResult:
        """Execute one tool call against the current episode state."""
        ...

    @abstractmethod
    def verify(self, answer: str) -> VerificationResult:
        """Grade a final answer for the current episode."""
        ...

    @abstractmethod
    def oracle_actions(self, task: Task) -> list[OracleAction]:
        """Reference solution as a sequence of actions."""
        ...

    def privileged_context(self, task: Task) -> str | None:
        """Extra information given only to the teacher, never to the student.

        Returning ``None`` means the environment has no privileged view, and
        ``teacher.mode: privileged_context`` will be rejected for it.
        """
        return None

    # -- prompt rendering ------------------------------------------------

    @property
    def prompt_style(self) -> str:
        """``full`` (default) or ``compact``.

        ``compact`` drops per-tool descriptions and examples.  It exists because
        the toy tokenizer is nearly character-level, so a full prompt would
        dominate every toy sequence and the CPU demo would spend its time on
        boilerplate instead of on the parts under test.
        """
        style = str(self.params.get("prompt_style", "full"))
        if style not in ("full", "compact"):
            raise ValueError(f"prompt_style must be 'full' or 'compact', got {style!r}")
        return style

    def system_prompt(self) -> str:
        """System message describing the protocol and the available tools."""
        if self.prompt_style == "compact":
            tools = "\n".join(
                f"- {spec.name}({', '.join(spec.parameters)})" for spec in self.tool_specs()
            )
            return (
                "Use tools to solve the task.\n"
                f"{tools}\n"
                'Reply with one block:\n<tool_call>\n{"name": "t", "arguments": {}}\n'
                "</tool_call>\nor\n<final>\nanswer\n</final>"
            )
        tools = "\n".join(spec.render() for spec in self.tool_specs())
        return (
            "You are a tool-using assistant. Solve the task with the tools below.\n"
            "\n"
            "Tools:\n"
            f"{tools}\n"
            "\n"
            "Reply with exactly one block per turn.\n"
            "To call a tool:\n"
            '<tool_call>\n{"name": "<tool>", "arguments": {...}}\n</tool_call>\n'
            "To answer:\n"
            "<final>\n<answer>\n</final>"
        )

    def user_prompt(self, task: Task) -> str:
        """User message for ``task``."""
        return task.prompt

    # -- diagnostics -----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Environment identity for the run manifest."""
        return {"name": self.name, "params": dict(self.params)}


#: Index offsets keeping the three splits disjoint by construction.
_SPLIT_OFFSET = {"train": 0, "eval": 1_000_000, "test": 2_000_000}


def make_splits(
    env: ToolEnvironment,
    *,
    counts: dict[str, int],
    seed: int,
    difficulty: str = "easy",
) -> dict[str, list[Task]]:
    """Generate deterministic, prompt-disjoint train/eval/test splits.

    Splits are built in a fixed order and any prompt already produced by an
    earlier split is skipped, so a task can never leak from train into eval.
    """
    seen: set[str] = set()
    out: dict[str, list[Task]] = {}
    for split in ("train", "eval", "test"):
        wanted = int(counts.get(split, 0))
        tasks: list[Task] = []
        index = _SPLIT_OFFSET[split]
        guard = 0
        while len(tasks) < wanted:
            guard += 1
            if guard > max(wanted * 200, 2000):
                raise RuntimeError(
                    f"environment {env.name!r} could not generate {wanted} distinct "
                    f"{split} tasks at difficulty {difficulty!r}; the task space is too small"
                )
            task = env.generate_task(index, seed, difficulty=difficulty, split=split)
            index += 1
            if task.prompt in seen:
                continue
            seen.add(task.prompt)
            tasks.append(task)
        out[split] = tasks
    return out


def unique_prompts(tasks: Sequence[Task]) -> int:
    """Number of distinct prompts in ``tasks`` (used by split assertions)."""
    return len({t.prompt for t in tasks})
