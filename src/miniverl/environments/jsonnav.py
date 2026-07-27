"""JSON navigation environment.

A seeded synthetic document is explored with three read-only tools.  Tasks at
``medium`` and ``hard`` difficulty are *genuinely* multi-call: the second call's
arguments depend on the first call's result, so a single-turn policy cannot
solve them by luck.
"""

from __future__ import annotations

import json
import random
import re
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

__all__ = ["JsonNavEnvironment", "parse_path", "resolve_path", "build_document"]

MAX_RESULTS = 10
MAX_PATH_CHARS = 120
_SEGMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[\d+\])*)$")

_SECTIONS = ("alpha", "beta", "gamma", "delta", "epsilon")
_FIELDS = ("label", "threshold", "owner", "region", "capacity", "tier")
_REGIONS = ("north", "south", "east", "west", "central")
_TIERS = ("bronze", "silver", "gold")


def parse_path(path: str) -> list[str | int]:
    """Parse ``a.b[2].c`` into ``['a', 'b', 2, 'c']``.

    An empty path denotes the document root.
    """
    if len(path) > MAX_PATH_CHARS:
        raise ToolEnvironmentError(
            f"path is {len(path)} characters, over the {MAX_PATH_CHARS} character limit"
        )
    cleaned = path.strip()
    if cleaned in ("", "$", "."):
        return []
    parts: list[str | int] = []
    for raw in cleaned.split("."):
        match = _SEGMENT_RE.match(raw)
        if not match:
            raise ToolEnvironmentError(
                f"invalid path segment {raw!r}; use names like 'config' and indices like 'items[2]'"
            )
        parts.append(match.group(1))
        for index in re.findall(r"\[(\d+)\]", match.group(2)):
            parts.append(int(index))
    return parts


def resolve_path(document: Any, path: str) -> Any:
    """Return the value at ``path`` or raise a helpful error."""
    node: Any = document
    walked: list[str] = []
    for part in parse_path(path):
        if isinstance(part, int):
            if not isinstance(node, list):
                raise ToolEnvironmentError(
                    f"'{'.'.join(walked) or '<root>'}' is not a list, cannot index [{part}]"
                )
            if not 0 <= part < len(node):
                raise ToolEnvironmentError(
                    f"index {part} is out of range at '{'.'.join(walked) or '<root>'}' "
                    f"(length {len(node)})"
                )
            node = node[part]
            walked.append(f"[{part}]")
        else:
            if not isinstance(node, dict):
                raise ToolEnvironmentError(
                    f"'{'.'.join(walked) or '<root>'}' is not an object, cannot read {part!r}"
                )
            if part not in node:
                available = ", ".join(sorted(node)[:8])
                raise ToolEnvironmentError(
                    f"key {part!r} not found at '{'.'.join(walked) or '<root>'}'; "
                    f"available keys: {available}"
                )
            node = node[part]
            walked.append(part)
    return node


def _walk(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            out.append((path, value))
            out.extend(_walk(value, path))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            path = f"{prefix}[{i}]"
            out.append((path, value))
            out.extend(_walk(value, path))
    return out


def build_document(seed: int) -> dict[str, Any]:
    """Deterministically build a small nested document."""
    rng = random.Random(f"jsonnav:{seed}")
    config: dict[str, Any] = {}
    sections = list(_SECTIONS)
    rng.shuffle(sections)
    chosen = sections[: rng.randrange(3, 5)]
    for name in chosen:
        entry: dict[str, Any] = {
            "label": f"{name}-{rng.randrange(100, 999)}",
            "region": _REGIONS[rng.randrange(len(_REGIONS))],
            "tier": _TIERS[rng.randrange(len(_TIERS))],
        }
        if rng.random() < 0.6:
            entry["capacity"] = rng.randrange(10, 500)
        config[name] = entry
    # Exactly one section carries the marker key, which is what `find` locates.
    marker_section = chosen[rng.randrange(len(chosen))]
    config[marker_section]["threshold"] = rng.randrange(1, 100)

    records = [
        {
            "id": f"r{rng.randrange(10, 99)}",
            "owner": f"user{rng.randrange(1, 40)}",
            "score": rng.randrange(0, 100),
        }
        for _ in range(rng.randrange(2, 5))
    ]
    return {
        "config": config,
        "records": records,
        "meta": {
            "version": rng.randrange(1, 9),
            "pointer": marker_section,
            "checked": bool(rng.random() < 0.5),
        },
    }


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class JsonNavEnvironment(ToolEnvironment):
    """Read-only navigation of a seeded synthetic JSON document."""

    name = "jsonnav"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._task: Task | None = None
        self._document: dict[str, Any] = {}
        self._steps = 0

    def tool_specs(self) -> list[ToolSpec]:
        """The three navigation tools."""
        return [
            ToolSpec(
                name="get",
                description="Read the value at a dotted path, for example config.alpha.label.",
                parameters={"path": "dotted path; empty string means the document root"},
                required=("path",),
                example={"path": "meta.pointer"},
            ),
            ToolSpec(
                name="keys",
                description="List the keys of the object (or indices of the list) at a path.",
                parameters={"path": "dotted path; empty string means the document root"},
                required=("path",),
                example={"path": "config"},
            ),
            ToolSpec(
                name="find",
                description=(
                    f"List up to {MAX_RESULTS} paths whose last segment equals the given key."
                ),
                parameters={"key": "exact key name to search for"},
                required=("key",),
                example={"key": "threshold"},
            ),
        ]

    def reset(self, task: Task) -> Observation:
        """Load the document described by the task."""
        self._task = task
        self._steps = 0
        self._document = build_document(int(task.metadata["document_seed"]))
        return Observation(text=task.prompt, state_id="json:0")

    def step(self, call: ToolCall) -> StepResult:
        """Run one read-only navigation tool."""
        self._steps += 1
        state_id = f"json:{self._steps}"
        try:
            if call.name == "get":
                path = call.arguments.get("path", "")
                if not isinstance(path, str):
                    raise ToolEnvironmentError("'path' must be a string")
                return StepResult(
                    ok=True, result=_render(resolve_path(self._document, path)), state_id=state_id
                )
            if call.name == "keys":
                path = call.arguments.get("path", "")
                if not isinstance(path, str):
                    raise ToolEnvironmentError("'path' must be a string")
                node = resolve_path(self._document, path)
                if isinstance(node, dict):
                    return StepResult(
                        ok=True, result=_render(sorted(node)), state_id=state_id
                    )
                if isinstance(node, list):
                    return StepResult(
                        ok=True,
                        result=_render([f"[{i}]" for i in range(len(node))]),
                        state_id=state_id,
                    )
                raise ToolEnvironmentError(
                    f"the value at {path!r} is a scalar and has no keys; use get instead"
                )
            if call.name == "find":
                key = call.arguments.get("key")
                if not isinstance(key, str) or not key:
                    raise ToolEnvironmentError("'key' must be a non-empty string")
                matches = [
                    path
                    for path, _ in _walk(self._document)
                    if path.rsplit(".", 1)[-1] == key
                ]
                return StepResult(
                    ok=True, result=_render(matches[:MAX_RESULTS]), state_id=state_id
                )
        except ToolEnvironmentError as exc:
            return StepResult(
                ok=False,
                error=exc.message,
                state_id=state_id,
                failure_category=FailureCategory.TOOL_ERROR,
            )
        return StepResult(
            ok=False,
            error=f"unknown tool {call.name!r}; available tools: get, keys, find",
            state_id=state_id,
            failure_category=FailureCategory.UNKNOWN_TOOL,
        )

    def verify(self, answer: str) -> VerificationResult:
        """Exact (whitespace- and quote-normalized) comparison."""
        if self._task is None:
            raise ToolEnvironmentError("verify() called before reset()")
        expected = self._task.answer
        predicted = answer.strip().strip('"').strip()
        if predicted == expected:
            return VerificationResult(
                solved=True, reward=1.0, expected=expected, predicted=predicted
            )
        return VerificationResult(
            solved=False,
            reward=0.0,
            expected=expected,
            predicted=predicted,
            failure_category=FailureCategory.WRONG_ANSWER,
            detail=f"expected {expected!r}, got {predicted!r}",
        )

    # -- tasks ----------------------------------------------------------

    def generate_task(self, index: int, seed: int, *, difficulty: str, split: str) -> Task:
        """Deterministically build one navigation task."""
        document_seed = seed * 7919 + index
        document = build_document(document_seed)
        rng = random.Random(f"jsonnav-task:{seed}:{difficulty}:{index}")
        pointer = str(document["meta"]["pointer"])

        # The document id is part of the prompt so a prompt identifies exactly
        # one instance; that is what keeps the train/eval/test splits provably
        # disjoint rather than only probably disjoint.
        header = f"Document #{document_seed}."

        if difficulty == "easy":
            section = sorted(document["config"])[rng.randrange(len(document["config"]))]
            field = "label"
            path = f"config.{section}.{field}"
            answer = _render(resolve_path(document, path))
            prompt = f"{header} Report the value at path {path} in the document."
            kind = "direct"
        elif difficulty == "medium":
            path = f"config.{pointer}.threshold"
            answer = _render(resolve_path(document, path))
            prompt = (
                f"{header} Exactly one path in the document ends with the key "
                "'threshold'. Report its value."
            )
            kind = "find"
        else:
            field = "label" if rng.random() < 0.5 else "region"
            path = f"config.{pointer}.{field}"
            answer = _render(resolve_path(document, path))
            prompt = (
                f"{header} The value at meta.pointer names one of the sub-objects of "
                f"config. Report that sub-object's '{field}' value."
            )
            kind = "pointer"

        return Task(
            task_id=f"json-{split}-{index}",
            prompt=prompt,
            answer=answer,
            difficulty=difficulty,
            split=split,
            metadata={
                "kind": kind,
                "document_seed": document_seed,
                "path": path,
                "pointer": pointer,
            },
        )

    def oracle_actions(self, task: Task) -> list[OracleAction]:
        """Reference navigation sequence."""
        kind = task.metadata.get("kind")
        path = str(task.metadata["path"])
        if kind == "direct":
            calls = [OracleAction(OracleActionKind.TOOL_CALL, tool_name="get", arguments={"path": path})]
        elif kind == "find":
            calls = [
                OracleAction(
                    OracleActionKind.TOOL_CALL, tool_name="find", arguments={"key": "threshold"}
                ),
                OracleAction(OracleActionKind.TOOL_CALL, tool_name="get", arguments={"path": path}),
            ]
        elif kind == "pointer":
            calls = [
                OracleAction(
                    OracleActionKind.TOOL_CALL, tool_name="get", arguments={"path": "meta.pointer"}
                ),
                OracleAction(OracleActionKind.TOOL_CALL, tool_name="get", arguments={"path": path}),
            ]
        else:
            raise ToolEnvironmentError(f"task {task.task_id} has unknown kind {kind!r}")
        return [*calls, OracleAction(OracleActionKind.FINAL, answer=task.answer)]

    def privileged_context(self, task: Task) -> str | None:
        """Reveal the resolved path and value to the teacher only."""
        return (
            f"Verified reference: the answer is at path {task.metadata['path']} "
            f"and its value is {task.answer}."
        )
