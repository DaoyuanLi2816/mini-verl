"""A complete custom miniVERL environment, start to finish.

The task: reverse a string, then report its length. Two dependent tool calls, so
it exercises the multi-turn path rather than degenerating into a single answer.

Run it:

    python examples/custom_environment/reverse_environment.py

It registers the environment, trains a toy student on it with genuine on-policy
distillation, and prints the run directory. Nothing is downloaded.

What a new environment must provide, all of which is checked by the base class
and by the trainer:

* deterministic task generation from a seed;
* typed tool specifications;
* a safe executor with no network, no shell and no arbitrary file access;
* an exact verifier -- there is no LLM judge anywhere in miniVERL;
* a deterministic oracle action sequence that solves every task;
* optionally, a privileged context string shown only to the teacher.
"""

from __future__ import annotations

import random
import string
import sys
from pathlib import Path
from typing import Any

from miniverl.config import RunConfig
from miniverl.environments import (
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
from miniverl.environments.registry import register
from miniverl.errors import ToolEnvironmentError

MAX_TEXT_CHARS = 64
ALPHABET = string.ascii_lowercase


@register
class ReverseEnvironment(ToolEnvironment):
    """Reverse a string with one tool, then measure it with another."""

    name = "reverse"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._task: Task | None = None
        self._steps = 0

    # -- tools ------------------------------------------------------------

    def tool_specs(self) -> list[ToolSpec]:
        """Two tools, so the hard difficulty needs both in sequence."""
        return [
            ToolSpec(
                name="reverse",
                description="Reverse a string of lowercase ASCII letters.",
                parameters={"text": "the string to reverse"},
                required=("text",),
                example={"text": "abc"},
            ),
            ToolSpec(
                name="length",
                description="Return the number of characters in a string.",
                parameters={"text": "the string to measure"},
                required=("text",),
                example={"text": "abc"},
            ),
        ]

    # -- episode ----------------------------------------------------------

    def reset(self, task: Task) -> Observation:
        """Begin an episode. There is no mutable world state to build here."""
        self._task = task
        self._steps = 0
        return Observation(text=task.prompt, state_id="rev:0")

    def step(self, call: ToolCall) -> StepResult:
        """Execute one tool call, defensively."""
        self._steps += 1
        state_id = f"rev:{self._steps}"
        text = call.arguments.get("text")
        if not isinstance(text, str):
            return StepResult(
                ok=False,
                error="'text' must be a string",
                state_id=state_id,
                failure_category=FailureCategory.INVALID_TOOL_CALL,
            )
        if len(text) > MAX_TEXT_CHARS:
            return StepResult(
                ok=False,
                error=f"'text' is {len(text)} characters, over the {MAX_TEXT_CHARS} limit",
                state_id=state_id,
                failure_category=FailureCategory.TOOL_ERROR,
            )
        if call.name == "reverse":
            return StepResult(ok=True, result=text[::-1], state_id=state_id)
        if call.name == "length":
            return StepResult(ok=True, result=str(len(text)), state_id=state_id)
        return StepResult(
            ok=False,
            error=f"unknown tool {call.name!r}; available tools: reverse, length",
            state_id=state_id,
            failure_category=FailureCategory.UNKNOWN_TOOL,
        )

    def verify(self, answer: str) -> VerificationResult:
        """Exact comparison after trimming surrounding whitespace and quotes."""
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

    # -- tasks -------------------------------------------------------------

    def generate_task(self, index: int, seed: int, *, difficulty: str, split: str) -> Task:
        """Deterministic given (index, seed, difficulty, split)."""
        rng = random.Random(f"reverse:{seed}:{difficulty}:{index}")
        size = {"easy": 3, "medium": 5, "hard": 7}.get(difficulty, 3)
        text = "".join(rng.choice(ALPHABET) for _ in range(size))
        if difficulty == "hard":
            prompt = f"Reverse the string '{text}', then report the length of the result."
            answer = str(len(text))
            kind = "reverse_then_length"
        else:
            prompt = f"Reverse the string '{text}' and report the result."
            answer = text[::-1]
            kind = "reverse"
        return Task(
            task_id=f"rev-{split}-{index}",
            prompt=prompt,
            answer=answer,
            difficulty=difficulty,
            split=split,
            metadata={"kind": kind, "text": text},
        )

    def oracle_actions(self, task: Task) -> list[OracleAction]:
        """The reference solution, used for the SFT cold start."""
        text = str(task.metadata["text"])
        actions = [
            OracleAction(OracleActionKind.TOOL_CALL, tool_name="reverse", arguments={"text": text})
        ]
        if task.metadata.get("kind") == "reverse_then_length":
            actions.append(
                OracleAction(
                    OracleActionKind.TOOL_CALL,
                    tool_name="length",
                    arguments={"text": text[::-1]},
                )
            )
        actions.append(OracleAction(OracleActionKind.FINAL, answer=task.answer))
        return actions

    def privileged_context(self, task: Task) -> str | None:
        """Shown to the teacher only, never to the student."""
        return f"Verified reference answer: {task.answer}."


def build_config(output_dir: str) -> RunConfig:
    """A toy on-policy distillation recipe over the new environment."""
    return RunConfig.model_validate(
        {
            "schema_version": 1,
            "run": {
                "name": "reverse-opd",
                "mode": "opd",
                "seed": 20260727,
                "output_dir": output_dir,
            },
            "models": {
                "backend": "toy",
                "device": "cpu",
                "student": {
                    "model_id": "toy-student",
                    "lora": {"enabled": False},
                    "toy": {
                        "hidden_size": 96,
                        "num_layers": 3,
                        "num_heads": 4,
                        "intermediate_size": 192,
                        "max_position_embeddings": 768,
                    },
                },
                "teacher": {
                    "model_id": "toy-teacher",
                    "toy_pretrain_steps": 200,
                    "toy": {
                        "hidden_size": 128,
                        "num_layers": 3,
                        "num_heads": 4,
                        "intermediate_size": 256,
                        "max_position_embeddings": 768,
                    },
                },
            },
            "environment": {
                "name": "reverse",  # <- the registered name
                "difficulty": "easy",
                "params": {"prompt_style": "compact"},
                "train_tasks": 96,
                "eval_tasks": 12,
                "test_tasks": 12,
                "split_seed": 5,
            },
            "rollout": {"max_turns": 3, "max_new_tokens_per_turn": 32, "max_total_tokens": 448},
            "selection": {"selector": "hybrid", "ratio": 0.7},
            "loss": {
                "mode": "bucketed_topk_tail",
                "divergence": "reverse_kl",
                "top_k": 16,
                "chunk_size": 64,
            },
            "train": {
                "cycles": 6,
                "rollouts_per_cycle": 6,
                "gradient_accumulation_steps": 6,
                "learning_rate": 0.003,
                "sft_warmup_cycles": 60,
            },
            "memory": {"strategy": "resident"},
            "cache": {"entries_per_shard": 6},
            "eval": {"enabled": True, "tasks": 12, "temperature": 0.0},
            "report": {"enabled": True, "max_trajectories": 2},
        }
    )


def main() -> int:
    """Register, sanity-check the oracle, then train."""
    from miniverl.environments import make_splits
    from miniverl.environments.registry import available_environments, make_environment
    from miniverl.trainer import OPDTrainer

    print("registered environments:", available_environments())
    assert "reverse" in available_environments()

    # A new environment is only useful if its oracle actually solves its tasks.
    env = make_environment("reverse", prompt_style="compact")
    splits = make_splits(env, counts={"train": 8, "eval": 4, "test": 4}, seed=5, difficulty="hard")
    solved = 0
    for task in splits["train"]:
        env.reset(task)
        for action in env.oracle_actions(task):
            if action.kind is OracleActionKind.TOOL_CALL:
                result = env.step(ToolCall(action.tool_name or "", action.arguments))
                assert result.ok, result.error
            else:
                solved += int(env.verify(action.answer or "").solved)
    print(f"oracle solves {solved}/{len(splits['train'])} hard tasks")
    assert solved == len(splits["train"])

    output = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/examples")
    trainer = OPDTrainer.from_config(build_config(str(output)), run_id="reverse-opd")
    try:
        result = trainer.train()
    finally:
        trainer.close()

    baseline = (result.baseline_eval or {}).get("success_rate")
    final = (result.eval or {}).get("success_rate")
    print(f"\nrun directory : {result.run_dir}")
    print(f"optimizer steps: {result.global_step}")
    print(f"task success   : {baseline} -> {final} (greedy, held-out eval split)")
    print(f"\nnext: miniverl report {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
