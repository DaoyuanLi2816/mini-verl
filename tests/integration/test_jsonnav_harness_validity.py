"""A zero has to be the model's, not the harness's.

Every starting-checkpoint candidate scored 0/64 on JSONNav with zero tool calls
emitted and exactly 128 tokens per rollout -- a uniform, deterministic failure
across four different adapters including the base model. That pattern is as
consistent with a misconfigured harness as with an incapable policy, and
publishing `checkpoint_selection_failed` on a broken measurement would be worse
than publishing nothing.

The environment ships an oracle. If the oracle clears the pinned settings, the
settings are sound and the zero belongs to the models. These run the oracle
through the same `RolloutRunner` path, the same `JsonNavSettings`, and the same
per-turn token budget the candidates got.
"""

from __future__ import annotations

from typing import Any

import pytest

from miniverl.agent.loop import RolloutRunner
from miniverl.alignment_external.jsonnav_utility import JsonNavSettings
from miniverl.config.models import RolloutConfig
from miniverl.environments.jsonnav import JsonNavEnvironment

# `network` as well as `torch`: the oracle replays scripted actions rather than
# sampling, but building the transcript still needs the real pinned tokenizer,
# so this fetches Qwen3-0.6B and cannot run where outgoing traffic is disabled.
pytestmark = [pytest.mark.torch, pytest.mark.network, pytest.mark.slow]

STUDENT = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TASKS = 8


class _TokenizerOnlyBackend:
    """`oracle_rollout` replays scripted actions; it only needs to tokenize."""

    def __init__(self) -> None:
        from transformers import AutoTokenizer

        from miniverl.models.tokenizers import HFTokenizerAdapter

        raw = AutoTokenizer.from_pretrained(STUDENT, revision=REVISION)
        if raw.pad_token_id is None:
            raw.pad_token = raw.eos_token
        self.tokenizer = HFTokenizerAdapter(raw, STUDENT, REVISION)
        self.model_id = STUDENT
        self.model_revision = REVISION
        self.capabilities = type("Caps", (), {"name": STUDENT, "device": "cpu"})()


def _runner(settings: JsonNavSettings) -> tuple[Any, JsonNavEnvironment]:
    environment = JsonNavEnvironment(
        protocol_version=settings.protocol_version, prompt_style=settings.prompt_style
    )
    runner = RolloutRunner(
        backend=_TokenizerOnlyBackend(),
        environment=environment,
        config=RolloutConfig(
            max_turns=settings.max_turns,
            max_new_tokens_per_turn=settings.max_new_tokens_per_turn,
            max_total_tokens=settings.max_total_tokens,
            temperature=0.0,
        ),
    )
    return runner, environment


def _oracle_rate(settings: JsonNavSettings, tasks: int = TASKS) -> tuple[float, set[str]]:
    runner, environment = _runner(settings)
    solved = 0
    terminations: set[str] = set()
    for index in range(tasks):
        task = environment.generate_task(
            index, settings.task_seed, difficulty=settings.difficulty, split=settings.split
        )
        trajectory = runner.oracle_rollout(task)
        terminations.add(str(trajectory.termination_reason))
        if trajectory.verification is not None and trajectory.verification.solved:
            solved += 1
    return solved / tasks, terminations


def test_the_oracle_clears_the_pinned_settings() -> None:
    """The exact settings the candidates were scored under."""
    rate, terminations = _oracle_rate(JsonNavSettings())

    assert rate == 1.0, (
        f"the oracle scored {rate} under the pinned settings, so a candidate's zero "
        "cannot be attributed to the policy"
    )
    assert terminations == {"TerminationReason.FINAL_ANSWER"}


def test_the_per_turn_token_budget_is_not_the_constraint() -> None:
    """64 tokens per turn is enough for a complete JSONNav tool call.

    Candidates hit PARSE_ERROR_LIMIT at exactly 128 tokens, which would also be
    the signature of a budget too small to finish a call. It is not.
    """
    tight, _ = _oracle_rate(JsonNavSettings(max_new_tokens_per_turn=64))
    generous, _ = _oracle_rate(JsonNavSettings(max_new_tokens_per_turn=256))

    assert tight == generous == 1.0


def test_the_pinned_prompt_style_is_supported() -> None:
    """An unsupported style raises rather than silently degrading.

    The check is lazy -- construction succeeds and the error surfaces when the
    style is actually used -- so a typo would otherwise survive until it had
    quietly changed what every candidate was prompted with.
    """
    assert _oracle_rate(JsonNavSettings(prompt_style="compact"))[0] == 1.0

    with pytest.raises(ValueError, match="prompt_style must be"):
        _oracle_rate(JsonNavSettings(prompt_style="minimal"), tasks=1)


def test_the_hard_difficulty_is_solvable() -> None:
    """`hard` bounds the tasks, it does not make them impossible."""
    assert _oracle_rate(JsonNavSettings(difficulty="hard"))[0] == 1.0
    assert _oracle_rate(JsonNavSettings(difficulty="easy"))[0] == 1.0
