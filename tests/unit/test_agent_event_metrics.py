"""Agent metrics count protocol, execution, and answer events separately."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from miniverl.agent.loop import RolloutRunner, RolloutStats
from miniverl.agent.protocol import render_final, render_tool_call
from miniverl.config.models import RolloutConfig
from miniverl.environments.base import Observation, StepResult, Task, ToolCall
from miniverl.environments.calculator import CalculatorEnvironment
from miniverl.errors import ToolEnvironmentError
from miniverl.models.base import GenerationOutput
from miniverl.models.tokenizers import ToyTokenizer
from miniverl.schemas.trajectory import SpanType, TerminationReason


class _ScriptedBackend:
    def __init__(self, outputs: Sequence[tuple[str, str]]) -> None:
        self._outputs = list(outputs)
        self._output_index = 0
        self.tokenizer = ToyTokenizer()
        self.model_id = "scripted"
        self.model_revision = None
        self.capabilities = SimpleNamespace(name=self.model_id)

    def generate(self, prefix_token_ids, **kwargs) -> GenerationOutput:
        del prefix_token_ids, kwargs
        text, stop_reason = self._outputs[self._output_index]
        self._output_index += 1
        return GenerationOutput(
            token_ids=self.tokenizer.encode(text),
            text=text,
            stop_reason=stop_reason,
            matched_stop=None,
        )


def _run(
    outputs: Sequence[tuple[str, str]],
    *,
    environment: CalculatorEnvironment | None = None,
    max_parse_errors: int = 2,
    max_repeated_calls: int = 2,
):
    runner = RolloutRunner(
        backend=_ScriptedBackend(outputs),
        environment=environment or CalculatorEnvironment(prompt_style="compact"),
        config=RolloutConfig(
            max_turns=len(outputs),
            max_new_tokens_per_turn=512,
            max_total_tokens=1600,
            max_parse_errors=max_parse_errors,
            max_repeated_calls=max_repeated_calls,
        ),
    )
    task = Task(task_id="metrics", prompt="Compute 1 + 1.", answer="2")
    return runner.rollout(task, policy_version=0, seed=0)


@pytest.mark.parametrize(
    ("answer", "solved", "format_valid"),
    [
        ("2", True, 1),
        ("3", False, 1),
        ("not-a-number", False, 0),
    ],
)
def test_final_answer_format_is_distinct_from_correctness(
    answer: str, solved: bool, format_valid: int
) -> None:
    trajectory = _run([(render_final(answer), "stop_sequence")])

    assert trajectory.verification is not None
    assert trajectory.verification.solved is solved
    assert trajectory.final_answers_emitted == 1
    assert trajectory.final_answers_verified == 1
    assert trajectory.final_answers_format_valid == format_valid

    stats = RolloutStats()
    stats.observe(trajectory)
    metrics = stats.to_dict()
    assert metrics["final_answer_format_validity_rate"] == pytest.approx(format_valid)


def test_no_final_and_verifier_exception_do_not_claim_format_validity() -> None:
    no_final = _run([("I give up.", "eos")])
    assert no_final.termination_reason is TerminationReason.EOS_WITHOUT_FINAL
    assert no_final.final_answers_emitted == 0
    assert no_final.final_answers_verified == 0
    assert no_final.final_answers_format_valid == 0

    class _BrokenVerifier(CalculatorEnvironment):
        def verify(self, answer: str):
            del answer
            raise ToolEnvironmentError("verifier unavailable")

    verifier_error = _run(
        [(render_final("2"), "stop_sequence")],
        environment=_BrokenVerifier(prompt_style="compact"),
    )
    assert verifier_error.termination_reason is TerminationReason.ENVIRONMENT_ERROR
    assert verifier_error.final_answers_emitted == 1
    assert verifier_error.final_answers_verified == 0
    assert verifier_error.final_answers_format_valid == 0


@pytest.mark.parametrize(
    ("call", "unknown"),
    [
        (render_tool_call("calculator", {}), 0),
        (render_tool_call("missing_tool", {}), 1),
    ],
)
def test_parsed_execution_failure_is_one_call_and_one_execution_error(
    call: str, unknown: int
) -> None:
    trajectory = _run([(call, "stop_sequence")])

    assert trajectory.emitted_tool_calls == 1
    assert trajectory.parsed_tool_calls == 1
    assert trajectory.tool_execution_successes == 0
    assert trajectory.tool_execution_errors == 1
    assert trajectory.unknown_tool_calls == unknown
    assert trajectory.parse_errors == 0

    stats = RolloutStats()
    stats.observe(trajectory)
    metrics = stats.to_dict()
    assert metrics["parse_valid_tool_call_rate"] == pytest.approx(1.0)
    assert metrics["tool_execution_success_rate"] == pytest.approx(0.0)
    assert metrics["tool_execution_error_rate"] == pytest.approx(1.0)


def test_environment_exception_is_an_execution_error_not_a_parse_error() -> None:
    class _BrokenTool(CalculatorEnvironment):
        def step(self, call: ToolCall) -> StepResult:
            del call
            raise ToolEnvironmentError("executor unavailable")

    trajectory = _run(
        [(render_tool_call("calculator", {"expression": "1 + 1"}), "stop_sequence")],
        environment=_BrokenTool(prompt_style="compact"),
    )
    assert trajectory.termination_reason is TerminationReason.ENVIRONMENT_ERROR
    assert trajectory.emitted_tool_calls == 1
    assert trajectory.parsed_tool_calls == 1
    assert trajectory.tool_execution_errors == 1
    assert trajectory.parse_errors == 0


def test_repeated_call_termination_does_not_invent_an_execution_error() -> None:
    call = render_tool_call("calculator", {"expression": "1 + 1"})
    trajectory = _run(
        [(call, "stop_sequence"), (call, "stop_sequence")],
        max_repeated_calls=1,
    )
    assert trajectory.termination_reason is TerminationReason.REPEATED_CALL_LIMIT
    assert trajectory.emitted_tool_calls == 2
    assert trajectory.parsed_tool_calls == 2
    assert trajectory.tool_execution_successes == 1
    assert trajectory.tool_execution_errors == 0
    assert trajectory.repeated_call_terminations == 1


@pytest.mark.parametrize(
    ("limit", "outputs", "expected_errors"),
    [
        (0, [("unparseable", "length")], 1),
        (2, [("first", "length"), ("second", "length")], 2),
    ],
)
def test_parse_error_limit_terminates_when_configured_count_is_reached(
    limit: int, outputs: Sequence[tuple[str, str]], expected_errors: int
) -> None:
    trajectory = _run(outputs, max_parse_errors=limit)
    assert trajectory.termination_reason is TerminationReason.PARSE_ERROR_LIMIT
    assert trajectory.parse_errors == expected_errors
    assert trajectory.tool_execution_errors == 0
    assert trajectory.parsed_tool_calls == 0


def test_reset_observation_is_authoritative_and_reset_runs_once_per_episode() -> None:
    class _DynamicReset(CalculatorEnvironment):
        reset_calls = 0

        def reset(self, task: Task) -> Observation:
            super().reset(task)
            self.reset_calls += 1
            return Observation(text=f"dynamic observation for {task.task_id}", state_id="dynamic:7")

        def user_prompt(self, task: Task) -> str:
            del task
            return "stale compatibility prompt"

    environment = _DynamicReset(prompt_style="compact")
    backend = _ScriptedBackend([(render_final("2"), "stop_sequence")])
    runner = RolloutRunner(
        backend=backend,
        environment=environment,
        config=RolloutConfig(max_turns=1, max_new_tokens_per_turn=512, max_total_tokens=1600),
    )
    task = Task(
        task_id="dynamic",
        prompt="ignored",
        answer="2",
        metadata={"kind": "arithmetic", "expression": "1 + 1"},
    )

    policy = runner.rollout(task, policy_version=0, seed=0)
    assert environment.reset_calls == 1
    user_span = next(span for span in policy.spans if span.span_type is SpanType.USER)
    user_text = backend.tokenizer.decode(policy.token_ids[user_span.start : user_span.end])
    assert "dynamic observation for dynamic" in user_text
    assert "stale compatibility prompt" not in user_text
    assert user_span.env_state_id == "dynamic:7"
    assert policy.metadata["initial_observation_state_id"] == "dynamic:7"

    oracle = runner.oracle_rollout(task)
    assert environment.reset_calls == 2
    oracle_user = next(span for span in oracle.spans if span.span_type is SpanType.USER)
    assert oracle_user.env_state_id == "dynamic:7"
    assert oracle.metadata["initial_observation_state_id"] == "dynamic:7"
