"""Multi-turn rollout loop.

Produces the trajectories that on-policy distillation is defined on: sampled
from the *current* student policy, with real tool execution interleaved, and
with every token's provenance recorded as it is produced rather than
reconstructed afterwards.

Two entry points share the same transcript machinery:

:meth:`RolloutRunner.rollout`
    Samples from the policy.  Model spans hold the **sampled token ids
    verbatim** -- nothing is re-tokenized, so the teacher later scores exactly
    the states the student visited.

:meth:`RolloutRunner.oracle_rollout`
    Renders the environment's reference actions.  Used for SFT cold starts and
    as the offline-KD trajectory source.

Every loop is bounded: turns, new tokens per turn, total tokens, parse errors
and repeated identical calls all have hard caps, and each cap has its own
termination reason so the failure taxonomy in reports is exact.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from miniverl.agent.protocol import (
    FINAL_OPEN,
    TOOL_CALL_OPEN,
    ActionKind,
    parse_assistant_text,
    render_final,
    render_tool_call,
    render_tool_result,
    stop_sequences,
)
from miniverl.agent.transcript import (
    ChatFormat,
    Segment,
    TokenizerLike,
    TranscriptBuilder,
    token_index_at_char,
)
from miniverl.config.models import RolloutConfig
from miniverl.environments.base import (
    FailureCategory,
    Observation,
    OracleActionKind,
    Task,
    ToolCall,
    ToolEnvironment,
)
from miniverl.errors import ToolEnvironmentError
from miniverl.models.base import CausalLMBackend
from miniverl.schemas.trajectory import (
    SpanType,
    TerminationReason,
    ToolCallRecord,
    ToolResultRecord,
    Trajectory,
    Turn,
    VerificationRecord,
)

__all__ = ["RolloutRunner", "RolloutStats"]


@dataclass
class RolloutStats:
    """Counters accumulated over a batch of rollouts."""

    rollouts: int = 0
    solved: int = 0
    turns: int = 0
    assistant_turns: int = 0
    emitted_tool_calls: int = 0
    parsed_tool_calls: int = 0
    tool_execution_successes: int = 0
    tool_execution_errors: int = 0
    unknown_tool_calls: int = 0
    parse_errors: int = 0
    repeated_call_terminations: int = 0
    final_answers_emitted: int = 0
    final_answers_format_valid: int = 0
    final_answers_verified: int = 0
    # Deprecated compatibility aliases. New code and artifacts should use the
    # event-specific counters above.
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    valid_final_answers: int = 0
    generated_tokens: int = 0
    termination_reasons: dict[str, int] | None = None
    failure_categories: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.termination_reasons is None:
            self.termination_reasons = {}
        if self.failure_categories is None:
            self.failure_categories = {}

    def observe(self, trajectory: Trajectory) -> None:
        """Fold one trajectory into the counters."""
        assert self.termination_reasons is not None
        assert self.failure_categories is not None
        self.rollouts += 1
        self.turns += len(trajectory.turns)
        has_precise_counters = getattr(trajectory, "assistant_turns", None) is not None
        if has_precise_counters:
            assistant_turns = int(trajectory.assistant_turns or 0)
            emitted_tool_calls = int(trajectory.emitted_tool_calls or 0)
            parsed_tool_calls = int(trajectory.parsed_tool_calls or 0)
            execution_successes = int(trajectory.tool_execution_successes or 0)
            execution_errors = int(trajectory.tool_execution_errors or 0)
            unknown_tool_calls = int(trajectory.unknown_tool_calls or 0)
            parse_errors = int(trajectory.parse_errors or 0)
            repeated_terminations = int(trajectory.repeated_call_terminations or 0)
            final_answers_emitted = int(trajectory.final_answers_emitted or 0)
            final_answers_format_valid = int(trajectory.final_answers_format_valid or 0)
            final_answers_verified = int(trajectory.final_answers_verified or 0)
        else:
            # Legacy trajectories predate the event counters. Derive the
            # narrowest backward-compatible interpretation available from their
            # turn records without claiming unavailable distinctions.
            assistant_turns = len(trajectory.turns)
            parsed_turns = [
                turn
                for turn in trajectory.turns
                if turn.tool_call is not None and turn.tool_call.valid
            ]
            parse_error_turns = [
                turn
                for turn in trajectory.turns
                if turn.tool_call is not None and not turn.tool_call.valid
            ]
            emitted_tool_calls = len(parsed_turns) + len(parse_error_turns)
            parsed_tool_calls = len(parsed_turns)
            execution_successes = 0
            execution_errors = 0
            for turn in parsed_turns:
                tool_result = getattr(turn, "tool_result", None)
                if tool_result is None or tool_result.ok:
                    execution_successes += 1
                else:
                    execution_errors += 1
            unknown_tool_calls = 0
            parse_errors = len(parse_error_turns)
            repeated_terminations = int(
                trajectory.termination_reason is TerminationReason.REPEATED_CALL_LIMIT
            )
            final_answers_emitted = int(
                trajectory.termination_reason is TerminationReason.FINAL_ANSWER
            )
            final_answers_verified = int(trajectory.verification is not None)
            final_answers_format_valid = int(
                trajectory.verification is not None
                and trajectory.verification.failure_category
                != FailureCategory.MALFORMED_ANSWER.value
            )

        self.assistant_turns += assistant_turns
        self.emitted_tool_calls += emitted_tool_calls
        self.parsed_tool_calls += parsed_tool_calls
        self.tool_execution_successes += execution_successes
        self.tool_execution_errors += execution_errors
        self.unknown_tool_calls += unknown_tool_calls
        self.parse_errors += parse_errors
        self.repeated_call_terminations += repeated_terminations
        self.final_answers_emitted += final_answers_emitted
        self.final_answers_format_valid += final_answers_format_valid
        self.final_answers_verified += final_answers_verified
        self.tool_calls += parsed_tool_calls
        self.invalid_tool_calls += trajectory.invalid_tool_calls
        self.generated_tokens += trajectory.generated_token_count
        reason = trajectory.termination_reason.value
        self.termination_reasons[reason] = self.termination_reasons.get(reason, 0) + 1
        if trajectory.verification is not None:
            self.valid_final_answers += final_answers_format_valid
            if trajectory.verification.solved:
                self.solved += 1
            category = trajectory.verification.failure_category or "solved"
            self.failure_categories[category] = self.failure_categories.get(category, 0) + 1
        else:
            self.failure_categories["no_final_answer"] = (
                self.failure_categories.get("no_final_answer", 0) + 1
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly metrics view."""
        assert self.termination_reasons is not None
        assert self.failure_categories is not None
        n = max(self.rollouts, 1)
        tool_execution_attempts = self.tool_execution_successes + self.tool_execution_errors
        parse_valid_tool_call_rate = (
            self.parsed_tool_calls / self.emitted_tool_calls if self.emitted_tool_calls else None
        )
        tool_execution_success_rate = (
            self.tool_execution_successes / tool_execution_attempts
            if tool_execution_attempts
            else None
        )
        tool_execution_error_rate = (
            self.tool_execution_errors / tool_execution_attempts
            if tool_execution_attempts
            else None
        )
        return {
            "rollouts": self.rollouts,
            "solved": self.solved,
            "success_rate": self.solved / n,
            "strict_task_success_rate": self.solved / n,
            "assistant_turns": self.assistant_turns,
            "emitted_tool_calls": self.emitted_tool_calls,
            "parsed_tool_calls": self.parsed_tool_calls,
            "tool_execution_successes": self.tool_execution_successes,
            "tool_execution_errors": self.tool_execution_errors,
            "unknown_tool_calls": self.unknown_tool_calls,
            "parse_errors": self.parse_errors,
            "repeated_call_terminations": self.repeated_call_terminations,
            "final_answers_emitted": self.final_answers_emitted,
            "final_answers_format_valid": self.final_answers_format_valid,
            "final_answers_verified": self.final_answers_verified,
            "parse_valid_tool_call_rate": parse_valid_tool_call_rate,
            "tool_execution_success_rate": tool_execution_success_rate,
            "tool_execution_error_rate": tool_execution_error_rate,
            "final_answer_format_validity_rate": (
                self.final_answers_format_valid / self.final_answers_emitted
                if self.final_answers_emitted
                else None
            ),
            "avg_turns": self.assistant_turns / n,
            "avg_tool_calls": self.parsed_tool_calls / n,
            # Deprecated aliases retained for backward readers. A "valid" tool
            # call means parse-valid here; execution success is reported above.
            "tool_call_count": self.emitted_tool_calls,
            "valid_tool_call_rate": parse_valid_tool_call_rate,
            "invalid_tool_call_rate": (
                self.parse_errors / self.emitted_tool_calls if self.emitted_tool_calls else 0.0
            ),
            "generated_tokens": self.generated_tokens,
            "generated_tokens_per_task": self.generated_tokens / n,
            # None, not NaN: JSON has no NaN literal, so writing one produces a
            # metrics file that strict parsers (JavaScript's JSON.parse among
            # them) reject. The quantity is genuinely undefined when nothing was
            # solved, which is what null means.
            "tokens_per_solved_task": (
                self.generated_tokens / self.solved if self.solved else None
            ),
            "termination_reasons": dict(self.termination_reasons),
            "failure_categories": dict(self.failure_categories),
        }


def _canonical_call(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, ensure_ascii=False)


class RolloutRunner:
    """Runs bounded multi-turn episodes against a tool environment."""

    def __init__(
        self,
        *,
        backend: CausalLMBackend,
        environment: ToolEnvironment,
        config: RolloutConfig,
        chat_format: ChatFormat | None = None,
    ) -> None:
        self.backend = backend
        self.environment = environment
        self.config = config
        self.chat_format = chat_format or ChatFormat()

    # -- helpers ---------------------------------------------------------

    @property
    def tokenizer(self) -> TokenizerLike:
        """The backend's tokenizer."""
        return self.backend.tokenizer

    def _new_builder(self, initial_observation: Observation) -> TranscriptBuilder:
        builder = TranscriptBuilder(self.tokenizer, self.chat_format)
        builder.add_context(
            key="sys",
            span_type=SpanType.SYSTEM,
            turn_id=0,
            role="system",
            body=self.environment.system_prompt(),
            open_next_assistant=False,
        )
        builder.add_context(
            key="user",
            span_type=SpanType.USER,
            turn_id=0,
            role="user",
            body=initial_observation.text,
            open_next_assistant=True,
            env_state_id=initial_observation.state_id,
        )
        return builder

    def _add_observation(
        self,
        builder: TranscriptBuilder,
        *,
        turn_id: int,
        body: str,
        state_id: str | None,
        call_id: str | None,
    ) -> None:
        builder.add_context(
            key=f"obs:{turn_id}",
            span_type=SpanType.TOOL_RESULT,
            turn_id=turn_id,
            role="user",
            body=body,
            close_previous=True,
            open_next_assistant=True,
            tool_call_id=call_id,
            env_state_id=state_id,
        )

    def _add_model_spans(
        self,
        builder: TranscriptBuilder,
        *,
        turn_id: int,
        token_ids: list[int],
        text: str,
        typed_kind: SpanType,
        block_start: int,
        tool_name: str | None,
        call_id: str | None,
    ) -> None:
        """Append the sampled tokens, optionally split into prefix + typed span.

        Token ids are never re-derived from text: the split index is computed by
        decoding prefixes of the *sampled* ids, so the stored sequence is
        bit-identical to what the policy produced.
        """
        split = 0
        if block_start > 0 and text[:block_start].strip():
            split = token_index_at_char(self.tokenizer, token_ids, block_start)
        if 0 < split < len(token_ids):
            builder.add(
                Segment(
                    key=f"gen:{turn_id}:text",
                    span_type=SpanType.ASSISTANT_TEXT,
                    turn_id=turn_id,
                    token_ids=token_ids[:split],
                )
            )
            builder.add(
                Segment(
                    key=f"gen:{turn_id}:block",
                    span_type=typed_kind,
                    turn_id=turn_id,
                    token_ids=token_ids[split:],
                    tool_name=tool_name,
                    tool_call_id=call_id,
                )
            )
        else:
            builder.add(
                Segment(
                    key=f"gen:{turn_id}:block",
                    span_type=typed_kind,
                    turn_id=turn_id,
                    token_ids=list(token_ids),
                    tool_name=tool_name,
                    tool_call_id=call_id,
                )
            )

    # -- policy rollout ---------------------------------------------------

    def rollout(
        self,
        task: Task,
        *,
        policy_version: int,
        seed: int,
        temperature: float | None = None,
        max_turns: int | None = None,
        trajectory_id: str | None = None,
    ) -> Trajectory:
        """Sample one full episode from the current policy."""
        cfg = self.config
        turn_budget = max_turns if max_turns is not None else cfg.max_turns
        sample_temperature = cfg.temperature if temperature is None else temperature

        initial_observation = self.environment.reset(task)
        builder = self._new_builder(initial_observation)
        turns: list[Turn] = []
        call_counts: Counter[str] = Counter()
        invalid_calls = 0
        assistant_turns = 0
        emitted_tool_calls = 0
        parsed_tool_calls = 0
        tool_execution_successes = 0
        tool_execution_errors = 0
        unknown_tool_calls = 0
        parse_errors = 0
        repeated_call_terminations = 0
        final_answers_emitted = 0
        final_answers_format_valid = 0
        final_answers_verified = 0
        generated_tokens = 0
        verification: VerificationRecord | None = None
        metadata_error: str | None = None
        termination = TerminationReason.MAX_TURNS

        for turn_id in range(turn_budget):
            remaining = cfg.max_total_tokens - builder.length
            if remaining <= 0:
                termination = TerminationReason.MAX_TOKENS
                break
            generation = self.backend.generate(
                builder.token_ids,
                max_new_tokens=min(cfg.max_new_tokens_per_turn, remaining),
                stop_sequences=stop_sequences(),
                temperature=sample_temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                seed=seed * 1_000_003 + turn_id,
            )
            if not generation.token_ids:
                termination = TerminationReason.EOS_WITHOUT_FINAL
                break
            assistant_turns += 1
            generated_tokens += len(generation.token_ids)
            parsed = parse_assistant_text(generation.text)
            if TOOL_CALL_OPEN in generation.text:
                emitted_tool_calls += 1
            if FINAL_OPEN in generation.text:
                final_answers_emitted += 1

            if parsed.kind is ActionKind.FINAL:
                self._add_model_spans(
                    builder,
                    turn_id=turn_id,
                    token_ids=generation.token_ids,
                    text=generation.text,
                    typed_kind=SpanType.ASSISTANT_FINAL,
                    block_start=parsed.block_start,
                    tool_name=None,
                    call_id=None,
                )
                try:
                    result = self.environment.verify(parsed.final_answer or "")
                except ToolEnvironmentError as exc:
                    turns.append(Turn(turn_id=turn_id, is_final=True))
                    termination = TerminationReason.ENVIRONMENT_ERROR
                    metadata_error = exc.message
                    break
                verification = VerificationRecord(
                    solved=result.solved,
                    reward=result.reward,
                    predicted=result.predicted,
                    expected=result.expected,
                    failure_category=result.failure_category.value,
                    detail=result.detail,
                )
                final_answers_verified += 1
                if result.failure_category is not FailureCategory.MALFORMED_ANSWER:
                    final_answers_format_valid += 1
                turns.append(Turn(turn_id=turn_id, is_final=True))
                termination = TerminationReason.FINAL_ANSWER
                break

            if parsed.kind is ActionKind.TOOL_CALL:
                parsed_tool_calls += 1
                call_id = f"c{turn_id}"
                name = parsed.tool_name or ""
                arguments = parsed.arguments or {}
                self._add_model_spans(
                    builder,
                    turn_id=turn_id,
                    token_ids=generation.token_ids,
                    text=generation.text,
                    typed_kind=SpanType.ASSISTANT_TOOL_CALL,
                    block_start=parsed.block_start,
                    tool_name=name,
                    call_id=call_id,
                )
                signature = _canonical_call(name, arguments)
                call_counts[signature] += 1
                if call_counts[signature] > cfg.max_repeated_calls:
                    self._add_observation(
                        builder,
                        turn_id=turn_id,
                        body=render_tool_result(
                            False,
                            error=(
                                f"identical call repeated more than {cfg.max_repeated_calls} "
                                "times; the episode was stopped"
                            ),
                        ),
                        state_id=None,
                        call_id=call_id,
                    )
                    turns.append(
                        Turn(
                            turn_id=turn_id,
                            tool_call=ToolCallRecord(
                                call_id=call_id,
                                name=name,
                                arguments=arguments,
                                raw_text=generation.text,
                            ),
                            tool_result=ToolResultRecord(
                                call_id=call_id, ok=False, error="repeated call limit"
                            ),
                        )
                    )
                    repeated_call_terminations += 1
                    termination = TerminationReason.REPEATED_CALL_LIMIT
                    break

                try:
                    step = self.environment.step(ToolCall(name=name, arguments=arguments))
                except ToolEnvironmentError as exc:
                    # A tool that raises rather than returning StepResult(ok=False)
                    # is an environment defect, not a policy mistake. End the
                    # episode with its own termination reason so the failure shows
                    # up in the taxonomy instead of being blamed on the model.
                    self._add_observation(
                        builder,
                        turn_id=turn_id,
                        body=render_tool_result(False, error=f"environment error: {exc.message}"),
                        state_id=None,
                        call_id=call_id,
                    )
                    turns.append(
                        Turn(
                            turn_id=turn_id,
                            tool_call=ToolCallRecord(
                                call_id=call_id,
                                name=name,
                                arguments=arguments,
                                raw_text=generation.text,
                            ),
                            tool_result=ToolResultRecord(
                                call_id=call_id, ok=False, error=exc.message
                            ),
                        )
                    )
                    tool_execution_errors += 1
                    termination = TerminationReason.ENVIRONMENT_ERROR
                    metadata_error = exc.message
                    break
                if not step.ok:
                    invalid_calls += 1
                    tool_execution_errors += 1
                    if step.failure_category is FailureCategory.UNKNOWN_TOOL:
                        unknown_tool_calls += 1
                else:
                    tool_execution_successes += 1
                self._add_observation(
                    builder,
                    turn_id=turn_id,
                    body=render_tool_result(step.ok, result=step.result, error=step.error),
                    state_id=step.state_id,
                    call_id=call_id,
                )
                turns.append(
                    Turn(
                        turn_id=turn_id,
                        tool_call=ToolCallRecord(
                            call_id=call_id,
                            name=name,
                            arguments=arguments,
                            raw_text=generation.text,
                            valid=True,
                        ),
                        tool_result=ToolResultRecord(
                            call_id=call_id,
                            ok=step.ok,
                            result=step.result,
                            error=step.error,
                            env_state_id=step.state_id,
                        ),
                    )
                )
                continue

            # Parse error.
            invalid_calls += 1
            parse_errors += 1
            self._add_model_spans(
                builder,
                turn_id=turn_id,
                token_ids=generation.token_ids,
                text=generation.text,
                typed_kind=SpanType.ASSISTANT_TEXT,
                block_start=0,
                tool_name=None,
                call_id=None,
            )
            turns.append(
                Turn(
                    turn_id=turn_id,
                    tool_call=ToolCallRecord(
                        call_id=f"c{turn_id}",
                        name="<unparsed>",
                        raw_text=generation.text,
                        valid=False,
                        parse_error=parsed.error,
                    ),
                )
            )
            if generation.stop_reason == "eos":
                termination = TerminationReason.EOS_WITHOUT_FINAL
                break
            if parse_errors >= max(cfg.max_parse_errors, 1):
                termination = TerminationReason.PARSE_ERROR_LIMIT
                break
            self._add_observation(
                builder,
                turn_id=turn_id,
                body=render_tool_result(False, error=str(parsed.error)),
                state_id=None,
                call_id=None,
            )

        if verification is None and termination is TerminationReason.FINAL_ANSWER:
            # Defensive: FINAL always sets a verification record above.
            termination = TerminationReason.EOS_WITHOUT_FINAL

        traj_id = trajectory_id or f"{task.task_id}:v{policy_version}:s{seed}"
        return builder.build(
            trajectory_id=traj_id,
            task_id=task.task_id,
            environment=self.environment.name,
            model_id=getattr(self.backend, "model_id", self.backend.capabilities.name),
            model_revision=getattr(self.backend, "model_revision", None),
            policy_version=policy_version,
            termination_reason=termination,
            turns=turns,
            verification=verification,
            generated_token_count=generated_tokens,
            invalid_tool_calls=invalid_calls,
            event_counters={
                "assistant_turns": assistant_turns,
                "emitted_tool_calls": emitted_tool_calls,
                "parsed_tool_calls": parsed_tool_calls,
                "tool_execution_successes": tool_execution_successes,
                "tool_execution_errors": tool_execution_errors,
                "unknown_tool_calls": unknown_tool_calls,
                "parse_errors": parse_errors,
                "repeated_call_terminations": repeated_call_terminations,
                "final_answers_emitted": final_answers_emitted,
                "final_answers_format_valid": final_answers_format_valid,
                "final_answers_verified": final_answers_verified,
            },
            metadata={
                "source": "policy",
                "seed": seed,
                "temperature": sample_temperature,
                "difficulty": task.difficulty,
                "split": task.split,
                "initial_observation_state_id": initial_observation.state_id,
                **({"environment_error": metadata_error} if metadata_error else {}),
            },
        )

    # -- oracle rollout ----------------------------------------------------

    def oracle_rollout(
        self,
        task: Task,
        *,
        policy_version: int = 0,
        trajectory_id: str | None = None,
    ) -> Trajectory:
        """Render the environment's reference actions as a trajectory.

        Unlike :meth:`rollout`, this does **not** catch
        :class:`~miniverl.errors.ToolEnvironmentError`. An oracle that cannot
        execute its own reference actions is a defect in the environment, and a
        truncated oracle trace would silently degrade every SFT target built
        from it, so the failure is allowed to propagate.
        """
        initial_observation = self.environment.reset(task)
        builder = self._new_builder(initial_observation)
        turns: list[Turn] = []
        generated_tokens = 0
        verification: VerificationRecord | None = None
        termination = TerminationReason.MAX_TURNS

        for turn_id, action in enumerate(self.environment.oracle_actions(task)):
            if action.kind is OracleActionKind.FINAL:
                text = render_final(action.answer or "")
                segment = builder.add(
                    Segment(
                        key=f"gen:{turn_id}:block",
                        span_type=SpanType.ASSISTANT_FINAL,
                        turn_id=turn_id,
                        text=text,
                    )
                )
                generated_tokens += len(segment.token_ids)
                result = self.environment.verify(action.answer or "")
                verification = VerificationRecord(
                    solved=result.solved,
                    reward=result.reward,
                    predicted=result.predicted,
                    expected=result.expected,
                    failure_category=result.failure_category.value,
                    detail=result.detail,
                )
                turns.append(Turn(turn_id=turn_id, is_final=True))
                termination = TerminationReason.FINAL_ANSWER
                break

            call_id = f"c{turn_id}"
            name = action.tool_name or ""
            text = render_tool_call(name, action.arguments)
            segment = builder.add(
                Segment(
                    key=f"gen:{turn_id}:block",
                    span_type=SpanType.ASSISTANT_TOOL_CALL,
                    turn_id=turn_id,
                    text=text,
                    tool_name=name,
                    tool_call_id=call_id,
                )
            )
            generated_tokens += len(segment.token_ids)
            step = self.environment.step(ToolCall(name=name, arguments=action.arguments))
            self._add_observation(
                builder,
                turn_id=turn_id,
                body=render_tool_result(step.ok, result=step.result, error=step.error),
                state_id=step.state_id,
                call_id=call_id,
            )
            turns.append(
                Turn(
                    turn_id=turn_id,
                    tool_call=ToolCallRecord(
                        call_id=call_id, name=name, arguments=action.arguments, raw_text=text
                    ),
                    tool_result=ToolResultRecord(
                        call_id=call_id,
                        ok=step.ok,
                        result=step.result,
                        error=step.error,
                        env_state_id=step.state_id,
                    ),
                )
            )

        traj_id = trajectory_id or f"{task.task_id}:oracle"
        return builder.build(
            trajectory_id=traj_id,
            task_id=task.task_id,
            environment=self.environment.name,
            model_id=getattr(self.backend, "model_id", self.backend.capabilities.name),
            model_revision=getattr(self.backend, "model_revision", None),
            policy_version=policy_version,
            termination_reason=termination,
            turns=turns,
            verification=verification,
            generated_token_count=generated_tokens,
            invalid_tool_calls=0,
            event_counters={
                "assistant_turns": len(turns),
                "emitted_tool_calls": sum(turn.tool_call is not None for turn in turns),
                "parsed_tool_calls": sum(turn.tool_call is not None for turn in turns),
                "tool_execution_successes": sum(
                    turn.tool_result is not None and turn.tool_result.ok for turn in turns
                ),
                "tool_execution_errors": sum(
                    turn.tool_result is not None and not turn.tool_result.ok for turn in turns
                ),
                "unknown_tool_calls": 0,
                "parse_errors": 0,
                "repeated_call_terminations": 0,
                "final_answers_emitted": int(verification is not None),
                "final_answers_format_valid": int(
                    verification is not None
                    and verification.failure_category != FailureCategory.MALFORMED_ANSWER.value
                ),
                "final_answers_verified": int(verification is not None),
            },
            metadata={
                "source": "oracle",
                "difficulty": task.difficulty,
                "split": task.split,
                "initial_observation_state_id": initial_observation.state_id,
            },
        )

    # -- privileged teacher render -----------------------------------------

    def privileged_render(self, student: Trajectory, task: Task) -> Trajectory | None:
        """Rebuild ``student``'s transcript with an oracle block for the teacher.

        The privileged block is inserted as an extra ``system`` segment before
        the shared content, so every student segment keeps its key and its exact
        token ids while its absolute positions shift.
        :func:`miniverl.trajectory.alignment.build_alignment_map` recovers the
        per-segment offsets and verifies the target tokens still match.
        """
        hint = self.environment.privileged_context(task)
        if not hint:
            return None
        builder = TranscriptBuilder(self.tokenizer, self.chat_format)
        builder.add_context(
            key="privileged",
            span_type=SpanType.SYSTEM,
            turn_id=0,
            role="system",
            body=f"Privileged reference (not visible to the student): {hint}",
            open_next_assistant=False,
        )
        for span in student.spans:
            builder.add(
                Segment(
                    key=str(
                        span.metadata.get("segment_key", f"{span.span_type.value}:{span.start}")
                    ),
                    span_type=span.span_type,
                    turn_id=span.turn_id,
                    token_ids=list(student.token_ids[span.start : span.end]),
                    tool_name=span.tool_name,
                    tool_call_id=span.tool_call_id,
                    env_state_id=span.env_state_id,
                )
            )
        return builder.build(
            trajectory_id=f"{student.trajectory_id}:privileged",
            task_id=student.task_id,
            environment=student.environment,
            model_id=student.model_id,
            model_revision=student.model_revision,
            policy_version=student.policy_version,
            termination_reason=student.termination_reason,
            turns=list(student.turns),
            verification=student.verification,
            generated_token_count=student.generated_token_count,
            invalid_tool_calls=student.invalid_tool_calls,
            event_counters={
                field: int(getattr(student, field))
                for field in (
                    "assistant_turns",
                    "emitted_tool_calls",
                    "parsed_tool_calls",
                    "tool_execution_successes",
                    "tool_execution_errors",
                    "unknown_tool_calls",
                    "parse_errors",
                    "repeated_call_terminations",
                    "final_answers_emitted",
                    "final_answers_format_valid",
                    "final_answers_verified",
                )
                if getattr(student, field) is not None
            },
            metadata={"source": "privileged_teacher_render", "of": student.trajectory_id},
        )
