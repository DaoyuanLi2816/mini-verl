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
    tool_calls: int = 0
    invalid_tool_calls: int = 0
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
        self.tool_calls += sum(
            1 for t in trajectory.turns if t.tool_call is not None and t.tool_call.valid
        )
        self.invalid_tool_calls += trajectory.invalid_tool_calls
        self.generated_tokens += trajectory.generated_token_count
        reason = trajectory.termination_reason.value
        self.termination_reasons[reason] = self.termination_reasons.get(reason, 0) + 1
        if trajectory.verification is not None:
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
        return {
            "rollouts": self.rollouts,
            "solved": self.solved,
            "success_rate": self.solved / n,
            "avg_turns": self.turns / n,
            "avg_tool_calls": self.tool_calls / n,
            "invalid_tool_call_rate": self.invalid_tool_calls
            / max(self.tool_calls + self.invalid_tool_calls, 1),
            "generated_tokens": self.generated_tokens,
            "generated_tokens_per_task": self.generated_tokens / n,
            "tokens_per_solved_task": (
                self.generated_tokens / self.solved if self.solved else float("nan")
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

    def _new_builder(self, task: Task) -> TranscriptBuilder:
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
            body=self.environment.user_prompt(task),
            open_next_assistant=True,
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

        self.environment.reset(task)
        builder = self._new_builder(task)
        turns: list[Turn] = []
        call_counts: Counter[str] = Counter()
        invalid_calls = 0
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
            generated_tokens += len(generation.token_ids)
            parsed = parse_assistant_text(generation.text)

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
                turns.append(Turn(turn_id=turn_id, is_final=True))
                termination = TerminationReason.FINAL_ANSWER
                break

            if parsed.kind is ActionKind.TOOL_CALL:
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
                    termination = TerminationReason.ENVIRONMENT_ERROR
                    metadata_error = exc.message
                    break
                if not step.ok:
                    invalid_calls += 1
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
            if invalid_calls > cfg.max_parse_errors:
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
            metadata={
                "source": "policy",
                "seed": seed,
                "temperature": sample_temperature,
                "difficulty": task.difficulty,
                "split": task.split,
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
        self.environment.reset(task)
        builder = self._new_builder(task)
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
            metadata={
                "source": "oracle",
                "difficulty": task.difficulty,
                "split": task.split,
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
            metadata={"source": "privileged_teacher_render", "of": student.trajectory_id},
        )
