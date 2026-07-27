"""Local (same-process) teacher scorer over a :class:`CausalLMBackend`."""

from __future__ import annotations

from typing import Any

import torch

from miniverl.config.models import LossConfig, LossMode
from miniverl.errors import AlignmentError, ConfigError, TokenizerMismatchError
from miniverl.losses.bucketed import bucketed_teacher_entropy, teacher_topk_targets
from miniverl.losses.chunked import BucketedTargetProvider, ExactTargetProvider
from miniverl.losses.exact import exact_teacher_entropy
from miniverl.models.base import CausalLMBackend
from miniverl.schemas.alignment import AlignmentMap
from miniverl.schemas.cache import TeacherTargetBatch
from miniverl.schemas.trajectory import Trajectory
from miniverl.teachers.base import TeacherScoreResult, TeacherScorer

__all__ = ["LocalTeacherScorer"]


class LocalTeacherScorer(TeacherScorer):
    """Scores student states with a frozen local model.

    ``keep_exact_resident`` chooses between the two supervision shapes.  It is
    set by the trainer from ``memory.strategy``: a resident teacher keeps its
    hidden states and LM head available (cheapest exact path), while a swapped
    teacher must materialize serializable targets before it is evicted.
    """

    def __init__(
        self,
        backend: CausalLMBackend,
        loss: LossConfig,
        *,
        keep_exact_resident: bool = True,
        device: str | None = None,
    ) -> None:
        self.backend = backend
        self.loss = loss
        self.keep_exact_resident = keep_exact_resident
        self.device = device or backend.device

    # -- helpers ---------------------------------------------------------

    def _effective_top_k(self) -> int:
        if self.loss.mode is LossMode.EXACT_FULL_VOCAB:
            return int(self.backend.vocab_size)
        return int(min(self.loss.top_k, self.backend.vocab_size))

    def _check_exact_is_affordable(self) -> None:
        vocab = int(self.backend.vocab_size)
        if vocab <= self.loss.exact_max_vocab or self.loss.allow_large_exact:
            return
        raise ConfigError(
            f"loss.mode=exact_full_vocab with memory.strategy=swap must persist a "
            f"[positions, {vocab}] teacher tensor, which exceeds the "
            f"loss.exact_max_vocab={self.loss.exact_max_vocab} guard rail.",
            hint="use loss.mode=bucketed_topk_tail for large vocabularies, or set "
            "memory.strategy=resident so the exact teacher distribution can be "
            "rebuilt one chunk at a time, or set loss.allow_large_exact=true if "
            "you really mean it",
        )

    # -- scoring ----------------------------------------------------------

    def score(
        self,
        *,
        student: Trajectory,
        alignment: AlignmentMap,
        teacher_view: Trajectory | None = None,
    ) -> TeacherScoreResult:
        """Run the teacher over the student's states and compress the targets."""
        source = teacher_view if teacher_view is not None else student
        if source.tokenizer_fingerprint != student.tokenizer_fingerprint:
            raise TokenizerMismatchError(
                "the teacher view was produced by a different tokenizer than the student"
            )
        if teacher_view is None and not alignment.is_identity():
            raise AlignmentError(
                "a non-identity alignment was given without a teacher view; "
                "privileged-context scoring needs the re-rendered teacher trajectory"
            )

        positions = alignment.teacher_prediction_positions
        target_ids = torch.tensor(alignment.target_token_ids, dtype=torch.long)
        weights = torch.tensor(alignment.token_weights, dtype=torch.float32)
        n = len(positions)

        if n == 0:
            empty = torch.zeros(0, dtype=torch.float32)
            return TeacherScoreResult(
                trajectory_id=student.trajectory_id,
                policy_version=student.policy_version,
                shape="bucketed",
                provider=BucketedTargetProvider(
                    topk_indices=torch.zeros(0, 1, dtype=torch.long),
                    topk_log_probs=torch.zeros(0, 1),
                    tail_log_prob=empty,
                    divergence_name=self.loss.divergence.value,
                    temperature=self.loss.temperature,
                    scale_by_temperature_squared=self.loss.scale_by_temperature_squared,
                    jsd_beta=self.loss.jsd_beta,
                    tail_epsilon=self.loss.tail_epsilon,
                ),
                target_token_ids=target_ids,
                weights=weights,
                span_types=list(alignment.span_types),
                teacher_entropy=empty,
                num_positions=0,
                cacheable=None,
                metrics={"selected_positions": 0.0},
            )

        exact_resident = (
            self.loss.mode is LossMode.EXACT_FULL_VOCAB and self.keep_exact_resident
        )

        with torch.no_grad():
            hidden = self.backend.hidden_states_at(source.token_ids, positions, with_grad=False)

            if exact_resident:
                # Keep only [N, H]; the [chunk, V] distribution is rebuilt on demand.
                hidden = hidden.detach()
                project = self.backend.project
                entropy_parts = []
                for start in range(0, n, self.loss.chunk_size):
                    chunk = project(hidden[start : start + self.loss.chunk_size])
                    entropy_parts.append(
                        exact_teacher_entropy(chunk, temperature=self.loss.temperature)
                    )
                    del chunk
                provider: Any = ExactTargetProvider(
                    teacher_logits_fn=lambda a, b: project(hidden[a:b]),
                    divergence_name=self.loss.divergence.value,
                    temperature=self.loss.temperature,
                    scale_by_temperature_squared=self.loss.scale_by_temperature_squared,
                    jsd_beta=self.loss.jsd_beta,
                )
                return TeacherScoreResult(
                    trajectory_id=student.trajectory_id,
                    policy_version=student.policy_version,
                    shape="exact_hidden",
                    provider=provider,
                    target_token_ids=target_ids,
                    weights=weights,
                    span_types=list(alignment.span_types),
                    teacher_entropy=torch.cat(entropy_parts),
                    num_positions=n,
                    cacheable=None,
                    metrics={
                        "selected_positions": float(n),
                        "teacher_hidden_bytes": float(hidden.numel() * hidden.element_size()),
                    },
                )

            if self.loss.mode is LossMode.EXACT_FULL_VOCAB:
                self._check_exact_is_affordable()

            top_k = self._effective_top_k()
            index_parts, logprob_parts, tail_parts = [], [], []
            for start in range(0, n, self.loss.chunk_size):
                logits = self.backend.project(
                    hidden[start : start + self.loss.chunk_size]
                ).to(torch.float32)
                idx, lp, tail = teacher_topk_targets(
                    logits, top_k=top_k, temperature=self.loss.temperature
                )
                index_parts.append(idx)
                logprob_parts.append(lp)
                tail_parts.append(tail)
                del logits
            topk_indices = torch.cat(index_parts)
            topk_log_probs = torch.cat(logprob_parts)
            tail_log_prob = torch.cat(tail_parts)
            del hidden, index_parts, logprob_parts, tail_parts

            entropy = bucketed_teacher_entropy(
                topk_log_probs, tail_log_prob, tail_epsilon=self.loss.tail_epsilon
            )

        batch = TeacherTargetBatch(
            trajectory_id=student.trajectory_id,
            policy_version=student.policy_version,
            positions=torch.tensor(positions, dtype=torch.long),
            topk_indices=topk_indices,
            topk_log_probs=topk_log_probs,
            tail_log_prob=tail_log_prob,
            target_token_ids=target_ids,
            weights=weights,
            temperature=self.loss.temperature,
            top_k=top_k,
            span_types=list(alignment.span_types),
        )
        provider = BucketedTargetProvider(
            topk_indices=topk_indices,
            topk_log_probs=topk_log_probs,
            tail_log_prob=tail_log_prob,
            divergence_name=self.loss.divergence.value,
            temperature=self.loss.temperature,
            scale_by_temperature_squared=self.loss.scale_by_temperature_squared,
            jsd_beta=self.loss.jsd_beta,
            tail_epsilon=self.loss.tail_epsilon,
        )
        covered = torch.logsumexp(topk_log_probs, dim=-1).exp()
        return TeacherScoreResult(
            trajectory_id=student.trajectory_id,
            policy_version=student.policy_version,
            shape="bucketed",
            provider=provider,
            target_token_ids=target_ids,
            weights=weights,
            span_types=list(alignment.span_types),
            teacher_entropy=entropy,
            num_positions=n,
            cacheable=batch,
            metrics={
                "selected_positions": float(n),
                "teacher_topk_mass_mean": float(covered.mean()),
                "teacher_topk_mass_min": float(covered.min()),
                "top_k": float(top_k),
            },
        )

    def describe(self) -> dict[str, Any]:
        """Teacher identity for the manifest."""
        return {
            "kind": "local",
            "model_id": getattr(self.backend, "model_id", self.backend.capabilities.name),
            "revision": getattr(self.backend, "model_revision", None),
            "capabilities": self.backend.capabilities.to_dict(),
            "loss_mode": self.loss.mode.value,
            "top_k": self._effective_top_k(),
            "temperature": self.loss.temperature,
        }
