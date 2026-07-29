"""A custom teacher scorer, start to finish.

Everything miniVERL needs from a teacher is one method: given the student's
trajectory and an alignment map, return supervision for exactly those aligned
positions. This example wraps the built-in local scorer and *sharpens* the
teacher distribution -- it raises the top-k log-probabilities to a power and
renormalizes, which concentrates mass on the teacher's preferred tokens.

Run it:

    python examples/custom_teacher/sharpened_teacher.py

It asserts that sharpening actually changed the targets, that the tail bucket
still contains valid mass, and that a real training step runs against them.
Nothing is downloaded.

The contract, all of which the trainer relies on:

* score() must not re-generate, re-tokenize or "improve" the trajectory. It is
  handed the student's own states and must score those.
* The returned provider must accept (start, end, student_logits) slices, because
  the loss projects the LM head in chunks.
* target_token_ids, weights and span_types must line up with the alignment map.
* Returning `cacheable` makes the targets persistable; returning None says they
  are live-only and the trainer will not try to write them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from miniverl.config import RunConfig
from miniverl.config.models import LossConfig
from miniverl.losses.bucketed import bucketed_teacher_entropy
from miniverl.losses.chunked import BucketedTargetProvider
from miniverl.schemas.alignment import AlignmentMap
from miniverl.schemas.cache import TeacherTargetBatch
from miniverl.schemas.trajectory import Trajectory
from miniverl.teachers.base import TeacherScorer, TeacherScoreResult
from miniverl.teachers.local import LocalTeacherScorer


class SharpenedTeacherScorer(TeacherScorer):
    """Wraps a local scorer and sharpens its top-k distribution.

    ``sharpness > 1`` concentrates mass on the teacher's preferred tokens, which
    makes reverse KL more aggressively mode-seeking. ``sharpness < 1`` flattens
    it. At ``1.0`` this is exactly the wrapped scorer.

    The tail bucket is preserved and the (K+1)-way distribution is renormalized,
    so the targets remain a proper probability distribution and the bucketed loss
    stays well defined.
    """

    def __init__(self, inner: LocalTeacherScorer, *, sharpness: float = 1.5) -> None:
        if sharpness <= 0.0:
            raise ValueError(f"sharpness must be > 0, got {sharpness}")
        self.inner = inner
        self.sharpness = sharpness
        self.loss: LossConfig = inner.loss

    def score(
        self,
        *,
        student: Trajectory,
        alignment: AlignmentMap,
        teacher_view: Trajectory | None = None,
    ) -> TeacherScoreResult:
        """Score the student's own states, then sharpen the result."""
        base = self.inner.score(student=student, alignment=alignment, teacher_view=teacher_view)
        if base.cacheable is None or base.num_positions == 0:
            # The exact-resident shape has no explicit top-k to sharpen; pass it
            # through untouched rather than pretending to have modified it.
            return base

        batch = base.cacheable
        # Sharpen in log space, keeping the tail as its own category, then
        # renormalize over the K+1 buckets so the targets stay a distribution.
        topk = batch.topk_log_probs * self.sharpness
        tail = torch.where(
            torch.isinf(batch.tail_log_prob),
            batch.tail_log_prob,
            batch.tail_log_prob * self.sharpness,
        )
        stacked = torch.cat([topk, tail.unsqueeze(-1)], dim=-1)
        normalizer = torch.logsumexp(stacked, dim=-1, keepdim=True)
        topk = topk - normalizer
        tail = tail - normalizer.squeeze(-1)

        sharpened = TeacherTargetBatch(
            trajectory_id=batch.trajectory_id,
            policy_version=batch.policy_version,
            positions=batch.positions,
            topk_indices=batch.topk_indices,
            topk_log_probs=topk,
            tail_log_prob=tail,
            target_token_ids=batch.target_token_ids,
            weights=batch.weights,
            temperature=batch.temperature,
            top_k=batch.top_k,
            span_types=list(batch.span_types),
        )
        provider = BucketedTargetProvider(
            topk_indices=sharpened.topk_indices,
            topk_log_probs=sharpened.topk_log_probs,
            tail_log_prob=sharpened.tail_log_prob,
            divergence_name=self.loss.divergence.value,
            temperature=self.loss.temperature,
            scale_by_temperature_squared=self.loss.scale_by_temperature_squared,
            jsd_beta=self.loss.jsd_beta,
            tail_epsilon=self.loss.tail_epsilon,
        )
        return TeacherScoreResult(
            trajectory_id=base.trajectory_id,
            policy_version=base.policy_version,
            shape="bucketed",
            provider=provider,
            target_token_ids=base.target_token_ids,
            weights=base.weights,
            span_types=base.span_types,
            teacher_entropy=bucketed_teacher_entropy(
                sharpened.topk_log_probs,
                sharpened.tail_log_prob,
                tail_epsilon=self.loss.tail_epsilon,
            ),
            num_positions=base.num_positions,
            cacheable=sharpened,
            metrics={**base.metrics, "sharpness": self.sharpness},
        )

    def describe(self) -> dict[str, Any]:
        """Identity recorded in the run manifest."""
        return {**self.inner.describe(), "kind": "sharpened", "sharpness": self.sharpness}


def build_config(output_dir: str) -> RunConfig:
    """A small toy OPD recipe; the teacher is swapped in after construction."""
    return RunConfig.model_validate(
        {
            "schema_version": 1,
            "run": {
                "name": "sharpened-teacher",
                "mode": "opd",
                "seed": 4242,
                "output_dir": output_dir,
            },
            "models": {
                "backend": "toy",
                "device": "cpu",
                "student": {
                    "model_id": "toy-student",
                    "lora": {"enabled": False},
                    "toy": {
                        "hidden_size": 64,
                        "num_layers": 2,
                        "num_heads": 4,
                        "intermediate_size": 128,
                        "max_position_embeddings": 640,
                    },
                },
                "teacher": {
                    "model_id": "toy-teacher",
                    "toy_pretrain_steps": 40,
                    "toy": {
                        "hidden_size": 96,
                        "num_layers": 2,
                        "num_heads": 4,
                        "intermediate_size": 192,
                        "max_position_embeddings": 640,
                    },
                },
            },
            "environment": {
                "name": "calculator",
                "difficulty": "easy",
                "params": {"prompt_style": "compact"},
                "train_tasks": 32,
                "eval_tasks": 4,
                "test_tasks": 4,
            },
            "rollout": {"max_turns": 2, "max_new_tokens_per_turn": 24, "max_total_tokens": 448},
            "loss": {
                "mode": "bucketed_topk_tail",
                "divergence": "reverse_kl",
                "top_k": 16,
                "chunk_size": 64,
            },
            "train": {
                "cycles": 3,
                "rollouts_per_cycle": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 0.002,
                "sft_warmup_cycles": 10,
            },
            "memory": {"strategy": "resident"},
            "cache": {"entries_per_shard": 4},
            "eval": {"enabled": True, "tasks": 4, "temperature": 0.0},
            "report": {"enabled": False},
        }
    )


def main() -> int:
    """Swap in the custom scorer and train with it."""
    from miniverl.trainer import OPDTrainer

    output = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/examples")
    with OPDTrainer.from_config(build_config(str(output)), run_id="sharpened-teacher") as trainer:
        assert isinstance(trainer.scorer, LocalTeacherScorer)
        baseline_scorer = trainer.scorer
        trainer.scorer = SharpenedTeacherScorer(baseline_scorer, sharpness=2.0)
        print(
            "teacher:",
            trainer.scorer.describe()["kind"],
            "sharpness",
            trainer.scorer.describe()["sharpness"],
        )

        result = trainer.train()

    # The targets really were modified: re-score one stored trajectory both ways
    # and confirm the sharpened distribution is more concentrated.
    from miniverl.config.models import SelectionConfig
    from miniverl.selection.selectors import select_positions
    from miniverl.trajectory.alignment import build_alignment_map
    from miniverl.trajectory.io import read_trajectories

    trajectory = read_trajectories(trainer.paths.trajectories)[0]
    selection = select_positions(trajectory, SelectionConfig(), run_seed=4242)
    alignment = build_alignment_map(trajectory, selection.positions, selection.weights)
    plain = baseline_scorer.score(student=trajectory, alignment=alignment)
    sharp = SharpenedTeacherScorer(baseline_scorer, sharpness=2.0).score(
        student=trajectory, alignment=alignment
    )
    plain_entropy = float(plain.teacher_entropy.mean())
    sharp_entropy = float(sharp.teacher_entropy.mean())
    print(f"mean teacher entropy: {plain_entropy:.4f} -> {sharp_entropy:.4f} (nats)")
    assert sharp_entropy < plain_entropy, "sharpening must reduce teacher entropy"

    assert sharp.cacheable is not None
    total = torch.cat(
        [sharp.cacheable.topk_log_probs, sharp.cacheable.tail_log_prob.unsqueeze(-1)],
        dim=-1,
    ).logsumexp(dim=-1)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-5), (
        "the sharpened targets must still be a normalized distribution"
    )

    print(f"\nrun directory : {result.run_dir}")
    print(f"optimizer steps: {result.global_step}")
    print("sharpened targets stayed normalized and trained without error")
    # A reader who runs this file directly never sees examples/README.md, and a
    # bare 0% in the log above reads as a broken example rather than as the
    # expected outcome of a 200k-parameter smoke test.
    print(
        "\nThe run logs 0% task success. That is expected: this example shows how to\n"
        "plug in a teacher, not that the toy model can solve the task at this budget.\n"
        "The measured claim here is the entropy drop, which is what sharpening does."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
