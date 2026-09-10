"""Single-GPU PPO value-role lifecycle.

The general trainer owns orchestration and artifacts.  This component owns the
PPO-specific critic phase: old-value binding, clipped value optimization,
version checks, and actor/critic temporal placement.
"""

from __future__ import annotations

import math
import time
from typing import Any

from miniverl.config.models import MemoryStrategy
from miniverl.errors import ConfigError, LifecycleError
from miniverl.utils.runs import utc_now

__all__ = ["PPOPhaseRuntime"]


class PPOPhaseRuntime:
    """Execute the critic half of PPO against a trainer orchestration host."""

    def __init__(self, host: Any) -> None:
        self.host = host
        self.microbatch_size_history: list[int] = []

    def score_old_values(self, trajectories: list[Any]) -> None:
        """Bind old value predictions to the exact rollout-policy batch."""
        import torch

        host = self.host
        critic = host.critic
        if critic is None:
            raise ConfigError("PPO requires a loaded critic role")
        swap = host.plan.strategy is MemoryStrategy.SWAP
        actor_state = None
        if swap:
            actor_state = host._student_off_device()
            host._critic_to_device()
        try:
            critic.set_train(False)
            for trajectory in trajectories:
                target_positions = [
                    index
                    for index, generated in enumerate(trajectory.model_generated_mask)
                    if generated
                ]
                if not target_positions:
                    raise ConfigError("PPO requires a non-empty generated response")
                values = (
                    critic.values_at(
                        trajectory.token_ids,
                        [index - 1 for index in target_positions],
                        with_grad=False,
                    )
                    .detach()
                    .to("cpu", dtype=torch.float32)
                )
                trajectory.metadata["critic_old_values"] = values.tolist()
                trajectory.metadata["critic_provenance"] = {
                    "identity": critic.identity(),
                    "version": host.critic_update_count,
                    "rollout_policy_version": trajectory.policy_version,
                }
        finally:
            if swap:
                host._critic_off_device()
                host._student_on_device()
                if actor_state is not None:
                    host.student.load_trainable_state_dict(actor_state)

    def compute_group_gradients(
        self, group: list[Any], *, physical_cap: int | None = None
    ) -> dict[str, Any]:
        """Backpropagate one logical PPO critic minibatch."""
        import torch

        from miniverl.algorithms.value import clipped_value_loss
        from miniverl.training.batching import (
            build_padded_trajectory_batch,
            deterministic_padded_token_batches,
        )

        host = self.host
        critic = host.critic
        optimizer = host.critic_optimizer
        if critic is None or optimizer is None:
            raise LifecycleError("PPO critic optimization is unavailable")
        optimizer.zero_grad(set_to_none=True)
        critic.set_train(True)
        requested_batch_size = host.config.train.trajectory_batch_size
        physical_batch_size = (
            len(group) if requested_batch_size == "auto" else int(requested_batch_size)
        )
        physical_batch_size = max(1, min(physical_batch_size, len(group)))
        if physical_cap is not None:
            physical_batch_size = min(physical_batch_size, physical_cap)
        batch_indices = deterministic_padded_token_batches(
            [len(sample.trajectory.token_ids) for sample in group],
            batch_size=physical_batch_size,
            max_padded_tokens=host.config.train.max_update_padded_tokens,
            sort_by_length=host.config.train.length_bucketing,
        )
        total_positions = sum(sample.alignment.num_positions for sample in group)
        loss_total = 0.0
        clipfrac_total = 0.0
        value_sum = 0.0
        for index_group in batch_indices:
            samples = [group[index] for index in index_group]
            batch = build_padded_trajectory_batch(
                token_ids=[sample.trajectory.token_ids for sample in samples],
                selected_positions=[
                    sample.alignment.student_prediction_positions for sample in samples
                ],
                pad_token_id=host.tokenizer.pad_token_id,
                device=critic.device,
            )
            current = critic.values_at_batch(batch, with_grad=True)
            old = torch.cat(
                [
                    torch.tensor(
                        sample.trajectory.metadata["critic_old_values"],
                        dtype=torch.float32,
                        device=critic.device,
                    )
                    for sample in samples
                ]
            )
            returns = torch.cat(
                [
                    torch.tensor(
                        sample.trajectory.metadata["ppo_returns"],
                        dtype=torch.float32,
                        device=critic.device,
                    )
                    for sample in samples
                ]
            )
            output = clipped_value_loss(
                current,
                old,
                returns,
                torch.ones_like(current),
                cliprange_value=host.config.algorithm.cliprange_value,
            )
            positions = int(current.numel())
            scale = positions / max(total_positions, 1)
            (output.loss * scale).backward()
            loss_total += float(output.loss.detach()) * scale
            clipfrac_total += output.metrics["value_clipfrac"] * scale
            value_sum += output.metrics["value_mean"] * positions
        return {
            "value_loss": loss_total,
            "value_clipfrac": clipfrac_total,
            "value_mean": value_sum / max(total_positions, 1),
            "selected_positions": total_positions,
            "physical_trajectory_batches": len(batch_indices),
            "physical_trajectory_batch_size": physical_batch_size,
        }

    def _compute_with_oom_retry(self, group: list[Any]) -> dict[str, Any]:
        """Reduce only physical critic microbatching after a CUDA OOM."""
        from miniverl.utils import gpu
        from miniverl.utils.seeding import capture_rng, restore_rng

        host = self.host
        configured = host.config.train.trajectory_batch_size
        cap = len(group) if configured == "auto" else min(int(configured), len(group))
        attempts = host.config.memory.oom_retries + 1
        last_error: BaseException | None = None
        retry_rng = capture_rng()
        for attempt in range(attempts):
            try:
                return self.compute_group_gradients(group, physical_cap=cap)
            except (RuntimeError, MemoryError) as exc:
                if not gpu.is_oom_error(exc):
                    raise
                last_error = exc
                if host.critic_optimizer is not None:
                    host.critic_optimizer.zero_grad(set_to_none=True)
                restore_rng(retry_rng)
                gpu.empty_cache()
                next_cap = max(1, cap // 2)
                if next_cap == cap or attempt == attempts - 1:
                    break
                self.microbatch_size_history.append(cap)
                cap = next_cap
        from miniverl.errors import GpuMemoryError

        raise GpuMemoryError(
            "CUDA ran out of memory during the PPO critic phase after "
            f"{host.config.memory.oom_retries} physical-microbatch retries",
            hint=(
                "reduce train.trajectory_batch_size or rollout token bounds, or select "
                f"memory.strategy=swap. Original error: {last_error}"
            ),
        )

    def commit_update(self) -> dict[str, float]:
        """Commit one critic step after a complete finite backward pass."""
        import torch

        host = self.host
        if host.critic is None or host.critic_optimizer is None:
            raise LifecycleError("PPO critic optimizer is unavailable")
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                host.critic.trainable_parameters(), host.config.critic.max_grad_norm
            )
        )
        if not math.isfinite(grad_norm):
            host.critic_optimizer.zero_grad(set_to_none=True)
            raise LifecycleError("PPO critic gradients must be finite before an optimizer step")
        lr = host.critic_schedule.lr_at(
            host.cycle
            if host.config.critic.lr_step_unit == "rollout_iteration"
            else host.critic_update_count
        )
        for parameter_group in host.critic_optimizer.param_groups:
            parameter_group["lr"] = lr
        host.critic_optimizer.step()
        host.critic_optimizer.zero_grad(set_to_none=True)
        host.critic_update_count += 1
        return {"grad_norm": grad_norm, "lr": lr}

    def optimize(self, samples: list[Any]) -> list[dict[str, Any]]:
        """Run verl-style critic epochs/minibatches before the actor update."""
        host = self.host
        if host.critic is None:
            raise LifecycleError("PPO has no critic role")
        starting_version = host.critic_update_count
        if not samples:
            raise LifecycleError("PPO critic update requires non-empty rollout samples")
        identity = host.critic.identity()
        for sample in samples:
            trajectory = sample.trajectory
            provenance = trajectory.metadata.get("critic_provenance")
            if (
                not isinstance(provenance, dict)
                or provenance.get("identity") != identity
                or type(provenance.get("version")) is not int
                or provenance["version"] != starting_version
            ):
                raise LifecycleError(
                    "PPO old values disagree with the pre-update critic identity/version"
                )
            if (
                type(provenance.get("rollout_policy_version")) is not int
                or provenance["rollout_policy_version"] != trajectory.policy_version
                or trajectory.policy_version != host.parameter_version
            ):
                raise LifecycleError(
                    "PPO old values disagree with the current rollout policy version"
                )
        swap = host.plan.strategy is MemoryStrategy.SWAP
        actor_state = None
        if swap:
            actor_state = host._student_off_device()
            host._critic_to_device()
        records: list[dict[str, Any]] = []
        try:
            accum = (
                host.config.critic.gradient_accumulation_steps
                or host.config.train.gradient_accumulation_steps
            )
            for epoch in range(host.config.critic.ppo_epochs):
                for start in range(0, len(samples), accum):
                    group = samples[start : start + accum]
                    if not group:
                        continue
                    started = time.perf_counter()
                    record = self._compute_with_oom_retry(group)
                    record.update(self.commit_update())
                    record.update(
                        {
                            "phase": "ppo_critic",
                            "ppo_epoch": epoch,
                            "cycle": host.cycle,
                            "rollout_iteration": host.cycle,
                            "rollout_critic_version": starting_version,
                            "critic_update_count": host.critic_update_count,
                            "critic_microbatch_size_history": list(self.microbatch_size_history),
                            "seconds": round(time.perf_counter() - started, 4),
                            "ts": utc_now(),
                        }
                    )
                    host.metrics_log.write(record)
                    records.append(record)
        finally:
            if swap:
                host._critic_off_device()
                host._student_on_device()
                if actor_state is not None:
                    host.student.load_trainable_state_dict(actor_state)
        return records
