"""Independent trainable value role for single-GPU PPO.

The critic owns a causal-LM backbone and a scalar value head.  It is a separate
logical and optimizer role even when it uses the same pretrained model identity
as the actor.  Single-GPU execution schedules the role temporally; no distributed
worker abstraction leaks into the checkpoint contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from miniverl.models.base import CausalLMBackend
    from miniverl.training.batching import PaddedTrajectoryBatch

__all__ = ["ValueCritic"]


class ValueCritic:
    """A trainable scalar value head over an independently owned backbone."""

    def __init__(self, backbone: CausalLMBackend, *, head_seed: int) -> None:
        self.backbone = backbone
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.value_head = nn.Linear(backbone.hidden_size, 1, bias=True)
            nn.init.zeros_(self.value_head.weight)
            nn.init.zeros_(self.value_head.bias)
        self.value_head.to(backbone.device)

    @property
    def device(self) -> str:
        return self.backbone.device

    def set_train(self, mode: bool) -> None:
        self.backbone.set_train(mode)
        self.value_head.train(mode)

    def values_at(
        self,
        token_ids: list[int],
        positions: list[int],
        *,
        with_grad: bool = False,
    ) -> torch.Tensor:
        hidden = self.backbone.hidden_states_at(token_ids, positions, with_grad=with_grad)
        context = torch.enable_grad() if with_grad else torch.no_grad()
        with context:
            return self.value_head(hidden.to(torch.float32)).squeeze(-1)

    def values_at_batch(
        self,
        batch: PaddedTrajectoryBatch,
        *,
        with_grad: bool = False,
    ) -> torch.Tensor:
        hidden = self.backbone.hidden_states_at_batch(batch, with_grad=with_grad)
        context = torch.enable_grad() if with_grad else torch.no_grad()
        with context:
            return self.value_head(hidden.to(torch.float32)).squeeze(-1)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [*self.backbone.trainable_parameters(), *self.value_head.parameters()]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        state = {
            f"backbone.{name}": tensor
            for name, tensor in self.backbone.trainable_state_dict().items()
        }
        state["value_head.weight"] = self.value_head.weight.detach().to("cpu").clone()
        state["value_head.bias"] = self.value_head.bias.detach().to("cpu").clone()
        return state

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        backbone = {
            name.removeprefix("backbone."): tensor
            for name, tensor in state.items()
            if name.startswith("backbone.")
        }
        if not backbone:
            raise ValueError("critic checkpoint has no backbone trainable state")
        try:
            weight = state["value_head.weight"]
            bias = state["value_head.bias"]
        except KeyError as exc:
            raise ValueError("critic checkpoint has no complete value head") from exc
        self.backbone.load_trainable_state_dict(backbone)
        with torch.no_grad():
            self.value_head.weight.copy_(weight.to(self.value_head.weight.device))
            self.value_head.bias.copy_(bias.to(self.value_head.bias.device))

    def to_device(self, device: str) -> None:
        self.backbone.to_device(device)
        self.value_head.to(device)

    def release(self) -> None:
        self.backbone.release()
        self.value_head.to("cpu")

    def identity(self) -> dict[str, Any]:
        capabilities = self.backbone.capabilities
        return {
            "implementation": "independent-backbone-linear-value-head-v1",
            "backbone": capabilities.name,
            "hidden_size": capabilities.hidden_size,
            "head_dtype": str(self.value_head.weight.dtype).removeprefix("torch."),
        }
