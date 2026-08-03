"""Backend contract shared by the toy and Hugging Face causal-LM backends.

This module is torch-free at import time; ``torch.Tensor`` appears only under
``TYPE_CHECKING``.  Concrete backends import torch at their own module level
and are themselves imported lazily.

The contract is deliberately narrow.  A backend must be able to:

* sample a continuation with explicit stop strings and a seed,
* return hidden states **at chosen positions only**,
* project hidden states through its LM head,
* expose its trainable parameters,
* move itself on and off the accelerator (for the ``swap`` strategy).

Anything else -- schedulers, checkpoint layout, memory policy -- lives in the
trainer, not in the backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from miniverl.agent.transcript import TokenizerLike
    from miniverl.training.batching import PaddedTrajectoryBatch

__all__ = ["GenerationOutput", "BackendCapabilities", "CausalLMBackend"]


@dataclass
class GenerationOutput:
    """One sampled continuation."""

    token_ids: list[int]
    text: str
    stop_reason: str
    matched_stop: str | None = None
    #: Per-token log-probability under the sampling policy, when recorded.
    logprobs: list[float] = field(default_factory=list)


@dataclass
class BackendCapabilities:
    """What a loaded backend can actually do, resolved at load time."""

    name: str
    device: str
    dtype: str
    vocab_size: int
    hidden_size: int
    tied_embeddings: bool
    quantization: str = "none"
    gradient_checkpointing: bool = False
    lora: bool = False
    attn_implementation: str = "eager"
    num_parameters: int = 0
    num_trainable_parameters: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view for the run manifest."""
        return {
            "name": self.name,
            "device": self.device,
            "dtype": self.dtype,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "tied_embeddings": self.tied_embeddings,
            "quantization": self.quantization,
            "gradient_checkpointing": self.gradient_checkpointing,
            "lora": self.lora,
            "attn_implementation": self.attn_implementation,
            "num_parameters": self.num_parameters,
            "num_trainable_parameters": self.num_trainable_parameters,
        }


class CausalLMBackend(ABC):
    """Abstract causal-LM backend."""

    tokenizer: TokenizerLike
    capabilities: BackendCapabilities

    # -- generation -----------------------------------------------------

    @abstractmethod
    def generate(
        self,
        prefix_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        stop_sequences: Sequence[str] = (),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
    ) -> GenerationOutput:
        """Sample a continuation, stopping on EOS, a stop string, or the budget."""
        ...

    # -- scoring --------------------------------------------------------

    @abstractmethod
    def hidden_states_at(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
        *,
        with_grad: bool = False,
    ) -> torch.Tensor:
        """Return ``[len(positions), hidden_size]`` states at ``positions``.

        ``positions`` are *prediction* positions.  Implementations must run the
        backbone once over the whole sequence and gather, never materialize
        full-sequence logits.
        """
        ...

    @abstractmethod
    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the LM head: ``[N, H] -> [N, V]``."""
        ...

    # -- training -------------------------------------------------------

    @abstractmethod
    def set_train(self, mode: bool) -> None:
        """Switch between train and eval mode."""
        ...

    @abstractmethod
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Parameters the optimizer should own."""
        ...

    @abstractmethod
    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU state dict of the trainable weights only (adapter for QLoRA)."""
        ...

    @abstractmethod
    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore weights previously produced by :meth:`trainable_state_dict`."""
        ...

    # -- placement ------------------------------------------------------

    @abstractmethod
    def to_device(self, device: str) -> None:
        """Move the model to ``device``."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Free accelerator memory held by this backend (``swap`` strategy)."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Current device string."""
        ...

    # -- shared helpers -------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Output vocabulary size."""
        return self.capabilities.vocab_size

    @property
    def hidden_size(self) -> int:
        """Backbone hidden width."""
        return self.capabilities.hidden_size

    def logits_at(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
        *,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Full-vocabulary logits at ``positions``, computed in chunks, no grad.

        Used by the teacher.  The peak vocabulary-sized allocation is
        ``[chunk_size, V]``, never ``[batch, seq_len, V]``.
        """
        import torch as _torch

        with _torch.no_grad():
            hidden = self.hidden_states_at(token_ids, positions, with_grad=False)
            outputs = []
            for start in range(0, hidden.shape[0], chunk_size):
                outputs.append(self.project(hidden[start : start + chunk_size]).to(_torch.float32))
            if not outputs:
                return _torch.zeros((0, self.vocab_size), dtype=_torch.float32)
            return _torch.cat(outputs, dim=0)

    def hidden_states_at_batch(
        self,
        batch: PaddedTrajectoryBatch,
        *,
        with_grad: bool = False,
    ) -> torch.Tensor:
        """Selected states for a typed padded batch.

        The compatibility implementation is sequential. Concrete consumer
        runtimes override it with one masked backbone forward; keeping this
        fallback avoids widening the minimum third-party backend contract.
        """
        import torch as _torch

        parts = []
        for batch_index, length in enumerate(batch.lengths):
            start = batch.selected_offsets[batch_index]
            end = batch.selected_offsets[batch_index + 1]
            token_ids = batch.input_ids[batch_index, :length].tolist()
            positions = batch.selected_positions[start:end].tolist()
            parts.append(self.hidden_states_at(token_ids, positions, with_grad=with_grad))
        if parts:
            return _torch.cat(parts, dim=0)
        return _torch.zeros((0, self.hidden_size), device=self.device)
