"""A tiny, fully local transformer backend.

This is not a mock.  It is a real decoder-only transformer -- RMSNorm,
multi-head causal attention with a KV cache, SwiGLU MLP, untied LM head --
just small enough (~100k parameters, ~200-entry vocabulary) that the whole
SFT / offline-KD / OPD pipeline runs on a CPU in seconds and that
``exact_full_vocab`` losses are affordable.

Everything the real Hugging Face path exercises is exercised here too:
selected-position hidden states, chunked LM-head projection, gradient
accumulation, checkpoint/resume, and both memory strategies.  That is what lets
the CPU test suite assert behaviour the GPU path relies on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn

from miniverl.errors import BackendError
from miniverl.models.base import BackendCapabilities, CausalLMBackend, GenerationOutput
from miniverl.models.sampling import run_generation
from miniverl.models.tokenizers import ToyTokenizer

__all__ = ["ToyCausalLM", "ToyBackend", "fit_toy_model"]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension."""
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the halves of the last dimension, as used by RoPE."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to ``q`` and ``k`` (``[B, H, T, D]``)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class ToyAttention(nn.Module):
    """Causal multi-head self-attention with RoPE and an optional KV cache.

    Rotary embeddings rather than learned absolute positions: relative position
    information is what makes copy/induction behaviour learnable at this scale.
    With absolute embeddings the toy model reliably learns the tool-call
    *syntax* and then fails to copy the operands -- measured, not assumed (see
    ``docs/adr/0007-toy-backend-rope.md``).
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend over ``x`` plus any cached keys/values."""
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, k, cos, sin)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        is_causal = past is None and t > 1
        attn_mask = None
        if not is_causal and t > 1:
            total = k.shape[2]
            offset = total - t
            attn_mask = torch.ones(t, total, dtype=torch.bool, device=x.device).tril(
                diagonal=offset
            )
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )
        out = out.transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out), (k, v)


class ToyMLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated MLP."""
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = ToyAttention(hidden_size, num_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = ToyMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One residual block."""
        attn_out, present = self.self_attn(self.input_layernorm(x), past, cos, sin)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


class ToyCausalLM(nn.Module):
    """A small decoder-only language model: RMSNorm + RoPE attention + SwiGLU.

    Structurally the same family as Qwen/Llama, three orders of magnitude
    smaller.  The LM head is deliberately untied so that both the tied and
    untied projection paths are exercised somewhere in the test suite.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: int = 128,
        max_position_embeddings: int = 1024,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.head_dim = hidden_size // num_heads
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [ToyBlock(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        cos, sin = self._rope_tables(max_position_embeddings, self.head_dim, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _rope_tables(
        max_positions: int, head_dim: int, theta: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_positions, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Return final hidden states ``[B, T, H]`` and the updated cache."""
        _, t = input_ids.shape
        offset = 0 if past_key_values is None else int(past_key_values[0][0].shape[2])
        if offset + t > self.max_position_embeddings:
            raise BackendError(
                f"sequence length {offset + t} exceeds the toy model's "
                f"max_position_embeddings={self.max_position_embeddings}",
                hint="lower rollout.max_total_tokens or raise "
                "models.<role>.toy.max_position_embeddings",
            )
        dtype = self.embed_tokens.weight.dtype
        rope_cos = cast("torch.Tensor", self.rope_cos)
        rope_sin = cast("torch.Tensor", self.rope_sin)
        cos = rope_cos[offset : offset + t].to(dtype=dtype)
        sin = rope_sin[offset : offset + t].to(dtype=dtype)
        x = self.embed_tokens(input_ids)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = layer(x, past, cos, sin)
            if use_cache:
                presents.append(present)
        x = self.norm(x)
        return x, (presents if use_cache else None)


class ToyBackend(CausalLMBackend):
    """:class:`~miniverl.models.base.CausalLMBackend` over :class:`ToyCausalLM`."""

    def __init__(
        self,
        *,
        tokenizer: ToyTokenizer,
        model_id: str,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: int = 128,
        max_position_embeddings: int = 1024,
        seed: int = 0,
        device: str = "cpu",
        trainable: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_revision: str | None = None
        self._device = device
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
            self.model = ToyCausalLM(
                vocab_size=tokenizer.vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
            )
        self.model.to(device)
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()
        num_params = sum(p.numel() for p in self.model.parameters())
        self.capabilities = BackendCapabilities(
            name=model_id,
            device=device,
            dtype="float32",
            vocab_size=tokenizer.vocab_size,
            hidden_size=hidden_size,
            tied_embeddings=False,
            quantization="none",
            gradient_checkpointing=False,
            lora=False,
            attn_implementation="sdpa",
            num_parameters=num_params,
            num_trainable_parameters=sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        )

    # -- generation -----------------------------------------------------

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
        """Sample a continuation with a KV cache."""
        was_training = self.model.training
        self.model.eval()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed if seed is not None else 0)

        def step(new_ids: list[int], state: Any) -> tuple[torch.Tensor, Any]:
            ids = torch.tensor([new_ids], dtype=torch.long, device=self._device)
            with torch.no_grad():
                hidden, present = self.model(ids, past_key_values=state, use_cache=True)
                logits = self.model.lm_head(hidden[:, -1, :])
            return logits[0], present

        try:
            return run_generation(
                step=step,
                prefix_token_ids=prefix_token_ids,
                decode=self.tokenizer.decode,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                stop_sequences=stop_sequences,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
            )
        finally:
            if was_training:
                self.model.train()

    # -- scoring --------------------------------------------------------

    def hidden_states_at(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
        *,
        with_grad: bool = False,
    ) -> torch.Tensor:
        """Gather hidden states at ``positions`` after a single forward pass."""
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self._device)
        index = torch.tensor(list(positions), dtype=torch.long, device=self._device)
        context = torch.enable_grad() if with_grad else torch.no_grad()
        with context:
            hidden, _ = self.model(ids, past_key_values=None, use_cache=False)
            return hidden[0].index_select(0, index)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the LM head."""
        return self.model.lm_head(hidden)

    # -- training -------------------------------------------------------

    def set_train(self, mode: bool) -> None:
        """Switch train/eval mode."""
        self.model.train(mode)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """All parameters that require grad."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU copy of every trainable parameter."""
        return {
            name: param.detach().to("cpu").clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore trainable parameters in place."""
        own = dict(self.model.named_parameters())
        missing = [k for k in state if k not in own]
        if missing:
            raise BackendError(f"unknown parameter names in state dict: {missing[:5]}")
        with torch.no_grad():
            for name, value in state.items():
                own[name].copy_(value.to(own[name].device, own[name].dtype))

    def full_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU copy of the whole model (toy models are tiny enough to store)."""
        return {k: v.detach().to("cpu").clone() for k, v in self.model.state_dict().items()}

    def load_full_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load a full state dict produced by :meth:`full_state_dict`."""
        self.model.load_state_dict(state)

    # -- placement ------------------------------------------------------

    def to_device(self, device: str) -> None:
        """Move the toy model."""
        self.model.to(device)
        self._device = device
        self.capabilities.device = device

    def release(self) -> None:
        """Move to CPU; the toy model is too small to warrant anything else."""
        self.to_device("cpu")

    @property
    def device(self) -> str:
        """Current device."""
        return self._device


def fit_toy_model(
    backend: ToyBackend,
    batches: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    steps: int,
    lr: float = 3e-3,
    seed: int = 0,
    chunk_size: int = 512,
    batch_size: int = 4,
    warmup_fraction: float = 0.05,
) -> list[float]:
    """Fit a toy model with masked next-token cross-entropy.

    Used to give the toy *teacher* real competence before it supervises
    anything: a randomly initialized teacher is a uniform-noise oracle, and
    distilling from it would demonstrate nothing.  Each batch entry is
    ``(token_ids, target_positions)`` and only the listed positions contribute,
    so tool output is excluded here exactly as it is during training.

    Returns the per-step loss values so the caller can log them and assert that
    the fit actually converged.
    """

    from miniverl.losses.chunked import chunked_selected_position_loss

    if not batches or steps <= 0:
        return []
    torch.manual_seed(seed)
    backend.set_train(True)
    params = backend.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95))
    warmup = max(1, int(steps * warmup_fraction))
    losses: list[float] = []
    cursor = 0

    for step_index in range(steps):
        group = [batches[(cursor + i) % len(batches)] for i in range(batch_size)]
        cursor += batch_size
        scale = 1.0 / len(group)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for token_ids, target_positions in group:
            targets = list(target_positions)
            if not targets:
                continue
            hidden = backend.hidden_states_at(token_ids, [j - 1 for j in targets], with_grad=True)
            target_ids = torch.tensor(
                [token_ids[j] for j in targets], dtype=torch.long, device=hidden.device
            )
            output = chunked_selected_position_loss(
                hidden_states=hidden,
                lm_head=backend.project,
                weights=torch.ones(len(targets), dtype=torch.float32, device=hidden.device),
                provider=None,
                target_token_ids=target_ids,
                ce_weight=1.0,
                chunk_size=chunk_size,
                backward=True,
                loss_scale=scale,
            )
            step_loss += output.loss * scale
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if step_index < warmup:
            current_lr = lr * (step_index + 1) / warmup
        else:
            progress = (step_index - warmup) / max(steps - warmup, 1)
            current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group_params in optimizer.param_groups:
            group_params["lr"] = current_lr
        optimizer.step()
        losses.append(step_loss)

    optimizer.zero_grad(set_to_none=True)
    backend.set_train(False)
    return losses
