"""Architecture adapter for Hugging Face causal LMs.

Three things have to be reachable for miniVERL's selected-position path to
work, and every wrapper (PEFT, quantization, ``torch.compile``) hides them
somewhere slightly different:

1. the **decoder backbone**, so hidden states can be produced *without* the LM
   head running over the whole sequence;
2. the **LM head**, so only the selected positions get projected;
3. the **unwrapped base model**, so config and generation metadata are readable.

Rather than pattern-matching on class names, the adapter probes the documented
Transformers APIs (``get_decoder``, ``get_output_embeddings``,
``get_base_model``) and falls back to a small list of well-known attribute
paths.  If none of them work it raises with the model class name instead of
guessing, because a wrong guess here would silently train on the wrong tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from miniverl.errors import BackendError

__all__ = ["ArchitectureAdapter", "TESTED_ARCHITECTURES"]

#: Architectures miniVERL has an executed test for.  Anything else works only
#: if the probes below succeed, and the CLI says so out loud.
TESTED_ARCHITECTURES: tuple[str, ...] = ("Qwen3ForCausalLM", "Qwen2ForCausalLM")

_BACKBONE_PATHS = ("model", "transformer", "gpt_neox", "decoder")


@dataclass
class ArchitectureAdapter:
    """Resolved handles into one loaded causal LM."""

    model: Any
    base_model: Any
    backbone: Any
    lm_head: Any
    architecture: str
    tied_embeddings: bool
    hidden_size: int
    vocab_size: int

    @property
    def is_tested_architecture(self) -> bool:
        """``True`` when this architecture has an executed miniVERL test."""
        return self.architecture in TESTED_ARCHITECTURES

    @classmethod
    def resolve(cls, model: Any) -> ArchitectureAdapter:
        """Locate the backbone and LM head of ``model``."""
        base = _unwrap_peft(model)
        backbone = _find_backbone(base)
        lm_head = _find_lm_head(base)
        config = getattr(base, "config", None)
        if config is None:
            raise BackendError(f"{type(base).__name__} has no .config")
        architectures = list(getattr(config, "architectures", None) or [])
        architecture = architectures[0] if architectures else type(base).__name__
        hidden_size = int(getattr(config, "hidden_size", None) or getattr(config, "n_embd", 0) or 0)
        if hidden_size <= 0:
            raise BackendError(f"could not determine hidden_size for {architecture}")
        vocab_size = int(getattr(lm_head, "out_features", 0) or getattr(config, "vocab_size", 0))
        if vocab_size <= 0:
            raise BackendError(f"could not determine vocab_size for {architecture}")
        return cls(
            model=model,
            base_model=base,
            backbone=backbone,
            lm_head=lm_head,
            architecture=architecture,
            tied_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )


def _unwrap_peft(model: Any) -> Any:
    """Return the underlying transformers model behind any PEFT wrappers."""
    current = model
    for _ in range(4):
        getter = getattr(current, "get_base_model", None)
        if callable(getter):
            candidate = getter()
            if candidate is not current:
                current = candidate
                continue
        inner = getattr(current, "base_model", None)
        if inner is not None and inner is not current and hasattr(inner, "model"):
            current = inner.model
            continue
        break
    return current


def _find_backbone(base: Any) -> Any:
    """Locate the decoder stack that produces hidden states."""
    getter = getattr(base, "get_decoder", None)
    if callable(getter):
        try:
            decoder = getter()
        except (AttributeError, NotImplementedError):  # pragma: no cover - old models
            decoder = None
        if decoder is not None and decoder is not base:
            return decoder
    for name in _BACKBONE_PATHS:
        candidate = getattr(base, name, None)
        if candidate is not None and hasattr(candidate, "forward"):
            return candidate
    raise BackendError(
        f"could not locate the decoder backbone of {type(base).__name__}",
        hint="miniVERL needs a model exposing get_decoder() or a .model/.transformer "
        f"attribute; tested architectures: {', '.join(TESTED_ARCHITECTURES)}",
    )


def _find_lm_head(base: Any) -> Any:
    """Locate the output projection."""
    getter = getattr(base, "get_output_embeddings", None)
    if callable(getter):
        head = getter()
        if head is not None:
            return head
    head = getattr(base, "lm_head", None)
    if head is not None:
        return head
    raise BackendError(
        f"could not locate the LM head of {type(base).__name__}",
        hint="miniVERL needs get_output_embeddings() or an .lm_head attribute",
    )
