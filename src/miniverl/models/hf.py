"""Hugging Face causal-LM backend.

Key property: the LM head is **never** run over a whole sequence.  Both
scoring and training call the decoder backbone directly to obtain
``[1, T, H]`` hidden states, gather the selected positions into ``[N, H]``, and
project only those.  Even generation projects a single position per step.  For
a 152k-vocabulary model at sequence length 768 this avoids a 445 MB fp32 logit
tensor per forward pass, which is the difference between fitting on a 16 GB
card and not.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import torch

from miniverl.agent.transcript import TokenizerLike
from miniverl.config.models import (
    LoRAConfig,
    Precision,
    Quantization,
    StudentModelConfig,
    TeacherModelConfig,
)
from miniverl.errors import BackendError, MissingDependencyError
from miniverl.models.adapters import ArchitectureAdapter
from miniverl.models.base import BackendCapabilities, CausalLMBackend, GenerationOutput
from miniverl.models.sampling import run_generation
from miniverl.utils.lazy import have_module, require_peft, require_transformers

__all__ = ["HFBackend", "resolve_dtype", "supports_bfloat16"]


def supports_bfloat16(device: str) -> bool:
    """``True`` when bf16 is usable on ``device``."""
    if device.startswith("cuda"):
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    return hasattr(torch, "bfloat16")


def resolve_dtype(precision: Precision, device: str) -> torch.dtype:
    """Turn ``auto`` into a concrete dtype for ``device``."""
    if precision is Precision.FLOAT32:
        return torch.float32
    if precision is Precision.BFLOAT16:
        return torch.bfloat16
    if precision is Precision.FLOAT16:
        return torch.float16
    if device.startswith("cuda"):
        return torch.bfloat16 if supports_bfloat16(device) else torch.float16
    return torch.float32


def _quantization_config(quantization: Quantization, compute_dtype: torch.dtype) -> Any:
    if quantization is Quantization.NONE:
        return None
    if not have_module("bitsandbytes"):
        raise MissingDependencyError(
            "bitsandbytes", "cuda", f"{quantization.value} weight quantization"
        )
    transformers = require_transformers("Quantized model loading")
    if quantization is Quantization.NF4:
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return transformers.BitsAndBytesConfig(load_in_8bit=True)


def _dtype_kwarg_name() -> str:
    """Name of the dtype keyword this transformers version wants.

    ``torch_dtype`` was renamed to ``dtype`` in transformers 4.56 and the old
    name warns loudly on 5.x.  The signature is often ``**kwargs``, so version
    comparison is more reliable than introspection; introspection is kept as a
    fallback for forks that do declare the parameter.
    """
    transformers = require_transformers("Model loading")
    version = str(getattr(transformers, "__version__", "0"))
    try:
        parts = [int(p) for p in version.split(".")[:2]]
        major, minor = [*parts, 0, 0][:2]
    except ValueError:  # pragma: no cover - unusual version strings
        major, minor = 0, 0
    if (major, minor) >= (4, 56):
        return "dtype"
    signature = inspect.signature(transformers.AutoModelForCausalLM.from_pretrained)
    return "dtype" if "dtype" in signature.parameters else "torch_dtype"


def _from_pretrained_kwargs(dtype: torch.dtype) -> dict[str, Any]:
    """Build the dtype keyword argument for ``from_pretrained``."""
    return {_dtype_kwarg_name(): dtype}


class HFBackend(CausalLMBackend):
    """Local Hugging Face causal LM, optionally quantized and LoRA-adapted."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: TokenizerLike,
        model_id: str,
        model_revision: str | None,
        device: str,
        dtype: torch.dtype,
        quantization: Quantization,
        gradient_checkpointing: bool,
        attn_implementation: str,
        lora: bool,
        adapter_provenance: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_revision = model_revision
        self._device = device
        self._dtype = dtype
        self.adapter_provenance = adapter_provenance
        self.adapter = ArchitectureAdapter.resolve(model)
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.capabilities = BackendCapabilities(
            name=model_id,
            device=device,
            dtype=str(dtype).replace("torch.", ""),
            vocab_size=self.adapter.vocab_size,
            hidden_size=self.adapter.hidden_size,
            tied_embeddings=self.adapter.tied_embeddings,
            quantization=quantization.value,
            gradient_checkpointing=gradient_checkpointing,
            lora=lora,
            attn_implementation=attn_implementation,
            num_parameters=num_params,
            num_trainable_parameters=num_trainable,
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def load(
        cls,
        spec: StudentModelConfig | TeacherModelConfig,
        *,
        device: str,
        tokenizer: TokenizerLike,
        trainable: bool,
        local_files_only: bool = False,
    ) -> HFBackend:
        """Load a model from a local path or the Hub."""
        transformers = require_transformers("Loading a Hugging Face causal LM")
        dtype = resolve_dtype(spec.dtype, device)
        quantization = spec.quantization
        adapter_provenance = None
        validated_adapter = None
        if isinstance(spec, TeacherModelConfig) and spec.adapter is not None:
            from miniverl.models.adapter_io import validate_teacher_adapter

            validated_adapter = validate_teacher_adapter(
                spec.adapter,
                spec,
                tokenizer_fingerprint=tokenizer.fingerprint,
                local_files_only=local_files_only,
            )
            adapter_provenance = validated_adapter.provenance
        kwargs: dict[str, Any] = {
            "revision": spec.revision,
            "trust_remote_code": spec.trust_remote_code,
            "local_files_only": local_files_only,
            # Transformers performs a separate PEFT-config probe before loading
            # the base model. In 5.x that probe does not inherit hub kwargs, so
            # pass the policy explicitly instead of relying on global offline
            # environment variables.
            "adapter_kwargs": {
                "local_files_only": local_files_only,
                "revision": spec.revision,
            },
            "attn_implementation": spec.attn_implementation,
            **_from_pretrained_kwargs(dtype),
        }
        quant_config = _quantization_config(quantization, dtype)
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
            kwargs["device_map"] = {"": device}
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
        except OSError as exc:
            revision = f" at revision {spec.revision!r}" if spec.revision else ""
            preload = f"hf download {spec.model_id}"
            if spec.revision:
                preload += f" --revision {spec.revision}"
            offline_hint = (
                f"offline mode found no complete cached model snapshot; preload it online "
                f"with `{preload}`. "
                if local_files_only
                else ""
            )
            raise BackendError(
                f"could not load model {spec.model_id!r}{revision}",
                hint=(
                    offline_hint
                    + "check the model id and revision and that any gated license has been "
                    f"accepted on the Hub. Original error: {exc}"
                ),
            ) from exc

        if quant_config is None:
            model.to(device)

        lora_enabled = False
        gradient_checkpointing = getattr(spec, "gradient_checkpointing", False)
        if trainable:
            lora_spec = getattr(spec, "lora", None)
            if isinstance(lora_spec, LoRAConfig) and lora_spec.enabled:
                model = cls._attach_lora(
                    model,
                    lora_spec,
                    quantized=quant_config is not None,
                    gradient_checkpointing=gradient_checkpointing,
                )
                lora_enabled = True
            if gradient_checkpointing:
                model.gradient_checkpointing_enable()
                enable_grads = getattr(model, "enable_input_require_grads", None)
                if callable(enable_grads):
                    enable_grads()
            model.train()
        else:
            if isinstance(spec, TeacherModelConfig) and spec.adapter is not None:
                assert validated_adapter is not None
                peft = require_peft("Loading a frozen teacher adapter")
                try:
                    model = peft.PeftModel.from_pretrained(
                        model,
                        str(validated_adapter.snapshot_dir),
                        is_trainable=False,
                        local_files_only=True,
                    )
                except (OSError, ValueError) as exc:
                    raise BackendError(
                        f"could not attach teacher adapter {spec.adapter.path!r}: {exc}",
                        hint="verify the adapter/base pair and its PEFT target modules",
                    ) from exc
                lora_enabled = True
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            if any(param.requires_grad for param in model.parameters()):
                raise BackendError("teacher adapter load left trainable teacher parameters")

        if getattr(model, "config", None) is not None:
            model.config.use_cache = not (trainable and gradient_checkpointing)

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=spec.model_id,
            model_revision=spec.revision,
            device=device,
            dtype=dtype,
            quantization=quantization,
            gradient_checkpointing=bool(gradient_checkpointing),
            attn_implementation=spec.attn_implementation,
            lora=lora_enabled,
            adapter_provenance=adapter_provenance,
        )

    @staticmethod
    def _attach_lora(
        model: Any, spec: LoRAConfig, *, quantized: bool, gradient_checkpointing: bool
    ) -> Any:
        peft = require_peft("LoRA / QLoRA training")
        if quantized:
            model = peft.prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=gradient_checkpointing
            )
        config = peft.LoraConfig(
            r=spec.r,
            lora_alpha=spec.alpha,
            lora_dropout=spec.dropout,
            bias=spec.bias,
            target_modules=list(spec.target_modules),
            task_type="CAUSAL_LM",
        )
        try:
            return peft.get_peft_model(model, config)
        except ValueError as exc:
            raise BackendError(
                f"could not attach LoRA adapters: {exc}",
                hint="check models.student.lora.target_modules against the model's "
                "module names (for Qwen: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj)",
            ) from exc

    # -- forward helpers ----------------------------------------------------

    def _backbone_forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        backbone = self.adapter.backbone
        kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": use_cache}
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        outputs = backbone(**kwargs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        return hidden, getattr(outputs, "past_key_values", None)

    # -- generation ---------------------------------------------------------

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
        """Sample a continuation, projecting one position per step."""
        was_training = self.model.training
        self.model.eval()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed if seed is not None else 0)
        cache_flag = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True

        def step(new_ids: list[int], state: Any) -> tuple[torch.Tensor, Any]:
            ids = torch.tensor([new_ids], dtype=torch.long, device=self._device)
            with torch.no_grad():
                hidden, present = self._backbone_forward(ids, past_key_values=state, use_cache=True)
                logits = self.adapter.lm_head(hidden[:, -1, :])
            return logits[0].float(), present

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
            self.model.config.use_cache = cache_flag
            if was_training:
                self.model.train()

    # -- scoring ------------------------------------------------------------

    def hidden_states_at(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
        *,
        with_grad: bool = False,
    ) -> torch.Tensor:
        """Backbone hidden states gathered at ``positions``; no full-sequence logits."""
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self._device)
        index = torch.tensor(list(positions), dtype=torch.long, device=self._device)
        context = torch.enable_grad() if with_grad else torch.no_grad()
        with context:
            hidden, _ = self._backbone_forward(ids, use_cache=False)
            return hidden[0].index_select(0, index)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the LM head to selected hidden states."""
        head = self.adapter.lm_head
        weight_dtype = getattr(head, "weight", None)
        if weight_dtype is not None and hidden.dtype != weight_dtype.dtype:
            hidden = hidden.to(weight_dtype.dtype)
        return head(hidden)

    # -- training -----------------------------------------------------------

    def set_train(self, mode: bool) -> None:
        """Switch train/eval mode."""
        self.model.train(mode)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Parameters requiring gradients (the LoRA adapters under QLoRA)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU copy of the trainable weights only."""
        return {
            name: param.detach().to("cpu").clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore trainable weights in place."""
        own = dict(self.model.named_parameters())
        missing = [k for k in state if k not in own]
        if missing:
            raise BackendError(
                f"checkpoint contains {len(missing)} unknown parameter names, e.g. {missing[0]}",
                hint="the checkpoint was written by a different model or LoRA config",
            )
        with torch.no_grad():
            for name, value in state.items():
                own[name].copy_(value.to(own[name].device, own[name].dtype))

    # -- placement -----------------------------------------------------------

    def to_device(self, device: str) -> None:
        """Move the model, unless it is bitsandbytes-quantized (pinned at load)."""
        if self.capabilities.quantization != "none" and device != self._device:
            raise BackendError(
                "a bitsandbytes-quantized model cannot be moved between devices after loading",
                hint="use memory.strategy: resident for quantized models, or load the "
                "model unquantized",
            )
        self.model.to(device)
        self._device = device
        self.capabilities.device = device

    def release(self) -> None:
        """Move to host memory and free the CUDA caching allocator's blocks."""
        from miniverl.utils.gpu import empty_cache

        if self.capabilities.quantization == "none":
            self.model.to("cpu")
            self._device = "cpu"
            self.capabilities.device = "cpu"
        empty_cache()

    @property
    def device(self) -> str:
        """Current device."""
        return self._device
