"""Hugging Face backend against a tiny, locally constructed Qwen3 -- no network.

A real ``Qwen3ForCausalLM`` is built from a config object (no weights are
downloaded) and driven through the exact code path the 16 GB recipe uses:
architecture-adapter resolution through a PEFT wrapper, backbone-only forward,
selected-position projection, KV-cache generation, and a chunked distillation
step whose gradients must reach the LoRA tensors and nothing else.

This is what makes the GPU path testable in CPU CI.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_peft, requires_transformers

pytestmark = [requires_transformers, pytest.mark.torch]

torch = pytest.importorskip("torch")


@pytest.fixture
def tiny_tokenizer():
    """The toy tokenizer doubles as a tiny offline tokenizer for the HF backend."""
    from miniverl.models.tokenizers import ToyTokenizer

    return ToyTokenizer()


@pytest.fixture
def tiny_model(tiny_tokenizer):
    """A randomly initialized Qwen3 with the real architecture, ~30k parameters."""
    from transformers import AutoModelForCausalLM, Qwen3Config

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=tiny_tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )
    return AutoModelForCausalLM.from_config(config)


def _backend(model, tokenizer, *, lora: bool = False):
    from miniverl.config.models import Quantization
    from miniverl.models.hf import HFBackend

    return HFBackend(
        model=model,
        tokenizer=tokenizer,
        model_id="tiny-qwen3",
        model_revision="local-config",
        device="cpu",
        dtype=torch.float32,
        quantization=Quantization.NONE,
        gradient_checkpointing=False,
        attn_implementation="sdpa",
        lora=lora,
    )


def test_architecture_adapter_resolves_a_real_qwen3(tiny_model):
    from miniverl.models.adapters import TESTED_ARCHITECTURES, ArchitectureAdapter

    adapter = ArchitectureAdapter.resolve(tiny_model)
    assert adapter.architecture == "Qwen3ForCausalLM"
    assert adapter.architecture in TESTED_ARCHITECTURES
    assert adapter.is_tested_architecture
    assert type(adapter.backbone).__name__ == "Qwen3Model"
    assert adapter.hidden_size == 32
    assert adapter.vocab_size == tiny_model.config.vocab_size
    assert adapter.tied_embeddings is False


@requires_peft
def test_architecture_adapter_sees_through_a_peft_wrapper(tiny_model):
    import peft

    from miniverl.models.adapters import ArchitectureAdapter

    wrapped = peft.get_peft_model(
        tiny_model,
        peft.LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        ),
    )
    adapter = ArchitectureAdapter.resolve(wrapped)
    assert adapter.architecture == "Qwen3ForCausalLM"
    assert type(adapter.backbone).__name__ == "Qwen3Model"
    assert adapter.lm_head is not None


def test_adapter_rejects_a_model_it_cannot_understand():
    from miniverl.errors import BackendError
    from miniverl.models.adapters import ArchitectureAdapter

    class Opaque(torch.nn.Module):
        pass

    with pytest.raises(BackendError, match="decoder backbone"):
        ArchitectureAdapter.resolve(Opaque())


def test_selected_position_projection_avoids_full_sequence_logits(tiny_model, tiny_tokenizer):
    backend = _backend(tiny_model, tiny_tokenizer)
    ids = tiny_tokenizer.encode("<|im_start|>user\nCompute 1 + 1.<|im_end|>\n")
    positions = [1, 3, len(ids) - 2]
    hidden = backend.hidden_states_at(ids, positions, with_grad=False)
    assert hidden.shape == (len(positions), 32)
    logits = backend.project(hidden)
    assert logits.shape == (len(positions), tiny_tokenizer.vocab_size)

    # The same values must come out of a full forward, so the shortcut is exact.
    with torch.no_grad():
        reference = tiny_model(torch.tensor([ids])).logits[0]
    for i, position in enumerate(positions):
        assert torch.allclose(logits[i], reference[position], atol=1e-4)


def test_generation_uses_the_kv_cache_and_honours_stop_strings(tiny_model, tiny_tokenizer):
    backend = _backend(tiny_model, tiny_tokenizer)
    prefix = tiny_tokenizer.encode("<|im_start|>assistant\n")
    output = backend.generate(prefix, max_new_tokens=16, temperature=1.0, seed=3)
    assert 1 <= len(output.token_ids) <= 16
    assert tiny_tokenizer.decode(output.token_ids) == output.text
    assert output.stop_reason in {"max_new_tokens", "eos", "stop_sequence"}

    # Same seed, same continuation.
    again = backend.generate(prefix, max_new_tokens=16, temperature=1.0, seed=3)
    assert again.token_ids == output.token_ids
    # Greedy is deterministic regardless of the seed.
    a = backend.generate(prefix, max_new_tokens=8, temperature=0.0, seed=1)
    b = backend.generate(prefix, max_new_tokens=8, temperature=0.0, seed=99)
    assert a.token_ids == b.token_ids


def test_generation_with_a_cache_matches_a_cacheless_reference(tiny_model, tiny_tokenizer):
    """If the KV cache were not wired up, greedy decoding would diverge."""
    backend = _backend(tiny_model, tiny_tokenizer)
    prefix = tiny_tokenizer.encode("<|im_start|>assistant\n")
    cached = backend.generate(prefix, max_new_tokens=6, temperature=0.0)

    manual: list[int] = []
    context = list(prefix)
    with torch.no_grad():
        for _ in range(6):
            logits = tiny_model(torch.tensor([context])).logits[0, -1]
            token = int(logits.argmax())
            manual.append(token)
            context.append(token)
            if token == tiny_tokenizer.eos_token_id:
                break
    assert cached.token_ids == manual


@requires_peft
def test_qlora_style_training_step_updates_only_the_adapters(tiny_model, tiny_tokenizer):
    import peft

    from miniverl.losses.bucketed import teacher_topk_targets
    from miniverl.losses.chunked import BucketedTargetProvider, chunked_selected_position_loss

    wrapped = peft.get_peft_model(
        tiny_model,
        peft.LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    backend = _backend(wrapped, tiny_tokenizer, lora=True)
    trainable = backend.trainable_parameters()
    assert trainable, "LoRA must expose trainable parameters"
    frozen = [p for p in wrapped.parameters() if not p.requires_grad]
    assert frozen, "the base weights must stay frozen"

    ids = tiny_tokenizer.encode("<|im_start|>assistant\n<final>\n4\n</final>")
    positions = list(range(2, min(len(ids), 12)))
    hidden = backend.hidden_states_at(ids, positions, with_grad=True)
    teacher_logits = torch.randn(
        len(positions), tiny_tokenizer.vocab_size, generator=torch.Generator().manual_seed(5)
    )
    idx, lp, tail = teacher_topk_targets(teacher_logits, top_k=8)
    output = chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=backend.project,
        weights=torch.ones(len(positions)),
        provider=BucketedTargetProvider(topk_indices=idx, topk_log_probs=lp, tail_log_prob=tail),
        chunk_size=4,
        backward=True,
    )
    assert output.loss > 0.0
    assert all(p.grad is not None for p in trainable)
    assert any(float(p.grad.abs().sum()) > 0 for p in trainable)
    assert all(p.grad is None for p in frozen), "a frozen base weight received a gradient"

    state = backend.trainable_state_dict()
    assert state and all("lora" in name.lower() for name in state)


def test_the_backend_reports_honest_capabilities(tiny_model, tiny_tokenizer):
    backend = _backend(tiny_model, tiny_tokenizer)
    caps = backend.capabilities.to_dict()
    assert caps["vocab_size"] == tiny_tokenizer.vocab_size
    assert caps["hidden_size"] == 32
    assert caps["quantization"] == "none"
    assert caps["device"] == "cpu"
    assert caps["num_parameters"] > 0


def test_loading_an_unknown_model_id_fails_with_advice():
    from miniverl.config.models import StudentModelConfig
    from miniverl.errors import BackendError
    from miniverl.models.hf import HFBackend
    from miniverl.models.tokenizers import ToyTokenizer

    spec = StudentModelConfig(model_id="miniverl/definitely-not-a-real-model")
    with pytest.raises(BackendError, match="could not load model") as excinfo:
        HFBackend.load(
            spec, device="cpu", tokenizer=ToyTokenizer(), trainable=False, local_files_only=True
        )
    assert "revision" in (excinfo.value.hint or "")


def test_dtype_resolution_matches_the_device():
    from miniverl.config.models import Precision
    from miniverl.models.hf import resolve_dtype

    assert resolve_dtype(Precision.FLOAT32, "cpu") is torch.float32
    assert resolve_dtype(Precision.BFLOAT16, "cpu") is torch.bfloat16
    assert resolve_dtype(Precision.AUTO, "cpu") is torch.float32
