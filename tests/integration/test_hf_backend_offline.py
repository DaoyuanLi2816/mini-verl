"""Hugging Face backend against a tiny, locally constructed Qwen3 -- no network.

A real ``Qwen3ForCausalLM`` is built from a config object (no weights are
downloaded) and driven through the exact code path the 16 GB recipe uses:
architecture-adapter resolution through a PEFT wrapper, backbone-only forward,
selected-position projection, KV-cache generation, and a chunked distillation
step whose gradients must reach the LoRA tensors and nothing else.

This is what makes the GPU path testable in CPU CI.
"""

from __future__ import annotations

from pathlib import Path

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


def test_loading_an_unknown_model_id_fails_with_advice(deny_network):
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
    assert "hf download miniverl/definitely-not-a-real-model" in (excinfo.value.hint or "")
    assert deny_network == []


def test_missing_cached_tokenizer_fails_actionably_without_network(monkeypatch, deny_network):
    import transformers

    from miniverl.errors import BackendError
    from miniverl.models.tokenizers import HFTokenizerAdapter

    seen: list[dict[str, object]] = []

    def missing(model_id: str, **kwargs):
        seen.append({"model_id": model_id, **kwargs})
        raise OSError("snapshot is absent")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", missing)
    revision = "d" * 40
    with pytest.raises(BackendError, match="cached tokenizer snapshot") as excinfo:
        HFTokenizerAdapter.load(
            "owner/missing-tokenizer",
            revision=revision,
            local_files_only=True,
        )

    assert seen[0]["local_files_only"] is True
    assert f"hf download owner/missing-tokenizer --revision {revision}" in (
        excinfo.value.hint or ""
    )
    assert deny_network == []


def test_dtype_resolution_matches_the_device():
    from miniverl.config.models import Precision
    from miniverl.models.hf import resolve_dtype

    assert resolve_dtype(Precision.FLOAT32, "cpu") is torch.float32
    assert resolve_dtype(Precision.BFLOAT16, "cpu") is torch.bfloat16
    assert resolve_dtype(Precision.AUTO, "cpu") is torch.float32


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("4.51.3", {"revision": "abc"}),
        ("5.0.0", {"revision": "abc", "local_files_only": True}),
    ],
)
def test_peft_probe_kwargs_preserve_offline_policy_across_transformers_versions(
    version,
    expected,
    monkeypatch,
):
    import miniverl.models.hf as hf_module

    fake_transformers = type("FakeTransformers", (), {"__version__": version})()
    monkeypatch.setattr(hf_module, "require_transformers", lambda _feature: fake_transformers)

    assert (
        hf_module._adapter_probe_kwargs(
            revision="abc",
            local_files_only=True,
        )
        == expected
    )


@pytest.fixture
def local_teacher_adapter(tmp_path: Path, tiny_model, tiny_tokenizer):
    """A standard local PEFT LoRA adapter plus miniVERL provenance."""
    import peft

    from miniverl.cache.store import sha256_file
    from miniverl.models.adapter_io import ADAPTER_MANIFEST
    from miniverl.utils.runs import write_json

    base = tmp_path / "base"
    tiny_model.save_pretrained(base, safe_serialization=True)
    wrapped = peft.get_peft_model(
        tiny_model,
        peft.LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    wrapped.peft_config["default"].base_model_name_or_path = str(base)
    with torch.no_grad():
        for name, parameter in wrapped.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.01)
    adapter = tmp_path / "adapter"
    wrapped.save_pretrained(adapter, safe_serialization=True)
    checksums = {
        name: sha256_file(adapter / name)[0]
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
    write_json(
        adapter / ADAPTER_MANIFEST,
        {
            "schema_version": 1,
            "base_model_id": str(base),
            "base_model_revision": None,
            "tokenizer_fingerprint": tiny_tokenizer.fingerprint,
            "checksums": checksums,
        },
    )
    return base, adapter, wrapped


@requires_peft
def test_hub_teacher_adapter_downloads_and_validates_miniverl_manifest(
    local_teacher_adapter,
    tiny_tokenizer,
    monkeypatch,
):
    import huggingface_hub

    from miniverl.config.models import (
        AdapterSource,
        TeacherAdapterConfig,
        TeacherModelConfig,
    )
    from miniverl.models.adapter_io import ADAPTER_MANIFEST, validate_teacher_adapter
    from miniverl.utils.runs import read_json, write_json

    base, adapter, _ = local_teacher_adapter
    manifest = read_json(adapter / ADAPTER_MANIFEST)
    manifest["policy_evaluation"] = {
        "tag": "final",
        "split": "test",
        "tasks": 8,
        "strict_task_success_rate": 0.75,
        "lenient_diagnostic_success_rate": 0.75,
        "valid_tool_call_rate": 1.0,
        "tool_call_count": 16,
        "final_answer_format_validity_rate": 1.0,
        "avg_turns": 3.0,
        "protocol_token_accuracy": None,
        "policy_competence_measurement_status": {
            "strict_task_success_rate": "measured_primary",
            "protocol_token_accuracy": "not_applicable_free_running",
        },
    }
    write_json(adapter / ADAPTER_MANIFEST, manifest)

    policies: list[bool] = []

    def download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_files_only: bool,
    ) -> str:
        assert repo_id == "owner/protocol-teacher"
        assert revision == "a" * 40
        policies.append(local_files_only)
        return str(adapter / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    provenance = validate_teacher_adapter(
        TeacherAdapterConfig(
            source=AdapterSource.HUB,
            path="owner/protocol-teacher",
            revision="a" * 40,
            require_policy_evaluation=True,
            minimum_strict_success_rate=0.5,
        ),
        TeacherModelConfig(model_id=str(base)),
        tokenizer_fingerprint=tiny_tokenizer.fingerprint,
        local_files_only=True,
    )

    assert provenance["source"] == "hub"
    assert provenance["revision"] == "a" * 40
    assert provenance["weights_sha256"] == manifest["checksums"]["adapter_model.safetensors"]
    assert provenance["policy_evaluation"]["strict_task_success_rate"] == pytest.approx(0.75)
    assert provenance.snapshot_dir == adapter
    assert policies == [True, True, True]

    policies.clear()
    online = validate_teacher_adapter(
        TeacherAdapterConfig(
            source=AdapterSource.HUB,
            path="owner/protocol-teacher",
            revision="a" * 40,
            require_policy_evaluation=True,
            minimum_strict_success_rate=0.5,
        ),
        TeacherModelConfig(model_id=str(base)),
        tokenizer_fingerprint=tiny_tokenizer.fingerprint,
        local_files_only=False,
    )
    assert online.snapshot_dir == provenance.snapshot_dir
    assert policies == [False, False, False]


@requires_peft
def test_cached_hub_adapter_is_loaded_from_the_exact_validated_snapshot_without_network(
    local_teacher_adapter,
    tiny_tokenizer,
    monkeypatch,
    deny_network,
):
    import huggingface_hub
    import peft

    from miniverl.config.models import (
        AdapterSource,
        TeacherAdapterConfig,
        TeacherModelConfig,
    )
    from miniverl.models.hf import HFBackend

    base, adapter, _ = local_teacher_adapter
    revision = "b" * 40
    policies: list[bool] = []

    def download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_files_only: bool,
    ) -> str:
        assert repo_id == "owner/cached-adapter"
        assert revision == "b" * 40
        policies.append(local_files_only)
        return str(adapter / filename)

    original = peft.PeftModel.from_pretrained
    loaded_paths: list[str] = []

    def load_adapter(model, model_id, **kwargs):
        loaded_paths.append(str(model_id))
        assert kwargs["local_files_only"] is True
        return original(model, model_id, **kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", load_adapter)
    backend = HFBackend.load(
        TeacherModelConfig(
            model_id=str(base),
            adapter=TeacherAdapterConfig(
                source=AdapterSource.HUB,
                path="owner/cached-adapter",
                revision=revision,
            ),
        ),
        device="cpu",
        tokenizer=tiny_tokenizer,
        trainable=False,
        local_files_only=True,
    )

    assert backend.adapter_provenance["identity"] == "owner/cached-adapter"
    assert loaded_paths == [str(adapter)]
    assert policies == [True, True, True]
    assert deny_network == []


@requires_peft
def test_missing_cached_hub_adapter_file_is_actionable_and_never_retries_online(
    local_teacher_adapter,
    tiny_tokenizer,
    monkeypatch,
    deny_network,
):
    import huggingface_hub

    from miniverl.config.models import (
        AdapterSource,
        TeacherAdapterConfig,
        TeacherModelConfig,
    )
    from miniverl.errors import BackendError
    from miniverl.models.adapter_io import validate_teacher_adapter

    base, adapter, _ = local_teacher_adapter
    revision = "c" * 40
    requested: list[tuple[str, bool]] = []

    def download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_files_only: bool,
    ) -> str:
        requested.append((filename, local_files_only))
        if filename == "adapter_model.safetensors":
            raise FileNotFoundError("not in cache")
        return str(adapter / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(BackendError, match=r"adapter_model\.safetensors") as excinfo:
        validate_teacher_adapter(
            TeacherAdapterConfig(
                source=AdapterSource.HUB,
                path="owner/partial-adapter",
                revision=revision,
            ),
            TeacherModelConfig(model_id=str(base)),
            tokenizer_fingerprint=tiny_tokenizer.fingerprint,
            local_files_only=True,
        )

    message = str(excinfo.value)
    assert "owner/partial-adapter" in message
    assert revision in message
    assert (
        f"hf download owner/partial-adapter --revision {revision} "
        "--include adapter_config.json adapter_model.safetensors "
        "miniverl_adapter_manifest.json"
    ) in (excinfo.value.hint or "")
    assert requested == [
        ("adapter_config.json", True),
        ("adapter_model.safetensors", True),
    ]
    assert deny_network == []


@requires_peft
def test_fully_missing_cached_hub_adapter_stops_at_first_file_without_network(
    local_teacher_adapter,
    tiny_tokenizer,
    monkeypatch,
    deny_network,
):
    import huggingface_hub

    from miniverl.config.models import (
        AdapterSource,
        TeacherAdapterConfig,
        TeacherModelConfig,
    )
    from miniverl.errors import BackendError
    from miniverl.models.adapter_io import validate_teacher_adapter

    base, _, _ = local_teacher_adapter
    revision = "e" * 40
    requested: list[tuple[str, bool]] = []

    def missing(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_files_only: bool,
    ) -> str:
        requested.append((filename, local_files_only))
        raise FileNotFoundError("snapshot absent")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    with pytest.raises(BackendError, match=r"missing 'adapter_config\.json'") as excinfo:
        validate_teacher_adapter(
            TeacherAdapterConfig(
                source=AdapterSource.HUB,
                path="owner/missing-adapter",
                revision=revision,
            ),
            TeacherModelConfig(model_id=str(base)),
            tokenizer_fingerprint=tiny_tokenizer.fingerprint,
            local_files_only=True,
        )

    assert requested == [("adapter_config.json", True)]
    assert f"owner/missing-adapter --revision {revision}" in (excinfo.value.hint or "")
    assert deny_network == []


@requires_peft
def test_frozen_teacher_adapter_round_trip_preserves_logits(local_teacher_adapter, tiny_tokenizer):
    from miniverl.config.models import TeacherAdapterConfig, TeacherModelConfig
    from miniverl.models.hf import HFBackend

    base, adapter, reference = local_teacher_adapter
    spec = TeacherModelConfig(
        model_id=str(base),
        adapter=TeacherAdapterConfig(path=str(adapter)),
    )
    backend = HFBackend.load(
        spec,
        device="cpu",
        tokenizer=tiny_tokenizer,
        trainable=False,
        local_files_only=True,
    )
    ids = tiny_tokenizer.encode("<|im_start|>assistant\n<final>\n4\n</final>")
    with torch.no_grad():
        expected = reference(torch.tensor([ids])).logits
        actual = backend.model(torch.tensor([ids])).logits
    assert torch.allclose(actual, expected, atol=1e-6)
    assert not any(parameter.requires_grad for parameter in backend.model.parameters())
    assert backend.adapter_provenance["weights_sha256"]
    assert backend.capabilities.lora is True


@requires_peft
def test_teacher_adapter_rejects_wrong_base_and_tokenizer(
    tmp_path: Path,
    local_teacher_adapter,
    tiny_tokenizer,
):
    from miniverl.config.models import TeacherAdapterConfig, TeacherModelConfig
    from miniverl.errors import BackendError
    from miniverl.models.hf import HFBackend

    base, adapter, _ = local_teacher_adapter
    wrong_base = tmp_path / "wrong-base"
    wrong_base.mkdir()
    with pytest.raises(BackendError, match="does not match"):
        HFBackend.load(
            TeacherModelConfig(
                model_id=str(wrong_base),
                adapter=TeacherAdapterConfig(path=str(adapter)),
            ),
            device="cpu",
            tokenizer=tiny_tokenizer,
            trainable=False,
            local_files_only=True,
        )
    with pytest.raises(BackendError, match="tokenizer fingerprint"):
        HFBackend.load(
            TeacherModelConfig(
                model_id=str(base),
                adapter=TeacherAdapterConfig(
                    path=str(adapter),
                    tokenizer_fingerprint="wrong-fingerprint",
                ),
            ),
            device="cpu",
            tokenizer=tiny_tokenizer,
            trainable=False,
            local_files_only=True,
        )


@requires_peft
def test_missing_teacher_adapter_files_are_actionable(tmp_path: Path, tiny_tokenizer):
    from miniverl.config.models import TeacherAdapterConfig, TeacherModelConfig
    from miniverl.errors import BackendError
    from miniverl.models.hf import HFBackend

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BackendError, match=r"missing adapter_model\.safetensors") as excinfo:
        HFBackend.load(
            TeacherModelConfig(
                model_id="unused",
                adapter=TeacherAdapterConfig(path=str(incomplete)),
            ),
            device="cpu",
            tokenizer=tiny_tokenizer,
            trainable=False,
            local_files_only=True,
        )
    assert "standard PEFT adapter" in (excinfo.value.hint or "")


@requires_peft
def test_headline_teacher_gate_requires_recorded_policy_competence(
    local_teacher_adapter,
    tiny_tokenizer,
):
    from miniverl.config.models import TeacherAdapterConfig, TeacherModelConfig
    from miniverl.errors import BackendError
    from miniverl.models.adapter_io import ADAPTER_MANIFEST
    from miniverl.models.hf import HFBackend
    from miniverl.utils.runs import read_json, write_json

    base, adapter, _ = local_teacher_adapter
    with pytest.raises(BackendError, match="no recorded tool-policy evaluation") as excinfo:
        HFBackend.load(
            TeacherModelConfig(
                model_id=str(base),
                adapter=TeacherAdapterConfig(
                    path=str(adapter),
                    require_policy_evaluation=True,
                    minimum_strict_success_rate=0.5,
                ),
            ),
            device="cpu",
            tokenizer=tiny_tokenizer,
            trainable=False,
            local_files_only=True,
        )
    assert "SFT loss is not" in (excinfo.value.hint or "")

    manifest = read_json(adapter / ADAPTER_MANIFEST)
    manifest["policy_evaluation"] = {
        "tag": "final",
        "split": "test",
        "tasks": 8,
        "strict_task_success_rate": 0.75,
        "lenient_diagnostic_success_rate": 0.75,
        "valid_tool_call_rate": 1.0,
        "tool_call_count": 16,
        "final_answer_format_validity_rate": 1.0,
        "avg_turns": 3.0,
        "protocol_token_accuracy": None,
        "policy_competence_measurement_status": {
            "strict_task_success_rate": "measured_primary",
            "protocol_token_accuracy": "not_applicable_free_running",
        },
    }
    write_json(adapter / ADAPTER_MANIFEST, manifest)
    backend = HFBackend.load(
        TeacherModelConfig(
            model_id=str(base),
            adapter=TeacherAdapterConfig(
                path=str(adapter),
                require_policy_evaluation=True,
                minimum_strict_success_rate=0.5,
            ),
        ),
        device="cpu",
        tokenizer=tiny_tokenizer,
        trainable=False,
        local_files_only=True,
    )
    assert backend.adapter_provenance["policy_evaluation"][
        "strict_task_success_rate"
    ] == pytest.approx(0.75)


@requires_peft
def test_miniverl_checkpoint_exports_and_reloads_as_standard_peft(
    tmp_path: Path,
    tiny_model,
    tiny_tokenizer,
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from miniverl.config import RunConfig
    from miniverl.config.models import TeacherAdapterConfig, TeacherModelConfig
    from miniverl.models.adapter_io import export_adapter
    from miniverl.models.hf import HFBackend
    from miniverl.models.tokenizers import HFTokenizerAdapter
    from miniverl.trainer import OPDTrainer

    base = tmp_path / "export-base"
    tiny_model.save_pretrained(base, safe_serialization=True)
    special_ids = {0, tiny_tokenizer.eos_token_id}
    vocab = {
        f"token-{index}": index
        for index in range(tiny_tokenizer.vocab_size)
        if index not in special_ids
    }
    vocab["[UNK]"] = 0
    vocab["[EOS]"] = tiny_tokenizer.eos_token_id
    raw_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    fast_tokenizer.save_pretrained(base)

    config = RunConfig.from_mapping(
        {
            "run": {"name": "adapter-export", "mode": "sft", "output_dir": str(tmp_path)},
            "models": {
                "backend": "hf",
                "device": "cpu",
                "student": {
                    "model_id": str(base),
                    "tokenizer_id": str(base),
                    "dtype": "float32",
                    "lora": {
                        "enabled": True,
                        "r": 4,
                        "alpha": 8,
                        "target_modules": ["q_proj", "v_proj"],
                    },
                },
                "teacher": {
                    "model_id": str(base),
                    "tokenizer_id": str(base),
                    "dtype": "float32",
                },
            },
            "environment": {
                "name": "calculator",
                "train_tasks": 1,
                "eval_tasks": 1,
                "test_tasks": 1,
            },
            "train": {"cycles": 0, "rollouts_per_cycle": 1},
            "eval": {"enabled": False},
            "report": {"enabled": False},
        }
    )
    trainer = OPDTrainer.from_config(
        config,
        output_dir=tmp_path / "runs",
        run_id="adapter-export",
        local_files_only=True,
    )
    try:
        trainer.train()
        ids = [0, 1, 2, 3]
        with torch.no_grad():
            reference = trainer.student.model(torch.tensor([ids])).logits
        manifest, exported = export_adapter(
            trainer.paths.root,
            trainer.paths.checkpoints / "final",
            tmp_path / "exported-adapter",
            local_files_only=True,
        )
    finally:
        trainer.close()

    tokenizer = HFTokenizerAdapter.load(str(base), local_files_only=True)
    loaded = HFBackend.load(
        TeacherModelConfig(
            model_id=str(base),
            tokenizer_id=str(base),
            dtype="float32",
            adapter=TeacherAdapterConfig(path=str(exported)),
        ),
        device="cpu",
        tokenizer=tokenizer,
        trainable=False,
        local_files_only=True,
    )
    with torch.no_grad():
        actual = loaded.model(torch.tensor([ids])).logits
    assert torch.allclose(actual, reference, atol=1e-6)
    assert set(manifest["checksums"]) == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    assert (exported / "miniverl_adapter_manifest.json").is_file()
    assert manifest["policy_evaluation"] is None

    opd_mapping = config.model_dump(mode="json")
    opd_mapping["run"].update({"name": "adapter-provenance", "mode": "opd"})
    opd_mapping["models"]["teacher"]["adapter"] = {"path": str(exported)}
    opd_mapping["loss"]["sampled_token_nll_weight"] = 0.0
    opd_config = RunConfig.from_mapping(opd_mapping)
    opd_trainer = OPDTrainer.from_config(
        opd_config,
        output_dir=tmp_path / "runs",
        run_id="adapter-provenance",
        local_files_only=True,
    )
    try:
        run_manifest = opd_trainer.build_manifest()
    finally:
        opd_trainer.close()
    adapter_provenance = run_manifest["models"]["teacher"]["adapter"]
    assert adapter_provenance["identity"] == exported.name
    assert (
        adapter_provenance["weights_sha256"] == manifest["checksums"]["adapter_model.safetensors"]
    )
