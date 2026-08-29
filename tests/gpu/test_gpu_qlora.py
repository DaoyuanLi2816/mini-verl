"""GPU tests, deselected automatically when no CUDA device is present.

Run them explicitly with::

    pytest -m gpu

They use a locally constructed tiny Qwen3 rather than a downloaded checkpoint,
so they exercise the CUDA/bitsandbytes code paths without needing the network.
The one test that does need the pinned public pair is additionally marked
``network``.
"""

from __future__ import annotations

import importlib.util

import pytest

from tests.conftest import HAS_CUDA, requires_peft, requires_transformers

pytestmark = [pytest.mark.gpu, pytest.mark.torch]

torch = pytest.importorskip("torch")

pytest.importorskip("transformers")

_HAS_BNB = importlib.util.find_spec("bitsandbytes") is not None
requires_bitsandbytes = pytest.mark.skipif(
    not (HAS_CUDA and _HAS_BNB), reason="requires CUDA + bitsandbytes"
)


@pytest.fixture
def tiny_tokenizer():
    from miniverl.models.tokenizers import ToyTokenizer

    return ToyTokenizer()


@pytest.fixture
def tiny_config(tiny_tokenizer):
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=tiny_tokenizer.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )


def test_cuda_peak_memory_counters_move(tiny_config, tiny_tokenizer):
    """`reset_peak_stats` then a real allocation must produce a non-zero peak."""
    from transformers import AutoModelForCausalLM

    from miniverl.config.models import Quantization
    from miniverl.models.hf import HFBackend
    from miniverl.utils import gpu

    model = AutoModelForCausalLM.from_config(tiny_config).to("cuda")
    backend = HFBackend(
        model=model,
        tokenizer=tiny_tokenizer,
        model_id="tiny",
        model_revision=None,
        device="cuda",
        dtype=torch.float32,
        quantization=Quantization.NONE,
        gradient_checkpointing=False,
        attn_implementation="sdpa",
        lora=False,
    )
    gpu.reset_peak_stats()
    before = gpu.snapshot()
    ids = tiny_tokenizer.encode("<|im_start|>assistant\n<final>\n1\n</final>")
    hidden = backend.hidden_states_at(ids, list(range(1, len(ids))), with_grad=False)
    backend.project(hidden)
    after = gpu.snapshot()
    assert before.available and after.available
    assert after.peak_allocated_bytes > 0
    assert after.peak_reserved_bytes >= after.peak_allocated_bytes
    assert after.total_bytes > 0
    payload = after.to_dict()
    assert payload["cuda_available"] is True
    assert payload["peak_allocated_gib"] >= 0.0


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_hf_cached_matches_reference_and_batch_partition(
    tiny_config, tiny_tokenizer, temperature: float
) -> None:
    """The cached backend preserves logical seeds, tokens and fp32 policy log-probs."""

    from transformers import AutoModelForCausalLM

    from miniverl.config.models import Quantization
    from miniverl.models.hf import HFBackend

    torch.manual_seed(29)
    model = AutoModelForCausalLM.from_config(tiny_config).to("cuda")
    backend = HFBackend(
        model=model,
        tokenizer=tiny_tokenizer,
        model_id="tiny",
        model_revision=None,
        device="cuda",
        dtype=torch.float32,
        quantization=Quantization.NONE,
        gradient_checkpointing=False,
        attn_implementation="sdpa",
        lora=False,
    )
    prompts = [
        tiny_tokenizer.encode("short"),
        tiny_tokenizer.encode("a longer prompt"),
        tiny_tokenizer.encode("third"),
        tiny_tokenizer.encode("the longest prompt here"),
    ]
    seeds = [201, 202, 203, 204]
    reference = [
        backend.generate(
            prompt,
            max_new_tokens=8,
            temperature=temperature,
            top_p=0.9,
            top_k=32,
            seed=seed,
            record_logprobs=True,
        )
        for prompt, seed in zip(prompts, seeds, strict=True)
    ]
    whole = backend.generate_batch_cached(
        prompts,
        max_new_tokens=8,
        temperature=temperature,
        top_p=0.9,
        top_k=32,
        seeds=seeds,
        record_logprobs=True,
    )
    split = [
        *backend.generate_batch_cached(
            prompts[:2],
            max_new_tokens=8,
            temperature=temperature,
            top_p=0.9,
            top_k=32,
            seeds=seeds[:2],
            record_logprobs=True,
        ),
        *backend.generate_batch_cached(
            prompts[2:],
            max_new_tokens=8,
            temperature=temperature,
            top_p=0.9,
            top_k=32,
            seeds=seeds[2:],
            record_logprobs=True,
        ),
    ]

    assert [row.token_ids for row in whole] == [row.token_ids for row in reference]
    assert [row.stop_reason for row in whole] == [row.stop_reason for row in reference]
    assert [row.token_ids for row in split] == [row.token_ids for row in whole]
    for expected, actual, partitioned in zip(reference, whole, split, strict=True):
        assert actual.logprobs == pytest.approx(expected.logprobs, abs=2e-4, rel=0.0)
        assert partitioned.logprobs == pytest.approx(actual.logprobs, abs=2e-4, rel=0.0)


@requires_peft
@requires_bitsandbytes
@pytest.mark.network
def test_qlora_4bit_student_loads_trains_and_reports_memory():
    """The real QLoRA path on the pinned public student."""
    from miniverl.config.models import LoRAConfig, Precision, Quantization, StudentModelConfig
    from miniverl.losses.bucketed import teacher_topk_targets
    from miniverl.losses.chunked import BucketedTargetProvider, chunked_selected_position_loss
    from miniverl.models.hf import HFBackend
    from miniverl.models.tokenizers import HFTokenizerAdapter
    from miniverl.utils import gpu

    model_id = "Qwen/Qwen3-0.6B"
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    tokenizer = HFTokenizerAdapter.load(model_id, revision=revision)
    spec = StudentModelConfig(
        model_id=model_id,
        revision=revision,
        dtype=Precision.BFLOAT16,
        quantization=Quantization.NF4,
        gradient_checkpointing=True,
        lora=LoRAConfig(enabled=True, r=8, alpha=16),
    )
    gpu.empty_cache()
    gpu.reset_peak_stats()
    backend = HFBackend.load(spec, device="cuda", tokenizer=tokenizer, trainable=True)
    assert backend.capabilities.quantization == "nf4"
    assert backend.capabilities.lora is True
    assert backend.capabilities.vocab_size == 151936
    trainable = backend.trainable_parameters()
    assert trainable and all(p.requires_grad for p in trainable)

    ids = tokenizer.encode(
        "<|im_start|>user\nCompute 2+2.<|im_end|>\n<|im_start|>assistant\n<final>\n4\n</final>"
    )
    positions = list(range(len(ids) - 8, len(ids) - 1))
    hidden = backend.hidden_states_at(ids, positions, with_grad=True)
    assert hidden.shape == (len(positions), backend.hidden_size)

    with torch.no_grad():
        teacher_logits = backend.project(hidden.detach()).float()
    idx, lp, tail = teacher_topk_targets(teacher_logits, top_k=64)
    output = chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=backend.project,
        weights=torch.ones(len(positions), device="cuda"),
        provider=BucketedTargetProvider(topk_indices=idx, topk_log_probs=lp, tail_log_prob=tail),
        chunk_size=4,
        backward=True,
    )
    # The student was scored against its own distribution, so the divergence is 0.
    assert output.loss == pytest.approx(0.0, abs=1e-4)
    assert any(p.grad is not None for p in trainable)

    snapshot = gpu.snapshot()
    assert snapshot.peak_allocated_bytes > 0
    # A 0.6B NF4 student plus one [4, 151936] chunk must stay far under 16 GiB.
    assert snapshot.peak_allocated_gib < 8.0

    backend.release()
    gpu.empty_cache()


@pytest.mark.network
def test_the_pinned_pair_shares_one_tokenizer():
    """The same-tokenizer contract, checked against the real repositories."""
    from miniverl.models.tokenizers import HFTokenizerAdapter

    student = HFTokenizerAdapter.load(
        "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca"
    )
    teacher = HFTokenizerAdapter.load(
        "Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    assert student.fingerprint == teacher.fingerprint
    # len(tokenizer) counts the ids that exist (151669); config.vocab_size is
    # padded up to 151936. miniVERL sizes the cache and top-k from the model's
    # output dimension, never from the tokenizer, so the two must not be conflated.
    assert student.vocab_size == teacher.vocab_size == 151669
    probe = "<|im_start|>assistant\n<tool_call>\n{}\n</tool_call>"
    assert student.encode(probe) == teacher.encode(probe)
    assert max(student.encode(probe)) < 151936


@requires_transformers
def test_a_cuda_run_records_measured_memory_in_its_manifest(tmp_path):
    """An end-to-end CUDA smoke test with the toy backend forced onto the GPU."""
    from miniverl.config import RunConfig
    from miniverl.trainer import OPDTrainer

    payload = {
        "schema_version": 1,
        "run": {"name": "gpu-smoke", "mode": "opd", "seed": 3, "output_dir": str(tmp_path)},
        "models": {
            "backend": "toy",
            "device": "cuda",
            "student": {
                "model_id": "toy-student",
                "lora": {"enabled": False},
                "toy": {
                    "hidden_size": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "intermediate_size": 64,
                    "max_position_embeddings": 512,
                },
            },
            "teacher": {
                "model_id": "toy-teacher",
                "toy_pretrain_steps": 2,
                "toy": {
                    "hidden_size": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "intermediate_size": 64,
                    "max_position_embeddings": 512,
                },
            },
        },
        "environment": {
            "name": "calculator",
            "params": {"prompt_style": "compact"},
            "train_tasks": 4,
            "eval_tasks": 2,
            "test_tasks": 2,
        },
        "rollout": {"max_turns": 2, "max_new_tokens_per_turn": 8, "max_total_tokens": 400},
        "loss": {"top_k": 8, "chunk_size": 16},
        "train": {"cycles": 1, "rollouts_per_cycle": 2, "gradient_accumulation_steps": 2},
        "memory": {"strategy": "resident"},
        "eval": {"enabled": True, "tasks": 2},
        "report": {"enabled": False},
    }
    trainer = OPDTrainer.from_config(RunConfig.model_validate(payload), run_id="gpu-smoke")
    try:
        result = trainer.train()
    finally:
        trainer.close()
    assert result.global_step == 1
    assert trainer.student is None

    from miniverl.reporting.data import ReportData

    data = ReportData.from_run(trainer.paths.root)
    throughput = data.throughput()
    assert throughput["cuda_available"] is True
    assert throughput["peak_allocated_gib"] is not None
    assert throughput["peak_allocated_gib"] > 0.0
    manifest = data.manifest
    assert manifest["gpu"]["available"] is True
    assert manifest["measurement_status"]["cuda_metrics"] == "measured"
