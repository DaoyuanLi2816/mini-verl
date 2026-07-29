"""Teacher-target cache: round trip, provenance, staleness and corruption.

The cache is the artifact that decides whether a run is on-policy.  These tests
protect three things: the tensors survive a round trip exactly, a target
produced under one policy version cannot be consumed under another, and a
corrupted shard is detected rather than trained on.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from miniverl.cache.stats import compute_stats, format_stats
from miniverl.cache.store import TeacherCache, read_safetensors_header, sha256_file
from miniverl.errors import CacheCorruptionError, CacheError, StaleCacheError
from miniverl.schemas.cache import CACHE_SCHEMA_VERSION, CacheCompressionStats, TeacherTargetBatch
from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]

torch = pytest.importorskip("torch")

VOCAB = 512
TOP_K = 8


def _batch(
    trajectory_id: str, *, policy_version: int = 0, positions: int = 5
) -> TeacherTargetBatch:
    generator = torch.Generator().manual_seed(len(trajectory_id) + policy_version)
    logits = torch.randn(positions, VOCAB, generator=generator) * 2.0
    from miniverl.losses.bucketed import teacher_topk_targets

    idx, lp, tail = teacher_topk_targets(logits, top_k=TOP_K)
    return TeacherTargetBatch(
        trajectory_id=trajectory_id,
        policy_version=policy_version,
        positions=torch.arange(positions, dtype=torch.long),
        topk_indices=idx,
        topk_log_probs=lp,
        tail_log_prob=tail,
        target_token_ids=torch.randint(0, VOCAB, (positions,), generator=generator),
        weights=torch.ones(positions),
        temperature=1.0,
        top_k=TOP_K,
        span_types=["assistant_tool_call"] * positions,
    )


def _cache(path: Path, *, dtype: str = "float32", entries_per_shard: int = 2) -> TeacherCache:
    return TeacherCache.create(
        path,
        miniverl_version="0.1.0",
        teacher_model_id="toy-teacher",
        teacher_model_revision="rev-abc",
        tokenizer_fingerprint="fp-1234",
        vocab_size=VOCAB,
        top_k=TOP_K,
        temperature=1.0,
        loss_mode="bucketed_topk_tail",
        dtype=dtype,
        entries_per_shard=entries_per_shard,
    )


# ----------------------------------------------------------- round trip


def test_float32_round_trip_is_exact(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    batch = _batch("t0", policy_version=3)
    cache.write(batch, selector="all_model_tokens")
    cache.flush()

    loaded = cache.read("t0", expect_policy_version=3)
    assert torch.equal(loaded.topk_indices, batch.topk_indices)
    assert torch.allclose(loaded.topk_log_probs, batch.topk_log_probs, atol=0.0)
    assert torch.equal(loaded.target_token_ids, batch.target_token_ids)
    assert torch.equal(loaded.positions, batch.positions)
    assert loaded.top_k == TOP_K
    assert loaded.temperature == 1.0
    assert loaded.span_types == batch.span_types


def test_float16_round_trip_stays_within_its_documented_precision(tmp_path: Path):
    cache = _cache(tmp_path / "tc", dtype="float16")
    batch = _batch("t0")
    cache.write(batch, selector="all_model_tokens")
    cache.flush()
    loaded = cache.read("t0")
    # Indices are integers and must be exact even in a float16 cache.
    assert torch.equal(loaded.topk_indices, batch.topk_indices)
    assert torch.allclose(loaded.topk_log_probs, batch.topk_log_probs, atol=2e-3)


def test_exact_zero_tail_survives_storage(tmp_path: Path):
    """``k == V`` yields ``-inf`` and must round-trip as exact empty mass."""
    cache = TeacherCache.create(
        tmp_path / "tc",
        miniverl_version="0.1.0",
        teacher_model_id="toy",
        teacher_model_revision=None,
        tokenizer_fingerprint="fp",
        vocab_size=VOCAB,
        top_k=VOCAB,
        temperature=1.0,
        loss_mode="exact_full_vocab",
    )
    from miniverl.losses.bucketed import teacher_topk_targets

    logits = torch.randn(4, VOCAB, generator=torch.Generator().manual_seed(1))
    idx, lp, tail = teacher_topk_targets(logits, top_k=VOCAB)
    assert bool(torch.isinf(tail).all())
    cache.write(
        TeacherTargetBatch(
            trajectory_id="t",
            policy_version=0,
            positions=torch.arange(4),
            topk_indices=idx,
            topk_log_probs=lp,
            tail_log_prob=tail,
            target_token_ids=torch.zeros(4, dtype=torch.long),
            weights=torch.ones(4),
            temperature=1.0,
            top_k=VOCAB,
            span_types=["assistant_final"] * 4,
        ),
        selector="all_model_tokens",
        tail_is_exact_zero=True,
    )
    cache.flush()
    loaded = cache.read("t")
    assert bool(torch.isneginf(loaded.tail_log_prob).all())


def test_ordered_span_types_survive_a_heterogeneous_round_trip(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "tc")
    batch = _batch("ordered", positions=5)
    batch.span_types = [
        "assistant_text",
        "assistant_final",
        "assistant_tool_call",
        "assistant_text",
        "assistant_final",
    ]
    cache.write(batch, selector="hybrid")
    cache.flush()

    assert cache.read("ordered").span_types == batch.span_types


def test_exact_full_vocab_refuses_a_lossy_float16_cache(tmp_path: Path) -> None:
    with pytest.raises(CacheError, match=r"exact_full_vocab.*float32"):
        TeacherCache.create(
            tmp_path / "tc",
            miniverl_version="0.2.1.dev0",
            teacher_model_id="toy",
            teacher_model_revision=None,
            tokenizer_fingerprint="fp",
            vocab_size=VOCAB,
            top_k=VOCAB,
            temperature=1.0,
            loss_mode="exact_full_vocab",
            dtype="float16",
        )


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "jsd"])
def test_exact_full_vocab_cached_provider_matches_resident_exact_provider(
    tmp_path: Path,
    divergence: str,
) -> None:
    from miniverl.losses.bucketed import teacher_topk_targets
    from miniverl.losses.chunked import BucketedTargetProvider, ExactTargetProvider

    generator = torch.Generator().manual_seed(44)
    teacher_logits = torch.randn(4, 17, generator=generator)
    student_logits = torch.randn(4, 17, generator=generator)
    indices, log_probs, tail = teacher_topk_targets(teacher_logits, top_k=17)
    cache = TeacherCache.create(
        tmp_path / "exact",
        miniverl_version="0.2.1.dev0",
        teacher_model_id="teacher",
        teacher_model_revision="rev",
        tokenizer_fingerprint="fp",
        vocab_size=17,
        top_k=17,
        temperature=1.0,
        loss_mode="exact_full_vocab",
        dtype="float32",
    )
    cache.write(
        TeacherTargetBatch(
            trajectory_id="exact",
            policy_version=0,
            positions=torch.arange(4),
            topk_indices=indices,
            topk_log_probs=log_probs,
            tail_log_prob=tail,
            target_token_ids=torch.zeros(4, dtype=torch.long),
            weights=torch.ones(4),
            temperature=1.0,
            top_k=17,
            span_types=["assistant_final"] * 4,
        ),
        selector="all_model_tokens",
        tail_is_exact_zero=True,
    )
    cache.flush()
    loaded = cache.read("exact")

    resident = ExactTargetProvider(
        teacher_logits_fn=lambda start, end: teacher_logits[start:end],
        divergence_name=divergence,
    )
    cached = BucketedTargetProvider(
        topk_indices=loaded.topk_indices,
        topk_log_probs=loaded.topk_log_probs,
        tail_log_prob=loaded.tail_log_prob,
        divergence_name=divergence,
    )
    assert torch.allclose(
        resident.divergence(0, 4, student_logits),
        cached.divergence(0, 4, student_logits),
        atol=1e-6,
        rtol=1e-6,
    )


def test_cache_identity_includes_teacher_adapter_provenance(tmp_path: Path) -> None:
    provenance = {
        "source": "hub",
        "identity": "owner/adapter",
        "revision": "a" * 40,
        "weights_sha256": "b" * 64,
        "manifest_digest": "c" * 64,
        "base_model_revision": "d" * 40,
    }
    cache = TeacherCache.create(
        tmp_path / "tc",
        miniverl_version="0.2.1.dev0",
        teacher_model_id="teacher",
        teacher_model_revision="d" * 40,
        tokenizer_fingerprint="fp",
        tokenizer_identity={"structural_digest_v2": "e" * 64},
        teacher_adapter_provenance=provenance,
        vocab_size=VOCAB,
        top_k=TOP_K,
        temperature=1.0,
        loss_mode="bucketed_topk_tail",
    )
    cache.assert_compatible(
        teacher_model_id="teacher",
        teacher_model_revision="d" * 40,
        tokenizer_fingerprint="fp",
        tokenizer_identity={"structural_digest_v2": "e" * 64},
        teacher_adapter_provenance=provenance,
        vocab_size=VOCAB,
        top_k=TOP_K,
        temperature=1.0,
        loss_mode="bucketed_topk_tail",
        dtype="float32",
    )
    changed = {**provenance, "revision": "f" * 40}
    with pytest.raises(StaleCacheError, match="teacher_adapter_provenance"):
        cache.assert_compatible(
            teacher_model_id="teacher",
            teacher_model_revision="d" * 40,
            tokenizer_fingerprint="fp",
            tokenizer_identity={"structural_digest_v2": "e" * 64},
            teacher_adapter_provenance=changed,
            vocab_size=VOCAB,
            top_k=TOP_K,
            temperature=1.0,
            loss_mode="bucketed_topk_tail",
            dtype="float32",
        )


def test_sharding_is_deterministic_and_index_stays_consistent(tmp_path: Path):
    cache = _cache(tmp_path / "tc", entries_per_shard=2)
    for i in range(5):
        cache.write(_batch(f"t{i}"), selector="hybrid")
    cache.flush()
    assert len(cache) == 5
    assert len(cache.index.shards) == 3  # 2 + 2 + 1
    names = sorted(cache.index.shards)
    assert names == [
        "shard-00000.safetensors",
        "shard-00001.safetensors",
        "shard-00002.safetensors",
    ]
    for entry in cache.index.entries.values():
        assert entry.shard in cache.index.shards
    assert cache.validate() == []


def test_reopening_a_cache_validates_it(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    cache.write(_batch("t0"), selector="hybrid")
    cache.flush()
    reopened = TeacherCache.open(tmp_path / "tc")
    assert len(reopened) == 1
    assert reopened.index.teacher_model_revision == "rev-abc"
    assert "t0" in reopened


# --------------------------------------------------------- provenance


def test_reading_with_a_different_policy_version_raises(tmp_path: Path):
    """The guard that keeps OPD on-policy."""
    cache = _cache(tmp_path / "tc")
    cache.write(_batch("t0", policy_version=4), selector="hybrid")
    cache.flush()
    with pytest.raises(StaleCacheError, match="off-policy"):
        cache.read("t0", expect_policy_version=5)
    # Not passing an expectation is the offline-KD path and is allowed.
    assert cache.read("t0").policy_version == 4


def test_duplicate_trajectory_is_rejected(tmp_path: Path):
    cache = _cache(tmp_path / "tc", entries_per_shard=8)
    cache.write(_batch("t0"), selector="hybrid")
    with pytest.raises(CacheError, match="already in the cache"):
        cache.write(_batch("t0"), selector="hybrid")


def test_mismatched_top_k_or_temperature_is_rejected(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    batch = _batch("t0")
    batch.top_k = TOP_K + 1
    with pytest.raises(CacheError, match="does not match cache top_k"):
        cache.write(batch, selector="hybrid")
    batch.top_k = TOP_K
    batch.temperature = 2.0
    with pytest.raises(CacheError, match="does not match cache"):
        cache.write(batch, selector="hybrid")


def test_reading_an_unknown_trajectory_raises(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    cache.write(_batch("t0"), selector="hybrid")
    cache.flush()
    with pytest.raises(CacheError, match="is not in the teacher cache"):
        cache.read("nope")


def test_opening_a_missing_cache_raises_with_a_hint(tmp_path: Path):
    with pytest.raises(CacheError, match="no teacher cache") as excinfo:
        TeacherCache.open(tmp_path / "absent")
    assert excinfo.value.hint


def test_incompatible_schema_version_is_refused(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    cache.write(_batch("t0"), selector="hybrid")
    cache.flush()
    index_path = tmp_path / "tc" / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["schema_version"] = CACHE_SCHEMA_VERSION + 99
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StaleCacheError, match="not readable by this miniVERL build"):
        TeacherCache.open(tmp_path / "tc")


def test_prune_before_removes_old_policy_versions_and_their_shards(tmp_path: Path):
    cache = _cache(tmp_path / "tc", entries_per_shard=1)
    cache.write(_batch("old", policy_version=1), selector="hybrid")
    cache.write(_batch("new", policy_version=5), selector="hybrid")
    cache.flush()
    assert len(cache.index.shards) == 2
    removed = cache.prune_before(5)
    assert removed == 1
    assert "old" not in cache
    assert "new" in cache


def test_pruning_does_not_reduce_cumulative_bytes_written(tmp_path: Path):
    cache = _cache(tmp_path / "tc", entries_per_shard=1)
    cache.write(_batch("old", policy_version=1), selector="hybrid")
    cache.write(_batch("new", policy_version=5), selector="hybrid")
    cache.flush()
    before = cache.bytes_written_total
    footprint_before = cache.total_bytes()

    cache.prune_before(5)

    assert cache.total_bytes() < footprint_before
    assert cache.bytes_written_total >= before
    assert len(cache.index.shards) == 1
    assert cache.validate() == []


# --------------------------------------------------------- corruption


def test_a_flipped_byte_in_a_shard_is_detected(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    cache.write(_batch("t0"), selector="hybrid")
    cache.flush()
    shard = next((tmp_path / "tc").glob("shard-*.safetensors"))
    data = bytearray(shard.read_bytes())
    data[-3] ^= 0xFF  # corrupt inside the tensor payload
    shard.write_bytes(bytes(data))

    with pytest.raises(CacheCorruptionError, match="checksum mismatch"):
        TeacherCache.open(tmp_path / "tc")
    # Even skipping shard verification, the per-entry checksum catches it.
    lax = TeacherCache.open(tmp_path / "tc", verify_checksums=False)
    with pytest.raises(CacheCorruptionError, match="checksum mismatch"):
        lax.read("t0")


def test_a_truncated_shard_is_detected(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    cache.write(_batch("t0"), selector="hybrid")
    cache.flush()
    shard = next((tmp_path / "tc").glob("shard-*.safetensors"))
    shard.write_bytes(shard.read_bytes()[:20])
    with pytest.raises(CacheCorruptionError):
        TeacherCache.open(tmp_path / "tc")


def test_a_missing_shard_is_detected(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    cache.write(_batch("t0"), selector="hybrid")
    cache.flush()
    next((tmp_path / "tc").glob("shard-*.safetensors")).unlink()
    problems = cache.validate()
    assert problems and "missing" in problems[0]
    with pytest.raises(CacheCorruptionError, match="failed validation"):
        TeacherCache.open(tmp_path / "tc")


def test_index_referencing_an_unknown_shard_is_rejected(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    cache.write(_batch("t0"), selector="hybrid")
    cache.flush()
    index_path = tmp_path / "tc" / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["entries"]["t0"]["shard"] = "shard-99999.safetensors"
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="unknown shard"):
        TeacherCache.open(tmp_path / "tc")


def test_corrupt_index_json_is_rejected(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    cache.flush()
    (tmp_path / "tc" / "index.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CacheCorruptionError, match="not valid JSON"):
        TeacherCache.open(tmp_path / "tc")


# ------------------------------------------------- torch-free header path


def test_safetensors_header_parses_without_a_framework(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    cache.write(_batch("t0", positions=3), selector="hybrid")
    cache.flush()
    shard = next((tmp_path / "tc").glob("shard-*.safetensors"))
    header = read_safetensors_header(shard)
    keys = {k for k in header if k != "__metadata__"}
    assert "t0|topk_indices" in keys
    assert header["t0|topk_indices"]["shape"] == [3, TOP_K]
    assert header["t0|topk_indices"]["dtype"] == "I32"


def test_header_parser_rejects_nonsense(tmp_path: Path):
    tiny = tmp_path / "tiny.safetensors"
    tiny.write_bytes(b"abc")
    with pytest.raises(CacheCorruptionError, match="too short"):
        read_safetensors_header(tiny)
    absurd = tmp_path / "absurd.safetensors"
    absurd.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(CacheCorruptionError, match="implausible header length"):
        read_safetensors_header(absurd)
    truncated = tmp_path / "trunc.safetensors"
    truncated.write_bytes(struct.pack("<Q", 100) + b"{}")
    with pytest.raises(CacheCorruptionError, match="truncated"):
        read_safetensors_header(truncated)


def test_sha256_file_matches_hashlib(tmp_path: Path):
    import hashlib

    target = tmp_path / "blob"
    payload = b"miniverl" * 1000
    target.write_bytes(payload)
    digest, size = sha256_file(target)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)


# ---------------------------------------------------------- statistics


def test_compression_statistics_are_arithmetically_honest(tmp_path: Path):
    cache = _cache(tmp_path / "tc", entries_per_shard=4)
    for i in range(4):
        cache.write(_batch(f"t{i}", positions=6), selector="hybrid")
    cache.flush()
    stats = cache.stats()
    assert stats.num_trajectories == 4
    assert stats.num_selected_positions == 24
    # The reference is a dense [positions x vocab] fp16 dump, nothing larger.
    assert stats.theoretical_full_logit_bytes == 24 * VOCAB * 2
    assert stats.actual_bytes == cache.total_bytes()
    assert stats.compression_ratio == pytest.approx(
        stats.theoretical_full_logit_bytes / stats.actual_bytes
    )
    assert stats.bytes_per_selected_position == pytest.approx(stats.actual_bytes / 24)


def test_compression_stats_handle_an_empty_cache():
    stats = CacheCompressionStats.compute(
        num_trajectories=0,
        num_selected_positions=0,
        top_k=8,
        vocab_size=100,
        actual_bytes=0,
        policy_versions=[],
    )
    assert stats.compression_ratio == 0.0
    assert stats.bytes_per_selected_position == 0.0


def test_compute_stats_and_format_stats_report_provenance(tmp_path: Path):
    cache = _cache(tmp_path / "tc")
    cache.write(_batch("t0", policy_version=2), selector="tool_and_final")
    cache.flush()
    stats = compute_stats(tmp_path / "tc")
    assert stats["teacher_model_id"] == "toy-teacher"
    assert stats["teacher_model_revision"] == "rev-abc"
    assert stats["policy_versions"] == [2]
    assert stats["entries_by_selector"] == {"tool_and_final": 1}
    assert stats["problems"] == []
    assert stats["checksums_verified"] is True
    text = format_stats(stats)
    assert "toy-teacher" in text
    assert "compression" in text
    assert "problems         none" in text
