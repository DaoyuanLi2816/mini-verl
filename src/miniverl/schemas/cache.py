"""Schemas for the on-disk teacher-target cache.

The cache is deliberately *not* a pickle.  Metadata is JSON, tensors live in
`safetensors <https://github.com/huggingface/safetensors>`_ shards, and every
shard is checksummed.  Nothing in the load path executes code from the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "READABLE_CACHE_SCHEMA_VERSIONS",
    "CacheEntryMeta",
    "CacheShardMeta",
    "CacheIndex",
    "TeacherTargetBatch",
    "CacheCompressionStats",
]

CACHE_SCHEMA_VERSION = 2
READABLE_CACHE_SCHEMA_VERSIONS = frozenset({1, CACHE_SCHEMA_VERSION})


class CacheEntryMeta(BaseModel):
    """Metadata for the teacher targets of a single trajectory."""

    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    policy_version: int = Field(ge=0)
    shard: str
    num_positions: int = Field(ge=0)
    top_k: int = Field(ge=1)
    tail_is_exact_zero: bool = False
    selector: str
    loss_mode: str
    temperature: float = Field(gt=0.0)
    created_at: str
    tensor_keys: list[str]
    checksum: str
    selected_span_types: dict[str, int] = Field(default_factory=dict)
    ordered_span_types: list[str] | None = None
    prompt_row_digest: str | None = None
    actor_response_token_ids: list[int] | None = None
    binding_checksum: str | None = None


class CacheShardMeta(BaseModel):
    """Integrity record for one safetensors shard."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    sha256: str
    size_bytes: int = Field(ge=0)
    num_entries: int = Field(ge=0)


class CacheIndex(BaseModel):
    """Top-level ``index.json`` of a teacher-target cache directory."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = CACHE_SCHEMA_VERSION
    miniverl_version: str
    teacher_model_id: str
    teacher_model_revision: str | None = None
    tokenizer_fingerprint: str
    tokenizer_identity: dict[str, Any] = Field(default_factory=dict)
    teacher_adapter_provenance: dict[str, Any] | None = None
    vocab_size: int = Field(gt=0)
    top_k: int = Field(ge=1)
    temperature: float = Field(gt=0.0)
    loss_mode: str
    score_implementation_version: str | None = None
    execution_plan_digest: str | None = None
    dtype: str = "float32"
    entries_per_shard: int = Field(default=32, ge=1, le=4096)
    entries: dict[str, CacheEntryMeta] = Field(default_factory=dict)
    shards: dict[str, CacheShardMeta] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> CacheIndex:
        if self.schema_version not in READABLE_CACHE_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported cache schema_version {self.schema_version}")
        if self.top_k > self.vocab_size:
            raise ValueError(f"top_k={self.top_k} exceeds vocab_size={self.vocab_size}")
        for traj_id, entry in self.entries.items():
            if entry.trajectory_id != traj_id:
                raise ValueError(
                    f"cache index key {traj_id!r} does not match entry id {entry.trajectory_id!r}"
                )
            if entry.shard not in self.shards:
                raise ValueError(f"entry {traj_id!r} references unknown shard {entry.shard!r}")
        return self

    def policy_versions(self) -> set[int]:
        """Distinct policy versions represented in this cache."""
        return {e.policy_version for e in self.entries.values()}


@dataclass(slots=True)
class TeacherTargetBatch:
    """In-memory compressed teacher targets for one trajectory.

    Tensor fields are typed ``Any`` so this module stays importable without
    torch; at runtime they are ``torch.Tensor`` objects with shapes:

    ``positions``      ``[N]`` int64 -- teacher prediction positions
    ``topk_indices``   ``[N, K]`` int64 -- vocabulary ids of the teacher top-k
    ``topk_log_probs`` ``[N, K]`` float32 -- ``log p_teacher`` over the *full*
                       vocabulary, restricted to the top-k ids (they do **not**
                       sum to one)
    ``tail_log_prob``  ``[N]`` float32 -- ``log(1 - sum_k p_teacher)``
    ``target_token_ids`` ``[N]`` int64
    ``weights``        ``[N]`` float32
    """

    trajectory_id: str
    policy_version: int
    positions: Any
    topk_indices: Any
    topk_log_probs: Any
    tail_log_prob: Any
    target_token_ids: Any
    weights: Any
    temperature: float = 1.0
    top_k: int = 0
    span_types: list[str] = field(default_factory=list)
    prompt_row_digest: str | None = None
    actor_response_token_ids: list[int] | None = None


class CacheCompressionStats(BaseModel):
    """Compression accounting for a teacher-target cache."""

    model_config = ConfigDict(extra="forbid")

    num_trajectories: int = Field(ge=0)
    num_selected_positions: int = Field(ge=0)
    top_k: int = Field(ge=1)
    vocab_size: int = Field(gt=0)
    actual_bytes: int = Field(ge=0)
    theoretical_full_logit_bytes: int = Field(ge=0)
    bytes_per_selected_position: float = Field(ge=0.0)
    compression_ratio: float = Field(ge=0.0)
    dtype_bytes_assumed: int = Field(gt=0, default=2)
    policy_versions: list[int] = Field(default_factory=list)

    @classmethod
    def compute(
        cls,
        *,
        num_trajectories: int,
        num_selected_positions: int,
        top_k: int,
        vocab_size: int,
        actual_bytes: int,
        policy_versions: list[int],
        dtype_bytes_assumed: int = 2,
    ) -> CacheCompressionStats:
        """Derive compression statistics.

        ``theoretical_full_logit_bytes`` is the size a dense
        ``[num_selected_positions, vocab_size]`` FP16 logit dump would need --
        the honest baseline this cache is compared against.  It is *not* the
        size of a full ``[batch, seq_len, vocab]`` dump, which would be larger
        still.
        """
        theoretical = num_selected_positions * vocab_size * dtype_bytes_assumed
        per_pos = actual_bytes / num_selected_positions if num_selected_positions else 0.0
        ratio = theoretical / actual_bytes if actual_bytes else 0.0
        return cls(
            num_trajectories=num_trajectories,
            num_selected_positions=num_selected_positions,
            top_k=top_k,
            vocab_size=vocab_size,
            actual_bytes=actual_bytes,
            theoretical_full_logit_bytes=theoretical,
            bytes_per_selected_position=per_pos,
            compression_ratio=ratio,
            dtype_bytes_assumed=dtype_bytes_assumed,
            policy_versions=sorted(policy_versions),
        )
