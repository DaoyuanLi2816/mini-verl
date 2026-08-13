"""On-disk teacher-target cache.

Format
------
::

    teacher-cache/
      index.json                 # CacheIndex: schema, provenance, checksums
      shard-00000.safetensors    # tensors for up to `entries_per_shard` entries
      shard-00001.safetensors

Nothing in the read path can execute code: metadata is JSON and tensors are
safetensors.  ``torch.save``/``pickle`` are never used, so a cache directory
received from someone else is inert data.

Provenance and staleness
------------------------
The index records the teacher model id and revision, the tokenizer
fingerprint, the vocabulary size, ``top_k``, the temperature and the loss mode.
Every entry additionally records its ``policy_version``.  Reading with
``expect_policy_version`` set -- which the OPD trainer always does -- raises
:class:`~miniverl.errors.StaleCacheError` on a mismatch.  Reusing one cache
across policy versions is possible only through the explicit
``offline_kd`` mode, whose whole point is that the targets are fixed.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miniverl.errors import CacheCorruptionError, CacheError, StaleCacheError
from miniverl.schemas.cache import (
    CACHE_SCHEMA_VERSION,
    READABLE_CACHE_SCHEMA_VERSIONS,
    CacheCompressionStats,
    CacheEntryMeta,
    CacheIndex,
    CacheShardMeta,
    TeacherTargetBatch,
)
from miniverl.utils.logging import get_logger
from miniverl.utils.runs import write_json_atomic

__all__ = ["TeacherCache", "read_safetensors_header", "sha256_file"]

_INDEX_NAME = "index.json"
_SHARD_NAME = re.compile(r"^shard-(\d+)\.safetensors$")
_TENSOR_FIELDS = (
    "positions",
    "topk_indices",
    "topk_log_probs",
    "tail_log_prob",
    "target_token_ids",
    "weights",
)
logger = get_logger("cache")


def _binding_checksum(
    *,
    prompt_row_digest: str | None,
    actor_response_token_ids: list[int] | None,
    policy_version: int,
    score_implementation_version: str | None,
) -> str:
    payload = {
        "actor_response_token_ids": actor_response_token_ids,
        "policy_version": policy_version,
        "prompt_row_digest": prompt_row_digest,
        "score_implementation_version": score_implementation_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _replace_shard_file(source: Path, target: Path) -> None:
    source.replace(target)


def _delete_shard_file(path: Path) -> None:
    path.unlink()


def _next_shard_id(path: Path, index: CacheIndex) -> int:
    names = set(index.shards)
    names.update(shard.filename for shard in index.shards.values())
    if path.is_dir():
        names.update(child.name for child in path.iterdir() if child.is_file())
    suffixes = [
        int(match.group(1)) for name in names if (match := _SHARD_NAME.fullmatch(name)) is not None
    ]
    return max(suffixes, default=-1) + 1


def sha256_file(path: Path) -> tuple[str, int]:
    """Return ``(hex digest, size in bytes)`` of a file, streamed."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Parse a safetensors header without torch, numpy or the safetensors lib.

    The format is an 8-byte little-endian header length followed by that many
    bytes of UTF-8 JSON.  Parsing it directly is what lets ``miniverl cache
    stats`` inspect and validate a cache from a bare ``pip install miniverl``.
    """
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise CacheCorruptionError(f"{path.name} is too short to be a safetensors file")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len <= 0 or header_len > 100_000_000:
            raise CacheCorruptionError(
                f"{path.name} declares an implausible header length of {header_len} bytes"
            )
        blob = fh.read(header_len)
        if len(blob) != header_len:
            raise CacheCorruptionError(f"{path.name} header is truncated")
    try:
        header = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheCorruptionError(f"{path.name} header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise CacheCorruptionError(f"{path.name} header is not a JSON object")
    return header


class TeacherCache:
    """Append-only, shard-based store for compressed teacher targets."""

    def __init__(self, path: str | Path, index: CacheIndex) -> None:
        self.path = Path(path)
        self.index = index
        self.entries_per_shard = index.entries_per_shard
        self._pending: dict[str, dict[str, Any]] = {}
        self._pending_order: list[str] = []
        self._shard_counter = _next_shard_id(self.path, index)
        # Cumulative I/O accounting is intentionally independent of the current
        # footprint: pruning may shrink ``total_bytes()`` but cannot un-write the
        # shards that consumed bandwidth earlier in the run.
        self._bytes_written_total = 0

    @property
    def bytes_written_total(self) -> int:
        """Bytes written by this cache instance, including rewritten indexes."""
        return self._bytes_written_total

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        miniverl_version: str,
        teacher_model_id: str,
        teacher_model_revision: str | None,
        tokenizer_fingerprint: str,
        tokenizer_identity: dict[str, Any] | None = None,
        teacher_adapter_provenance: dict[str, Any] | None = None,
        vocab_size: int,
        top_k: int,
        temperature: float,
        loss_mode: str,
        score_implementation_version: str | None = None,
        execution_plan_digest: str | None = None,
        dtype: str = "float32",
        entries_per_shard: int = 32,
        overwrite: bool = False,
    ) -> TeacherCache:
        """Create (or reset) a cache directory."""
        target = Path(path)
        if loss_mode == "exact_full_vocab" and dtype != "float32":
            raise CacheError(
                "exact_full_vocab requires a float32 teacher cache so its objective "
                "is not silently quantized"
            )
        if target.exists() and overwrite:
            for child in sorted(target.iterdir()):
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
        elif target.exists() and any(target.iterdir()):
            raise CacheError(
                f"teacher cache already exists at {target}",
                hint="open and validate it when resuming, or choose a new cache directory",
            )
        target.mkdir(parents=True, exist_ok=True)
        index = CacheIndex(
            schema_version=CACHE_SCHEMA_VERSION,
            miniverl_version=miniverl_version,
            teacher_model_id=teacher_model_id,
            teacher_model_revision=teacher_model_revision,
            tokenizer_fingerprint=tokenizer_fingerprint,
            tokenizer_identity=dict(tokenizer_identity or {}),
            teacher_adapter_provenance=(
                dict(teacher_adapter_provenance) if teacher_adapter_provenance is not None else None
            ),
            vocab_size=vocab_size,
            top_k=top_k,
            temperature=temperature,
            loss_mode=loss_mode,
            score_implementation_version=score_implementation_version,
            execution_plan_digest=execution_plan_digest,
            dtype=dtype,
            entries_per_shard=entries_per_shard,
        )
        cache = cls(target, index)
        cache._write_index()
        return cache

    @classmethod
    def open(cls, path: str | Path, *, verify_checksums: bool = True) -> TeacherCache:
        """Open an existing cache and validate its structure."""
        target = Path(path)
        index_path = target / _INDEX_NAME
        if not index_path.is_file():
            raise CacheError(
                f"no teacher cache at {target} (missing {_INDEX_NAME})",
                hint="run `miniverl train <recipe>` first, or point at the run's "
                "teacher-cache/ directory",
            )
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CacheCorruptionError(f"{index_path} is not valid JSON: {exc}") from exc
        version = payload.get("schema_version")
        if version not in READABLE_CACHE_SCHEMA_VERSIONS:
            raise StaleCacheError(
                f"cache schema_version {version!r} is not readable by this miniVERL build "
                f"(readable versions: {sorted(READABLE_CACHE_SCHEMA_VERSIONS)})",
                hint="delete the cache directory and re-score; teacher targets are "
                "cheap to regenerate and must never be silently reinterpreted",
            )
        index = CacheIndex.model_validate(payload)
        cache = cls(target, index)
        problems = cache.validate(verify_checksums=verify_checksums)
        if problems:
            raise CacheCorruptionError(
                "teacher cache failed validation:\n  - " + "\n  - ".join(problems)
            )
        return cache

    def assert_compatible(
        self,
        *,
        teacher_model_id: str,
        teacher_model_revision: str | None,
        tokenizer_fingerprint: str,
        tokenizer_identity: dict[str, Any] | None,
        teacher_adapter_provenance: dict[str, Any] | None,
        vocab_size: int,
        top_k: int,
        temperature: float,
        loss_mode: str,
        score_implementation_version: str | None = None,
        execution_plan_digest: str | None = None,
        dtype: str,
    ) -> None:
        """Reject reuse when any objective or teacher identity component changed."""
        if self.index.schema_version < 2:
            unverified: list[str] = []
            if (tokenizer_identity or {}).get("structural_digest_v2"):
                unverified.append("structural tokenizer identity")
            if teacher_adapter_provenance is not None:
                unverified.append("teacher adapter provenance")
            if execution_plan_digest is not None:
                unverified.append("immutable execution plan identity")
            if unverified:
                raise StaleCacheError(
                    "schema-v1 teacher cache cannot verify " + " or ".join(unverified),
                    hint="re-score the legacy cache so its index records the current "
                    "tokenizer and adapter provenance",
                )
        expected: dict[str, Any] = {
            "teacher_model_id": teacher_model_id,
            "teacher_model_revision": teacher_model_revision,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "vocab_size": vocab_size,
            "top_k": top_k,
            "temperature": temperature,
            "loss_mode": loss_mode,
            "dtype": dtype,
        }
        if score_implementation_version is not None:
            expected["score_implementation_version"] = score_implementation_version
        if self.index.schema_version >= 2:
            expected["tokenizer_identity"] = dict(tokenizer_identity or {})
            expected["teacher_adapter_provenance"] = (
                dict(teacher_adapter_provenance) if teacher_adapter_provenance is not None else None
            )
            expected["execution_plan_digest"] = execution_plan_digest
        mismatches = {
            key: (getattr(self.index, key), value)
            for key, value in expected.items()
            if getattr(self.index, key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: cache={actual!r}, current={current!r}"
                for key, (actual, current) in sorted(mismatches.items())
            )
            raise StaleCacheError(f"teacher cache identity mismatch ({details})")

    # -- writing -----------------------------------------------------------

    def _torch(self) -> Any:
        from miniverl.utils.lazy import require_torch

        return require_torch("Reading or writing the teacher-target cache")

    def _dtype(self) -> Any:
        torch = self._torch()
        return torch.float16 if self.index.dtype == "float16" else torch.float32

    def write(
        self,
        batch: TeacherTargetBatch,
        *,
        selector: str,
        tail_is_exact_zero: bool = False,
    ) -> CacheEntryMeta:
        """Stage one trajectory's targets; shards are flushed automatically."""
        torch = self._torch()
        if batch.top_k and batch.top_k != self.index.top_k:
            raise CacheError(
                f"entry top_k={batch.top_k} does not match cache top_k={self.index.top_k}"
            )
        if abs(batch.temperature - self.index.temperature) > 1e-9:
            raise CacheError(
                f"entry temperature={batch.temperature} does not match cache "
                f"temperature={self.index.temperature}"
            )
        if batch.trajectory_id in self.index.entries or batch.trajectory_id in self._pending:
            raise CacheError(f"trajectory {batch.trajectory_id!r} is already in the cache")

        float_dtype = self._dtype()
        tensors = {
            "positions": batch.positions.to(torch.int64).cpu().contiguous(),
            "topk_indices": batch.topk_indices.to(torch.int32).cpu().contiguous(),
            "topk_log_probs": batch.topk_log_probs.to(float_dtype).cpu().contiguous(),
            "tail_log_prob": _finite_tail(batch.tail_log_prob, torch)
            .to(float_dtype)
            .cpu()
            .contiguous(),
            "target_token_ids": batch.target_token_ids.to(torch.int64).cpu().contiguous(),
            "weights": batch.weights.to(torch.float32).cpu().contiguous(),
        }
        digest = hashlib.sha256()
        for key in _TENSOR_FIELDS:
            digest.update(key.encode("utf-8"))
            digest.update(tensors[key].numpy().tobytes())
        span_counts: dict[str, int] = {}
        for name in batch.span_types:
            span_counts[name] = span_counts.get(name, 0) + 1
        binding_checksum = _binding_checksum(
            prompt_row_digest=batch.prompt_row_digest,
            actor_response_token_ids=batch.actor_response_token_ids,
            policy_version=batch.policy_version,
            score_implementation_version=self.index.score_implementation_version,
        )

        self._pending[batch.trajectory_id] = {
            "tensors": tensors,
            "meta": {
                "policy_version": batch.policy_version,
                "num_positions": int(batch.positions.numel()),
                "selector": selector,
                "checksum": digest.hexdigest(),
                "tail_is_exact_zero": tail_is_exact_zero,
                "selected_span_types": span_counts,
                "ordered_span_types": list(batch.span_types),
                "prompt_row_digest": batch.prompt_row_digest,
                "actor_response_token_ids": batch.actor_response_token_ids,
                "binding_checksum": binding_checksum,
            },
        }
        self._pending_order.append(batch.trajectory_id)
        if len(self._pending_order) >= self.entries_per_shard:
            self.flush()
        return CacheEntryMeta(
            trajectory_id=batch.trajectory_id,
            policy_version=batch.policy_version,
            shard=self._next_shard_name(),
            num_positions=int(batch.positions.numel()),
            top_k=self.index.top_k,
            tail_is_exact_zero=tail_is_exact_zero,
            selector=selector,
            loss_mode=self.index.loss_mode,
            temperature=self.index.temperature,
            created_at=_utc_now(),
            tensor_keys=list(_TENSOR_FIELDS),
            checksum=digest.hexdigest(),
            selected_span_types=span_counts,
            ordered_span_types=list(batch.span_types),
            prompt_row_digest=batch.prompt_row_digest,
            actor_response_token_ids=batch.actor_response_token_ids,
            binding_checksum=binding_checksum,
        )

    def _next_shard_name(self) -> str:
        return f"shard-{self._shard_counter:05d}.safetensors"

    def flush(self) -> None:
        """Publish a staged shard, then atomically replace the matching index."""
        if not self._pending_order:
            return
        from safetensors.torch import save_file

        shard_name = self._next_shard_name()
        shard_path = self.path / shard_name
        temporary_path = self.path / f".{shard_name}.tmp-{uuid.uuid4().hex}"
        payload: dict[str, Any] = {}
        for traj_id in self._pending_order:
            for key, tensor in self._pending[traj_id]["tensors"].items():
                payload[f"{traj_id}|{key}"] = tensor
        try:
            save_file(
                payload,
                str(temporary_path),
                metadata={"miniverl_cache_shard": shard_name},
            )
            digest, size = sha256_file(temporary_path)
            _replace_shard_file(temporary_path, shard_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self._bytes_written_total += size
        next_index = self.index.model_copy(deep=True)
        next_index.shards[shard_name] = CacheShardMeta(
            filename=shard_name,
            sha256=digest,
            size_bytes=size,
            num_entries=len(self._pending_order),
        )
        created = _utc_now()
        for traj_id in self._pending_order:
            meta = self._pending[traj_id]["meta"]
            next_index.entries[traj_id] = CacheEntryMeta(
                trajectory_id=traj_id,
                policy_version=meta["policy_version"],
                shard=shard_name,
                num_positions=meta["num_positions"],
                top_k=next_index.top_k,
                tail_is_exact_zero=meta["tail_is_exact_zero"],
                selector=meta["selector"],
                loss_mode=next_index.loss_mode,
                temperature=next_index.temperature,
                created_at=created,
                tensor_keys=list(_TENSOR_FIELDS),
                checksum=meta["checksum"],
                selected_span_types=meta["selected_span_types"],
                ordered_span_types=meta["ordered_span_types"],
                prompt_row_digest=meta["prompt_row_digest"],
                actor_response_token_ids=meta["actor_response_token_ids"],
                binding_checksum=meta["binding_checksum"],
            )
        self._write_index(next_index)
        self.index = next_index
        self._pending.clear()
        self._pending_order.clear()
        self._shard_counter += 1

    def _write_index(self, index: CacheIndex | None = None) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        index_path = write_json_atomic(
            self.path / _INDEX_NAME,
            (index or self.index).model_dump(mode="json"),
        )
        self._bytes_written_total += index_path.stat().st_size

    # -- reading -----------------------------------------------------------

    def read(
        self,
        trajectory_id: str,
        *,
        expect_policy_version: int | None = None,
        expect_prompt_row_digest: str | None = None,
        expect_actor_response_token_ids: list[int] | None = None,
        device: str = "cpu",
    ) -> TeacherTargetBatch:
        """Load one trajectory's targets, enforcing the policy-version contract."""
        torch = self._torch()
        from safetensors.torch import load_file

        entry = self.index.entries.get(trajectory_id)
        if entry is None:
            raise CacheError(
                f"trajectory {trajectory_id!r} is not in the teacher cache at {self.path}"
            )
        if expect_policy_version is not None and entry.policy_version != expect_policy_version:
            raise StaleCacheError(
                f"teacher targets for {trajectory_id!r} were produced by policy version "
                f"{entry.policy_version} but the update is running policy version "
                f"{expect_policy_version}",
                hint="that would make the update off-policy. Re-score the trajectory, "
                "or switch to run.mode=offline_kd if fixed targets are intended.",
            )
        if (
            expect_prompt_row_digest is not None
            and entry.prompt_row_digest != expect_prompt_row_digest
        ):
            raise StaleCacheError(
                f"teacher targets for {trajectory_id!r} have prompt-row digest "
                f"{entry.prompt_row_digest!r}, expected {expect_prompt_row_digest!r}"
            )
        if (
            expect_actor_response_token_ids is not None
            and entry.actor_response_token_ids != expect_actor_response_token_ids
        ):
            raise StaleCacheError(
                f"teacher targets for {trajectory_id!r} do not match the exact actor "
                "response token IDs"
            )
        if entry.binding_checksum is not None:
            actual_binding = _binding_checksum(
                prompt_row_digest=entry.prompt_row_digest,
                actor_response_token_ids=entry.actor_response_token_ids,
                policy_version=entry.policy_version,
                score_implementation_version=self.index.score_implementation_version,
            )
            if actual_binding != entry.binding_checksum:
                raise CacheCorruptionError(
                    f"binding checksum mismatch for {trajectory_id!r}: expected "
                    f"{entry.binding_checksum[:16]}..., got {actual_binding[:16]}..."
                )
        shard_path = self.path / entry.shard
        if not shard_path.is_file():
            raise CacheCorruptionError(f"shard {entry.shard} referenced by the index is missing")
        tensors = load_file(str(shard_path), device="cpu")
        loaded = {}
        digest = hashlib.sha256()
        for key in _TENSOR_FIELDS:
            full_key = f"{trajectory_id}|{key}"
            if full_key not in tensors:
                raise CacheCorruptionError(f"shard {entry.shard} is missing tensor {full_key!r}")
            tensor = tensors[full_key]
            digest.update(key.encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
            loaded[key] = tensor
        if digest.hexdigest() != entry.checksum:
            raise CacheCorruptionError(
                f"checksum mismatch for {trajectory_id!r} in shard {entry.shard}: "
                f"expected {entry.checksum[:16]}..., got {digest.hexdigest()[:16]}..."
            )
        if entry.ordered_span_types is not None:
            span_types = list(entry.ordered_span_types)
        elif len(entry.selected_span_types) <= 1:
            span_types = [
                name for name, count in entry.selected_span_types.items() for _ in range(count)
            ]
        else:
            raise StaleCacheError(
                f"v1 cache entry {trajectory_id!r} does not preserve ordered span types",
                hint="re-score this legacy cache before using per-span metrics",
            )
        tail_log_prob = loaded["tail_log_prob"].to(device=device, dtype=torch.float32)
        if entry.tail_is_exact_zero:
            tail_log_prob = torch.full_like(tail_log_prob, float("-inf"))
        return TeacherTargetBatch(
            trajectory_id=trajectory_id,
            policy_version=entry.policy_version,
            positions=loaded["positions"].to(device),
            topk_indices=loaded["topk_indices"].to(device=device, dtype=torch.long),
            topk_log_probs=loaded["topk_log_probs"].to(device=device, dtype=torch.float32),
            tail_log_prob=tail_log_prob,
            target_token_ids=loaded["target_token_ids"].to(device),
            weights=loaded["weights"].to(device),
            temperature=entry.temperature,
            top_k=entry.top_k,
            span_types=span_types,
            prompt_row_digest=entry.prompt_row_digest,
            actor_response_token_ids=entry.actor_response_token_ids,
        )

    def __contains__(self, trajectory_id: object) -> bool:
        return trajectory_id in self.index.entries

    def __len__(self) -> int:
        return len(self.index.entries)

    # -- integrity ----------------------------------------------------------

    def validate(self, *, verify_checksums: bool = True) -> list[str]:
        """Return a list of structural problems; empty means healthy."""
        problems: list[str] = []
        for name, shard in sorted(self.index.shards.items()):
            shard_path = self.path / name
            if not shard_path.is_file():
                problems.append(f"shard {name} is missing")
                continue
            try:
                header = read_safetensors_header(shard_path)
            except CacheCorruptionError as exc:
                problems.append(str(exc))
                continue
            tensor_names = {k for k in header if k != "__metadata__"}
            expected = {
                f"{traj_id}|{field}"
                for traj_id, entry in self.index.entries.items()
                if entry.shard == name
                for field in _TENSOR_FIELDS
            }
            missing = expected - tensor_names
            if missing:
                problems.append(
                    f"shard {name} is missing {len(missing)} tensors, e.g. {sorted(missing)[0]}"
                )
            if verify_checksums:
                digest, size = sha256_file(shard_path)
                if digest != shard.sha256:
                    problems.append(
                        f"shard {name} checksum mismatch (file {digest[:16]}..., "
                        f"index {shard.sha256[:16]}...)"
                    )
                elif size != shard.size_bytes:
                    problems.append(
                        f"shard {name} size mismatch ({size} on disk, {shard.size_bytes} in index)"
                    )
        return problems

    def total_bytes(self) -> int:
        """Bytes on disk, including the index."""
        total = sum(s.size_bytes for s in self.index.shards.values())
        index_path = self.path / _INDEX_NAME
        if index_path.is_file():
            total += index_path.stat().st_size
        return total

    def stats(self) -> CacheCompressionStats:
        """Compression accounting for this cache."""
        positions = sum(e.num_positions for e in self.index.entries.values())
        return CacheCompressionStats.compute(
            num_trajectories=len(self.index.entries),
            num_selected_positions=positions,
            top_k=self.index.top_k,
            vocab_size=self.index.vocab_size,
            actual_bytes=self.total_bytes(),
            policy_versions=sorted(self.index.policy_versions()),
        )

    def prune_before(self, policy_version: int) -> int:
        """Drop entries older than ``policy_version``; returns the number removed.

        The next index is published before its now-unreferenced shards are
        deleted.  An interrupted cleanup can therefore leave an inert orphan
        shard, but never an index that points at a shard already removed.
        """
        stale = [
            traj_id
            for traj_id, entry in self.index.entries.items()
            if entry.policy_version < policy_version
        ]
        if not stale:
            return 0

        next_index = self.index.model_copy(deep=True)
        for traj_id in stale:
            del next_index.entries[traj_id]
        live_shards = {entry.shard for entry in next_index.entries.values()}
        unreferenced_shards = []
        for name in list(next_index.shards):
            if name not in live_shards:
                unreferenced_shards.append(name)
                del next_index.shards[name]

        self._write_index(next_index)
        self.index = next_index
        for name in unreferenced_shards:
            shard_path = self.path / name
            if not shard_path.is_file():
                continue
            try:
                _delete_shard_file(shard_path)
            except OSError as exc:
                logger.warning(
                    "teacher-cache prune left unreferenced shard %s: %s",
                    shard_path,
                    exc,
                )
        return len(stale)


def _finite_tail(tensor: Any, torch: Any) -> Any:
    """Replace ``-inf`` tail log-probs with a finite sentinel for storage.

    ``-inf`` means "the top-k is the entire vocabulary".  safetensors stores it
    faithfully, but a float16 shard would turn it into ``-inf`` too and some
    downstream consumers dislike that, so it is stored as a large negative
    number.  The loss floors the tail anyway, so the substitution is invisible.
    """
    return torch.where(torch.isinf(tensor), torch.full_like(tensor, -1.0e4), tensor)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
