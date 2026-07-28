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
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miniverl.errors import CacheCorruptionError, CacheError, StaleCacheError
from miniverl.schemas.cache import (
    CACHE_SCHEMA_VERSION,
    CacheCompressionStats,
    CacheEntryMeta,
    CacheIndex,
    CacheShardMeta,
    TeacherTargetBatch,
)
from miniverl.utils.runs import canonical_json, write_text

__all__ = ["TeacherCache", "read_safetensors_header", "sha256_file"]

_INDEX_NAME = "index.json"
_TENSOR_FIELDS = (
    "positions",
    "topk_indices",
    "topk_log_probs",
    "tail_log_prob",
    "target_token_ids",
    "weights",
)


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

    def __init__(self, path: str | Path, index: CacheIndex, *, entries_per_shard: int = 32) -> None:
        self.path = Path(path)
        self.index = index
        self.entries_per_shard = entries_per_shard
        self._pending: dict[str, dict[str, Any]] = {}
        self._pending_order: list[str] = []
        self._shard_counter = len(index.shards)
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
        vocab_size: int,
        top_k: int,
        temperature: float,
        loss_mode: str,
        dtype: str = "float32",
        entries_per_shard: int = 32,
        overwrite: bool = True,
    ) -> TeacherCache:
        """Create (or reset) a cache directory."""
        target = Path(path)
        if target.exists() and overwrite:
            for child in sorted(target.glob("*")):
                if child.is_file():
                    child.unlink()
        target.mkdir(parents=True, exist_ok=True)
        index = CacheIndex(
            schema_version=CACHE_SCHEMA_VERSION,
            miniverl_version=miniverl_version,
            teacher_model_id=teacher_model_id,
            teacher_model_revision=teacher_model_revision,
            tokenizer_fingerprint=tokenizer_fingerprint,
            vocab_size=vocab_size,
            top_k=top_k,
            temperature=temperature,
            loss_mode=loss_mode,
            dtype=dtype,
        )
        cache = cls(target, index, entries_per_shard=entries_per_shard)
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
        if version != CACHE_SCHEMA_VERSION:
            raise StaleCacheError(
                f"cache schema_version {version!r} is not readable by this miniVERL build "
                f"(expected {CACHE_SCHEMA_VERSION})",
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

        self._pending[batch.trajectory_id] = {
            "tensors": tensors,
            "meta": {
                "policy_version": batch.policy_version,
                "num_positions": int(batch.positions.numel()),
                "selector": selector,
                "checksum": digest.hexdigest(),
                "tail_is_exact_zero": tail_is_exact_zero,
                "selected_span_types": span_counts,
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
        )

    def _next_shard_name(self) -> str:
        return f"shard-{self._shard_counter:05d}.safetensors"

    def flush(self) -> None:
        """Write staged entries into a new shard and update the index."""
        if not self._pending_order:
            return
        from safetensors.torch import save_file

        shard_name = self._next_shard_name()
        shard_path = self.path / shard_name
        payload: dict[str, Any] = {}
        for traj_id in self._pending_order:
            for key, tensor in self._pending[traj_id]["tensors"].items():
                payload[f"{traj_id}|{key}"] = tensor
        save_file(payload, str(shard_path), metadata={"miniverl_cache_shard": shard_name})
        digest, size = sha256_file(shard_path)
        self._bytes_written_total += size
        self.index.shards[shard_name] = CacheShardMeta(
            filename=shard_name,
            sha256=digest,
            size_bytes=size,
            num_entries=len(self._pending_order),
        )
        created = _utc_now()
        for traj_id in self._pending_order:
            meta = self._pending[traj_id]["meta"]
            self.index.entries[traj_id] = CacheEntryMeta(
                trajectory_id=traj_id,
                policy_version=meta["policy_version"],
                shard=shard_name,
                num_positions=meta["num_positions"],
                top_k=self.index.top_k,
                tail_is_exact_zero=meta["tail_is_exact_zero"],
                selector=meta["selector"],
                loss_mode=self.index.loss_mode,
                temperature=self.index.temperature,
                created_at=created,
                tensor_keys=list(_TENSOR_FIELDS),
                checksum=meta["checksum"],
                selected_span_types=meta["selected_span_types"],
            )
        self._pending.clear()
        self._pending_order.clear()
        self._shard_counter += 1
        self._write_index()

    def _write_index(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        index_path = write_text(
            self.path / _INDEX_NAME,
            canonical_json(self.index.model_dump(mode="json")),
        )
        self._bytes_written_total += index_path.stat().st_size

    # -- reading -----------------------------------------------------------

    def read(
        self,
        trajectory_id: str,
        *,
        expect_policy_version: int | None = None,
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
        span_types: list[str] = []
        for name, count in sorted(entry.selected_span_types.items()):
            span_types.extend([name] * count)
        return TeacherTargetBatch(
            trajectory_id=trajectory_id,
            policy_version=entry.policy_version,
            positions=loaded["positions"].to(device),
            topk_indices=loaded["topk_indices"].to(device=device, dtype=torch.long),
            topk_log_probs=loaded["topk_log_probs"].to(device=device, dtype=torch.float32),
            tail_log_prob=loaded["tail_log_prob"].to(device=device, dtype=torch.float32),
            target_token_ids=loaded["target_token_ids"].to(device),
            weights=loaded["weights"].to(device),
            temperature=entry.temperature,
            top_k=entry.top_k,
            span_types=span_types,
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

        Whole shards are deleted only when every entry they hold is dropped, so
        the index and the files stay consistent.
        """
        stale = [
            traj_id
            for traj_id, entry in self.index.entries.items()
            if entry.policy_version < policy_version
        ]
        for traj_id in stale:
            del self.index.entries[traj_id]
        live_shards = {e.shard for e in self.index.entries.values()}
        for name in list(self.index.shards):
            if name not in live_shards:
                shard_path = self.path / name
                if shard_path.is_file():
                    shard_path.unlink()
                del self.index.shards[name]
        if stale:
            self._write_index()
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
