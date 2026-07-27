"""Cache inspection that works without torch.

``miniverl cache stats`` must run from a bare ``pip install miniverl``: opening
``index.json``, parsing safetensors headers and checksumming shards needs no
tensor framework at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniverl.cache.store import TeacherCache, read_safetensors_header

__all__ = ["compute_stats", "format_stats"]


def compute_stats(path: str | Path, *, verify_checksums: bool = True) -> dict[str, Any]:
    """Open a cache directory and summarize it as JSON-friendly data."""
    cache = TeacherCache.open(path, verify_checksums=verify_checksums)
    stats = cache.stats()
    index = cache.index
    by_version: dict[str, int] = {}
    by_selector: dict[str, int] = {}
    span_totals: dict[str, int] = {}
    for entry in index.entries.values():
        key = str(entry.policy_version)
        by_version[key] = by_version.get(key, 0) + 1
        by_selector[entry.selector] = by_selector.get(entry.selector, 0) + 1
        for name, count in entry.selected_span_types.items():
            span_totals[name] = span_totals.get(name, 0) + count

    shard_details = []
    for name, shard in sorted(index.shards.items()):
        shard_path = Path(cache.path) / name
        tensors = 0
        if shard_path.is_file():
            header = read_safetensors_header(shard_path)
            tensors = sum(1 for k in header if k != "__metadata__")
        shard_details.append(
            {
                "filename": name,
                "size_bytes": shard.size_bytes,
                "entries": shard.num_entries,
                "tensors": tensors,
                "sha256": shard.sha256,
            }
        )

    return {
        "path": str(cache.path),
        "schema_version": index.schema_version,
        "miniverl_version": index.miniverl_version,
        "teacher_model_id": index.teacher_model_id,
        "teacher_model_revision": index.teacher_model_revision,
        "tokenizer_fingerprint": index.tokenizer_fingerprint,
        "loss_mode": index.loss_mode,
        "dtype": index.dtype,
        "temperature": index.temperature,
        "top_k": index.top_k,
        "vocab_size": index.vocab_size,
        "trajectories": stats.num_trajectories,
        "selected_positions": stats.num_selected_positions,
        "actual_bytes": stats.actual_bytes,
        "theoretical_full_logit_bytes": stats.theoretical_full_logit_bytes,
        "compression_ratio": stats.compression_ratio,
        "bytes_per_selected_position": stats.bytes_per_selected_position,
        "policy_versions": stats.policy_versions,
        "entries_by_policy_version": by_version,
        "entries_by_selector": by_selector,
        "selected_positions_by_span_type": span_totals,
        "shards": shard_details,
        "checksums_verified": verify_checksums,
        "problems": cache.validate(verify_checksums=verify_checksums),
    }


def format_stats(stats: dict[str, Any]) -> str:
    """Render :func:`compute_stats` output as plain text."""
    lines = [
        f"teacher cache      {stats['path']}",
        f"  schema           v{stats['schema_version']} (written by miniVERL {stats['miniverl_version']})",
        f"  teacher          {stats['teacher_model_id']} @ {stats['teacher_model_revision'] or 'unpinned'}",
        f"  tokenizer        {str(stats['tokenizer_fingerprint'])[:16]}...",
        f"  objective        {stats['loss_mode']} | top_k={stats['top_k']} | T={stats['temperature']}",
        f"  storage dtype    {stats['dtype']}",
        f"  trajectories     {stats['trajectories']}",
        f"  positions        {stats['selected_positions']}",
        f"  policy versions  {stats['policy_versions']}",
        f"  actual size      {_human(stats['actual_bytes'])}",
        f"  dense fp16 ref   {_human(stats['theoretical_full_logit_bytes'])}"
        f"  (positions x vocab {stats['vocab_size']} fp16 logits)",
        f"  compression      {stats['compression_ratio']:.1f}x",
        f"  bytes/position   {stats['bytes_per_selected_position']:.1f}",
        f"  shards           {len(stats['shards'])}",
        f"  checksums        {'verified' if stats['checksums_verified'] else 'not verified'}",
    ]
    problems = stats.get("problems") or []
    if problems:
        lines.append(f"  PROBLEMS         {len(problems)}")
        lines.extend(f"    - {p}" for p in problems)
    else:
        lines.append("  problems         none")
    return "\n".join(lines)


def _human(num_bytes: float) -> str:
    """Format a byte count with a binary unit."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"  # pragma: no cover - unreachable
