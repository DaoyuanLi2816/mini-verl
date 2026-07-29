# 0008. No pickle anywhere: safetensors for tensors, JSON for structure

Status: Accepted, 2026-07-27.

## Context

miniVERL writes four kinds of artifact that someone might receive from someone
else: trajectory JSONL files, a teacher-target cache directory, training
checkpoints, and run metadata. A teacher cache in particular is the sort of
thing people share, because regenerating it costs GPU time.

`torch.save` uses pickle. Loading a pickle executes code from the file. That is
a well-known hazard, and `weights_only=True` narrows it, but the narrowing is a
property of the call site rather than of the file: a single load without the
flag reintroduces the whole problem, and there is no way to tell from the file
that it was written by something benign.

## Decision

`torch.save` and `pickle` are not used anywhere in `src/miniverl/`. Tensors go
through safetensors; everything that is not a tensor goes through JSON.

**Teacher cache** (`src/miniverl/cache/store.py`): `index.json` holds the
schema version, provenance (teacher model id and revision, tokenizer
fingerprint, vocabulary size, `top_k`, temperature, loss mode, dtype) and one
`CacheShardMeta` per shard with a sha256 and a byte size. Tensors live in
`shard-NNNNN.safetensors` files written by `safetensors.torch.save_file`, keyed
`"{trajectory_id}|{field}"`. Every entry also carries its own sha256 over the
six tensor fields, recomputed and compared on read.

**Checkpoints** (`src/miniverl/training/checkpoint.py`):
`adapter.safetensors` (trainable weights only), optional
`optimizer.safetensors` (optimizer moment tensors), `state.json`, and a
`checkpoint.json` completion/integrity manifest written last. The optimizer
state dict is split by `_split_optimizer_state` into tensors, param groups and
scalars precisely so that no opaque blob has to be pickled. New checkpoints
record byte sizes and SHA-256 checksums for every payload file.

**Trajectories** (`src/miniverl/trajectory/io.py`): one JSON object per line,
schema-version-checked and pydantic-validated on both write and read.

**RNG state** (`src/miniverl/utils/seeding.py`): `random.getstate()` is
JSON-encoded, torch and CUDA generator states are base64-encoded raw bytes, and
the numpy state is captured with `legacy=True` to guarantee the 5-tuple form
rather than a dict whose layout could change.

Two consequences of the format choice were then turned into features. The
safetensors header is an 8-byte little-endian length followed by that many
bytes of UTF-8 JSON, so `read_safetensors_header` parses it with the standard
library alone -- no torch, no numpy, no safetensors package. That is what lets
`miniverl cache stats` and `miniverl cache validate` inspect and checksum a
cache from a bare `pip install miniverl`. Exactly empty probability tails are
stored as `-inf` and round-trip as exact zeros; exact full-vocabulary mode
rejects float16 cache storage so no lossy representation can be called exact.
`tests/unit/test_cache.py::test_exact_zero_tail_survives_storage` pins the
empty-tail behavior.

## Consequences

Positive:

- A cache directory or checkpoint received from a stranger is inert data.
  Reading it cannot execute anything.
- Corruption is detected rather than interpreted. `tests/unit/test_cache.py`
  covers a flipped byte, a truncated shard, a missing shard, an index
  referencing an unknown shard and corrupt index JSON, each with its own test.
- Schema versions are explicit and refused when unknown, in all three formats
  (`CACHE_SCHEMA_VERSION`, `CHECKPOINT_SCHEMA_VERSION`,
  `TRAJECTORY_SCHEMA_VERSION`), with hints that say to regenerate rather than
  reinterpret.
- `tests/integration/test_resume_and_swap.py::test_checkpoint_files_are_pickle_free`
  asserts the written files do not begin with the pickle protocol opcode
  `0x80`, so the property is checked rather than documented.

Negative:

- Optimizer state has to be flattened and rebuilt by hand
  (`_split_optimizer_state` / `_rebuild_optimizer_state`), which is code that
  `torch.save` would have provided. A future optimizer with an exotic state
  entry that is neither a tensor nor JSON-serializable would need work here.
- `_jsonable` is a best-effort conversion for optimizer scalars and falls back
  to `str(value)` on failure.
- Checksums are recomputed on read by default
  (`cache.verify_checksums_on_load`, and `--verify` on the CLI), which costs a
  full re-hash of each shard.
- JSON metadata is larger and slower to parse than a binary blob would be. At
  the scale of these runs this has not been a problem.

## Alternatives considered

**`torch.save` with `weights_only=True`.** Rejected: the safety lives at the
call site, not in the file, so one unguarded load anywhere in the ecosystem
undoes it. It also does not cover the non-tensor state, which is most of what a
checkpoint contains.

**A single-file archive format (npz, zip, HDF5).** Rejected: npz is numpy
pickle-adjacent for object arrays, and HDF5 adds a native dependency to a
package whose base install is deliberately dependency-light.

**Skip checksums and rely on the filesystem.** Rejected: the cache is
explicitly designed to be copied between machines and reused across cycles, and
a silently corrupted teacher target produces a plausible loss curve.
