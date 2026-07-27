# Changelog

All notable changes to miniVERL are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-27

First public release. Multi-turn, tool-aware on-policy distillation that runs on
one consumer GPU.

### Added

**Objective**

- Exact full-vocabulary forward KL, reverse KL and beta-weighted Jensen-Shannon
  divergence, each checked against a brute-force Python reference.
- Compressed `top-k + tail` variants, named `bucketed_forward_kl`,
  `bucketed_reverse_kl` and `bucketed_jsd` so they cannot be mistaken for the
  exact objective. Tests assert the data-processing-inequality lower bound and
  convergence to the exact loss at `k == V`.
- Optional temperature with the documented `T^2` gradient correction, per-token
  weights, weight-sum normalization, and a convex cross-entropy mixing term.
- A chunked selected-position objective whose two-stage backward reproduces the
  unchunked gradient exactly while never materializing more than
  `[chunk_size, vocab]`.

**Provenance**

- Span-partitioned trajectories with `system`, `user`, `assistant_text`,
  `assistant_tool_call`, `tool_result` and `assistant_final` types. Masks are
  stored and re-derived on every read; a mismatch is rejected.
- Explicit target-position versus prediction-position conversion, with position
  `0` permanently excluded.
- Privileged-context teacher mode with a per-segment alignment map that verifies
  target-token identity on both sides.

**Backends**

- A reversible ~190-entry toy tokenizer and a RoPE/RMSNorm/SwiGLU toy
  transformer, so the whole pipeline runs on a CPU with no network.
- A Hugging Face causal-LM backend that calls the decoder backbone directly and
  projects only the selected positions, with QLoRA (NF4), gradient
  checkpointing, SDPA attention and deterministic generation.
- An architecture adapter that resolves the backbone and LM head through PEFT
  wrappers; tested against `Qwen3ForCausalLM`.

**Training**

- SFT, offline KD and genuine OPD behind one trainer, with the on-policy
  distinction enforced by policy-version checks rather than documentation.
- `resident`, `swap` and `auto` memory strategies; bounded, mathematically
  neutral OOM retries that only shrink the projection chunk.
- Pickle-free checkpoints (safetensors plus JSON) and exact resume, asserted
  parameter-for-parameter against an uninterrupted run.

**Environments**

- Calculator (AST-only evaluation, no `eval`), JSON navigation, and SQLite with
  an authorizer-enforced read-only connection and an instruction budget. All
  three are seeded, have disjoint splits, exact verifiers and deterministic
  oracles.

**Tooling**

- `miniverl doctor / validate / demo / train / eval / benchmark / inspect /
  report / cache / export-benchmark / schema`, all with JSON output where
  useful, and a `--dry-run` path that downloads nothing.
- A versioned, checksummed teacher-target cache with compression statistics and
  corruption detection, readable without torch.
- Self-contained offline HTML reports with a token-level teacher/student
  divergence view, plus Markdown and JSON summaries.
- A matched-budget benchmark harness that starts every arm from one shared
  cold-start checkpoint and records what it held constant.
- `scripts/attribute_failures.py`, which re-scores collected trajectories with a
  lenient answer parser so a zero success rate can be split into "could not do
  the task" and "did the task and formatted it wrong".

### Measured

- RTX 4080 (16 GB), `recipes/qwen_consumer_gpu_calc.yaml`: 16 optimizer steps in
  481.1 s; peak 4.251 GiB allocated / 4.762 GiB reserved; held-out greedy task
  success 0.0% to 100.0% on 12 tasks. The supervised cold start does most of that
  work; see `docs/rtx4080-baselines.md`.
- Decode throughput on the same machine is kernel-launch bound: 11.19 tok/s with
  NF4 and 12.84 tok/s with bf16 LoRA, and a 14-token prefill (37.0 ms) costs
  about the same as a cached single-token step (30.9 ms).
- **The matched-budget comparison on the non-saturating `hard` split came out
  negative for on-policy distillation.** From one shared cold start at 62.5%,
  with 12 identical optimizer steps: supervised continuation 100.0%, OPD against
  the raw instruct teacher 0.0%, OPD against a privileged-context teacher 0.0%.
  The failure is diagnosed from decoded transcripts in
  `docs/rtx4080-baselines.md` and is most likely caused by the teacher, which was
  never trained on the tool protocol; the experiment that would confirm it is
  specified there but was not run. This is a single seed on one task family and
  is not a general claim about the method -- but it is what this repository
  measured about its own headline feature, so it is reported here rather than
  only in the docs.

### Known limitations

Same-tokenizer only; one trajectory per forward pass; `swap` unavailable for
quantized models; only Qwen3 and Qwen2 architectures tested; single-seed GPU
results. The full list is in `docs/limitations.md`.

[0.1.0]: https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.1.0
