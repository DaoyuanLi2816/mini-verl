# Changelog

All notable changes to miniVERL are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-08-02

### Added

- `miniverl align` for explicit base → SFT checkpoint → teacher/reference →
  alignment → evaluation → Alignment Card workflows, plus an
  uncertainty-aware, versioned `miniverl pilot` decision aid.
- Policy-conditioned and frozen aligned-adapter teachers, pinned TRL DPO
  provenance, a versioned verifier-gated selector, deterministic tool-policy
  evaluation and privacy-safe JSON/Markdown Alignment Cards.
- A preregistered three-seed Alignment Lab result, 864 task-level records, a
  matched State × Supervision diagnostic, four data-bound figures, technical
  report, article and reproducible short demo.

### Changed

- Public positioning now treats OPD as a post-SFT teacher-student mechanism to
  justify with alignment, over-alignment, retained utility and cost evidence,
  rather than as a generic replacement for SFT.
- DPO Alignment Cards include the external pinned TRL training time, peak VRAM,
  optimizer updates and exact provenance instead of counting evaluation only.

### Results

- The common Qwen3-0.6B SFT checkpoint saturated the deterministic Minipolicy
  suite at 100% alignment and 100% retained tool utility across all three
  seeds. No continuation method improved it; completed regressions from
  continued SFT, standard OPD and verifier-gated OPD are retained.
- Verifier gating reduced queried positions from 100% to 46.8% and mean GPU
  time from 76.7 to 66.0 seconds without improving quality. The matched signal
  diagnostic found only 0.0251% fresh soft probability mass beyond argmax, so
  the pilot recommends not spending online teacher-query cost for this recipe.

## [0.4.0] - 2026-08-02

### Added

- Typed, mask-isolated padded update batches for SFT, offline KD and strict OPD,
  with deterministic length bucketing, per-trajectory normalization and exact
  plus top-k-and-tail objectives.
- A local typed role graph and one-base multi-adapter runtime for trainable
  actor, frozen teacher and optional frozen reference roles. Checkpoints export
  the student as a standard PEFT adapter.
- A preregistered eight-cell RTX 4080 runtime matrix, checksummed profiler
  summary, data-bound Pareto figure and a public immutable systems-benchmark
  teacher adapter.

### Changed

- `train.trajectory_batch_size` independently controls physical update-forward
  size (`1`, an integer or `auto`) while
  `train.gradient_accumulation_steps` remains the optimizer-group size.
- `models.runtime` explicitly selects the backward-compatible `dual_model`
  ownership path or `shared_backbone` when all policy roles use one pinned base.

### Results

- Batch-4 improved end-to-end throughput by 1.63× for dual ownership and 1.54×
  for shared ownership on the declared Qwen3-0.6B workload. Shared batch-4 used
  2.227 GiB peak reserved memory versus 3.035 GiB for dual, but was 10.1% slower.
- Identical trajectory and teacher-target digests held across all eight cells;
  all 12 preregistered loss, full-gradient and post-update-logit comparisons
  passed. No task-quality improvement or cross-hardware speedup is claimed.

## [0.3.0] - 2026-08-01

RecoveryBench release. The experiment is a scoped mechanism study of fresh
student-visited states, not an alignment benchmark and not evidence that OPD
replaces SFT.

### Added

- A deterministic SQLite recovery environment with structured retryable errors,
  executable recovery oracles, disjoint template splits and exact recovery
  metrics.
- Schema-v3 benchmark provenance for preregistration, teacher gates, frozen
  datasets, task-level artifacts, selected-position and wall-time budget views.
- A public, immutable NF4-qualified SQLite recovery teacher adapter plus
  recorded failed teacher candidates and preparation cost.
- Frozen three-seed RecoveryBench results, task-paired bootstrap analysis,
  data-bound SVGs and a deterministic six-page technical report.

### Changed

- Public positioning now distinguishes SFT competence-building from OPD as an
  online teacher-student mechanism whose transferred behavior depends on the
  teacher.
- Frozen-student offline KD records and validates the exact cold checkpoint,
  task schedule, adapter and tokenizer identities reused across budget views.

### Results

- Under eight equal continuation updates, frozen-student KD reached 23.2%
  strict success and 22.8% recovery after error, versus 10.9% and 9.1% for
  strict fresh-state OPD. The fresh-minus-frozen paired differences were
  -12.24 and -13.79 percentage points.
- Fresh OPD averaged 686.8 continuation seconds versus 52.1 for frozen KD.
  Querying 49.77% of model-generated positions did not reduce teacher backbone
  forwards or wall time.
- The nominal 50-second result is retained and explicitly labeled a
  cycle-capped wall diagnostic, not exact equal-time evidence.

## [0.2.6] - 2026-08-01

Small concurrency, lifecycle and privacy correctness release. No training
objective, environment, benchmark, model family, adapter revision or frozen
scientific result changed.

### Changed

- Training, evaluation, checkpoint save/load and destructive close now share
  one non-blocking trainer-operation ownership contract. Checkpoint loading is
  READY-only both before and after ownership acquisition.
- Evaluation records the actual prior model mode and restores it in `finally`
  after success or any rollout, serialization, diagnostics, metrics or event
  failure.

### Fixed

- `close()` can no longer release or null resources while evaluation,
  checkpoint save/load or training owns them; failed ownership attempts mutate
  no state or artifact, while later and repeated close remain safe.
- Portable artifacts recognize credential semantics across common key styles,
  structurally redact userinfo in HTTP/SSH/database URLs, and sanitize embedded
  absolute Windows, UNC and POSIX paths without rewriting public URLs,
  relative paths or mathematical slash expressions.

## [0.2.5] - 2026-07-30

Focused correctness release. No training objective, environment, benchmark,
model family, adapter revision or frozen scientific result changed.

### Changed

- Writable manifests now publish `ready` at construction, transition atomically
  to `running` immediately before training, and record
  `closed_before_training` when a new trainer is closed unused.
- Training owns private evaluation/checkpoint implementations; public calls
  cannot switch model mode or snapshot parameters during an optimizer update.
- Protocol-v2 final examples are environment-specific and verifier-format
  valid, while protocol-v1 remains byte-frozen.
- Automatic reports remain under the training run lock, and standalone
  evaluation acquires and transfers one lock before reading configuration or
  selecting, validating and loading a checkpoint.

### Fixed

- SQLite verification classifies non-finite and overflowing numeric strings as
  malformed instead of leaking `ValueError` or `OverflowError`; all built-in
  verifiers are fuzzed as total functions after reset.
- Portable artifacts redact semantic secret suffixes, authorization/cookie/
  session fields, URL userinfo, Windows paths with spaces, UNC paths and
  private POSIX/macOS paths without hiding useful immutable provenance.
- The PyPI-description generator rewrites both targets of nested linked images
  and rejects every remaining relative project target in generated Markdown
  and built wheel metadata.
- Release verification accepts an exact, tag-pinned banner expressed as
  Markdown or HTML while retaining source-URL and alt-text checks.
- A browser-only PyPI challenge can be deferred after public metadata, every
  pinned target, distribution hashes and attestations pass; a recovery workflow
  binds already-published artifacts to their original tag run before creating a
  GitHub Release.

## [0.2.4] - 2026-07-29

Correctness, adversarial-input, concurrency, packaging and privacy hardening
release. No training objective, environment, model family or scientific result
changed.

### Added

- A bounded strict-JSON boundary for model-generated tool calls, including
  duplicate-key, non-finite-number, oversized-integer, excessive-depth and
  excessive-member rejection.
- An explicit one-shot trainer lifecycle and cross-platform, process-safe run
  ownership acquired before mutable resume, overwrite, evaluation, report or
  export work.
- Exact submitted, canonical validated and runtime-resolved configuration
  provenance layers with checksums, plus a canonical portable redaction view
  for shareable artifacts.
- A generated PyPI long description with immutable release links, an
  extracted-sdist self-test gate and a concise public compatibility policy.

### Changed

- Protocol-v2 numeric verification consumes a complete finite answer and
  validates supported unit suffixes; protocol/verifier-v1 remains explicitly
  historical so existing artifacts are not reinterpreted.
- Calculator, JSON-navigation and SQLite boundaries now convert adversarial
  model inputs into bounded parse, tool or verification failures instead of
  leaking built-in numeric/serialization exceptions.
- Machine JSON and JSONL writers reject non-finite values before publication.
  Shareable reports, summaries and benchmark exports redact paths, identities,
  secrets and environment references by default.
- The source distribution now includes the repository fixtures required by its
  shipped tests, while the wheel remains runtime-only.
- Exact release-quality evidence has one generated machine-readable record;
  other documentation uses a stable floor or links to that record.

### Fixed

- A trainer instance can no longer train twice or admit two threads into
  training, and a failed second call mutates no run artifact.
- Competing processes can no longer mutate the same run or overwrite a run
  while another writer owns it; abandoned OS locks do not permanently block a
  later process.
- File-backed recipes preserve their submitted UTF-8 bytes and comments instead
  of labeling a normalized, path-resolved reserialization as verbatim input.
- PyPI documentation links and images no longer depend on relative repository
  paths or a moving branch for stable releases.

## [0.2.3] - 2026-07-29

Clarity and defensive-hardening release.

### Changed

- The generated GPU benchmark figure now reports measured 0% negative controls,
  scopes its title and strict-success label to the saturated v0.2 calculator
  task, separates protocol qualification from the quantitative axis, and keeps
  continuation time distinct from teacher preparation without presenting an
  unsourced preparation duration.
- The banner and bilingual onboarding now describe the device-name-agnostic
  single-GPU CUDA path and install CUDA PyTorch before optional training extras.
- Protocol-v2 prompt examples are generated from each environment's active
  `ToolSpec`; the immutable protocol-v1 prompt is unchanged.

### Fixed

- Legacy teacher caches can no longer bypass adapter or structural-tokenizer
  identity checks, and cache shard/index publication is crash-safe.
- Teacher caches persist `entries_per_shard`, allocate after the highest
  numeric indexed or on-disk shard suffix, and publish a copied pruned index
  before best-effort orphan cleanup.
- Tokenizer structural digests ignore source-location metadata, and failed
  model construction cannot leave an orphan partial run directory.

## [0.2.2] - 2026-07-29

Single-GPU portability and presentation release.

### Added

- A hardware-portability guide for personal NVIDIA GPUs, including honest
  starting points for 8–12 GiB, 16–24 GiB and 24–32+ GiB cards, OOM controls,
  and a reproducible hardware-result contribution path.
- A prominent PyPI destination in both READMEs and package metadata.
- Visual regression assertions that keep benchmark grid lines below axis labels
  and preserve the dark generated figure.

### Changed

- The supported Qwen3 recipe now uses model-agnostic run metadata and
  `dtype: auto`, selecting bf16 when available and fp16 on older CUDA cards
  such as Titan V. The pinned models, adapter, objective and budgets are
  unchanged.
- The repository is positioned as a personal single-GPU training stack rather
  than a 16 GiB-specific implementation. RTX 4080 numbers remain explicitly
  labeled as the only measured GPU evidence.
- The banner and data-bound protocol benchmark figure use a new dark visual
  system. Axis grids no longer cross tick labels, and protocol-incompatible
  0% controls are rendered as diagnostic states rather than zero-length bars.
- GPU workflow language now names the portable single-GPU recipe instead of a
  particular VRAM tier.

## [0.2.1] - 2026-07-29

Correctness, lifecycle safety and reproducibility release.

### Added

- Exclusive new-run creation, collision-resistant generated IDs, mutually
  exclusive `--resume`/`--resume-from`/`--overwrite` behavior, and atomic
  whole-run replacement with rollback.
- Atomic sibling-directory checkpoints with a manifest written last, SHA-256
  and size validation, a content digest, model/config/tokenizer identity, and
  state-based latest-checkpoint selection. Legacy v0.2 checkpoints remain
  readable and are explicitly labeled `legacy_unchecksummed`.
- A persisted, checksummed offline-KD dataset containing the exact trajectories,
  task order, token spans and provenance required for exact resume.
- Immutable startup manifests and atomic terminal manifests for completed,
  failed and interrupted runs, including actual optimizer, parameter, rollout,
  chunk-size, OOM, artifact and checkpoint state.
- Precise agent-event counters separating emitted and parsed calls, successful
  executions, execution errors, unknown tools, parse errors, repeated
  termination and final-answer format/verification outcomes.
- Versioned structural tokenizer identity, revision-aware tokenizer comparison,
  LM-head vocabulary compatibility checks and fail-before-mutation trainable
  weight validation.
- Explicit `parameter_version`, `rollout_policy_version`,
  `rollout_iteration` and `global_optimizer_step` records while retaining
  `policy_version` as a compatibility alias.
- Benchmark resume support and regression coverage for run collisions,
  checkpoint corruption, standalone evaluation, exact offline resume, manifest
  terminal states, OOM transaction boundaries, event metrics, tokenizer
  identity and model-state loading.

### Changed

- The default consumer-GPU recipe now uses the pinned, competence-gated
  protocol-teacher adapter; the previous raw-teacher payload is preserved
  byte-for-byte as an explicitly labeled diagnostic-control recipe.
- The primary benchmark figure leads with the supported OPD/SFT comparison and
  separates protocol-incompatible controls instead of labeling their measured
  0% strict success as a generic collapse.
- Installation examples distinguish the torch-free `miniverl` core from the
  `miniverl[train]` extra required for local optimization.
- OOM recovery now retries only the gradient-computation phase, restoring RNG
  and clearing partial gradients. The optimizer commit is non-retryable, so a
  single update cannot execute twice.
- Historical tool prompt/protocol v1 is frozen byte-for-byte for the published
  adapter and benchmark; corrected v2 examples are parser-valid and adapter
  competence gates are protocol-version aware.
- Per-token loss output now reports the optimized objective, divergence and
  sampled-token cross-entropy separately; SFT span metrics report CE instead
  of a meaningless zero divergence.
- Cache schema v2 preserves exact zero tails and ordered span types, honors
  checksum configuration, records complete teacher-adapter provenance and
  rejects lossy dtypes in exact full-vocabulary mode while retaining v1 reads.
- `ToolEnvironment.reset()` is now the authoritative initial observation and
  is called once per trajectory. Public trainer examples consistently use
  context managers.
- The benchmark figure labels non-training and protocol-mismatch controls
  directly instead of rendering misleading zero-length training bars.

### Fixed

- New, demo and benchmark runs can no longer append into or silently mix with a
  non-empty output directory.
- Standalone evaluation validates a checkpoint before loading and restores only
  model weights, never optimizer, RNG or teacher state.
- Missing or shape-incompatible trainable weights fail before any backend
  parameter is mutated.
- No-op and failed updates no longer advance the parameter version; replay can
  perform multiple successful commits while preserving one rollout version.
- Tool calls are no longer double counted, malformed final markers are not
  format-valid, verifier failures are not successes, and
  `max_parse_errors: 0` terminates at the first parse error.
- Hugging Face offline model loading now resolves a concrete cached snapshot
  before entering Transformers, preventing version-dependent adapter probes
  from issuing a network request.

## [0.2.0] - 2026-07-28

Protocol-aligned teacher support and scientifically explicit benchmark
accounting.

### Added

- Standard frozen PEFT teacher-adapter validation, loading and
  `miniverl export-adapter`, with base/tokenizer compatibility checks,
  checksums, run/checkpoint provenance and an optional tool-policy competence
  gate.
- An executable Qwen3-1.7B QLoRA protocol-teacher recipe using deterministic
  oracle traces in the same tool protocol as the student.
- Benchmark schema/config v2 with explicit common versus cold-start overrides,
  pre-allocation structured config diffs, resolved-config/checkpoint digests,
  mode-aware objectives, cumulative accounting and separate train/eval/wall
  timings. Existing schema-v1 measurements remain readable and unchanged.
- Policy competence metrics: strict success, diagnostic lenient success, valid
  tool-call rate/count, final-answer format validity and average turns.
- A deterministic temperature-gradient sweep across forward KL, reverse KL and
  JSD in near-uniform and sharply peaked regimes.
- CI compatibility rows for Transformers 4.51.x and 5.x, plus a disabled OIDC
  PyPI publishing job.
- A measured five-arm, two-seed RTX 4080 comparison and data-bound SVG: the
  protocol-trained teacher prevents the 0% raw/privileged-teacher collapse and
  reaches 100% on both seeds, tying rather than beating SFT.
- A public, checksum-validated protocol-teacher adapter on the Hugging Face Hub,
  pinned by the benchmark to an immutable revision with a local/offline config.
- Destructive trainer lifecycle tests, including weak-reference coverage and a
  measured sequential CUDA-allocation regression.
- A strict `--offline` contract shared by train, benchmark, standalone
  evaluation and adapter export, including cached pinned Hub adapters and
  socket-denial regression tests.
- A tag-only release supply chain that builds wheel and sdist once, publishes
  those exact artifacts with OIDC attestations, verifies public PyPI metadata,
  hashes, provenance and a clean install, then creates the GitHub Release.

### Changed

- Strict OPD freshness is the default and permits exactly one optimizer update
  per newly sampled rollout batch. Explicit replay is labeled
  `online_distillation_with_replay` and is never reported as genuine OPD.
- `loss.sampled_token_nll_weight` replaces ambiguous distillation uses of
  `loss.ce_weight`; its labels are explicitly the student's sampled tokens.
- `k == V` now bypasses epsilon tail smoothing and reduces to the exact
  full-vocabulary objective.
- Lower-bound and `T^2` documentation now distinguishes mathematical theorems
  from the epsilon-smoothed implementation and reverse-KL/JSD heuristics.
- Qwen3's verified minimum dependency is `transformers>=4.51,<6`.
- GitHub Actions are pinned to full v7 commit SHAs, and release validation runs
  the complete CPU scientific test suite.
- Source installation is the primary README path until a real PyPI publication
  exists.
- Published schema-v2 provenance replaces machine-local absolute paths before
  hashing or rendering artifacts.
- `OPDTrainer.close()` now destructively and idempotently releases target
  providers, scorer, runner, optimizer, teacher, student, environment and CUDA
  allocator state; public operations fail with `LifecycleError` after close.
- Cold starts and benchmark arms now run inside isolated function-level trainer
  contexts, with garbage collection and CUDA cache release between arms.
- New schema-v2 output separates declared scientific differences,
  runtime-resolution decisions and harness-only bookkeeping while retaining
  compatibility fields for existing readers.
- Hub teacher validation now returns the exact resolved local snapshot and
  PEFT loads only that directory, preventing a second independent Hub
  resolution after checksum validation.

### Corrected

- The legacy RTX 4080 result now has an explicit erratum: its shared cold start
  used `medium`, continuations/evaluation used `hard`, the old `controlled`
  block came from the base recipe, selected-token fields held only the final
  cycle and SFT's teacher-query ratio was not meaningful.
- Periodic evaluation now honors `eval.enabled: false`; the incomplete GPU
  attempt that exposed the defect was preserved and the full benchmark rerun.
- Hub teacher adapters now download and validate the miniVERL manifest and
  checksums at the pinned revision instead of losing the competence record.
- Legacy schema-v1 commands now point to their immutable source commit, and
  historical `peak_reserved_bytes` are explicitly caveated rather than
  silently rewritten.

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

- RTX 4080 (16 GB), `recipes/qwen_consumer_gpu_calc_raw_teacher.yaml`: 16 optimizer steps in
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

[Unreleased]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.2.6...v0.3.0
[0.2.6]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/DaoyuanLi2816/mini-verl/compare/37781ef0b00f3346d4b7b40fbe4d1c0ce1355063...v0.2.0
[0.1.0]: https://github.com/DaoyuanLi2816/mini-verl/tree/37781ef0b00f3346d4b7b40fbe4d1c0ce1355063
