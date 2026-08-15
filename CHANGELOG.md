# Changelog

All notable changes to miniVERL are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Product maturity and release qualification

- Added a strict, torch-free candidate and GPU qualification chain: a hosted
  job builds one wheel/sdist pair, and the RTX 4080 job installs and qualifies
  that exact same-run wheel with import and CLI origin checks.
- Hardened release artifact discovery against expired, fork, wrong-workflow,
  cross-run, duplicate, traversal and symlink inputs. Release publication now
  reuses the qualified candidate bytes and performs no distribution rebuild.
- Added a machine-readable maintainer-measured CUDA stack with exact top-level
  constraints while preserving flexible package dependency ranges.
- Extended canonical release-state checks to security support and current
  product prose, and documented v1 readiness, immutable upstream-profile
  lifecycle, runner safety and maintainer architecture.
- Added one non-publishing release-gate command that composes metadata,
  candidate integrity, exact qualification binding, quality, documentation and
  clean-install checks into a strict JSON summary.

## [0.10.0] - 2026-08-14

### Versioned compatibility profiles

- Added a closed, typed compatibility-profile registry with independent
  identity over the upstream pin, field-rule digest, native compiler, loss
  conformance and export versions. New plans, native runs, teacher caches,
  checkpoints and scale-out reports carry that identity.
- Added torch-free `profiles list/show/schema` and `compat explain/check`
  commands. They distinguish accepted, effective, locally reinterpreted,
  informational, unsupported and non-applicable fields without loading
  third-party entry points.
- Began the CLI domain split by moving profile and compatibility commands out
  of the root module without changing existing command names.

### Sampled-k1 policy-gradient OPD

- Added the closed `verl-opd-v0.8-single-gpu-pg-k1-v1` profile for the pinned
  verl v0.8 `k1` estimator and vanilla policy loss, with task rewards,
  reference KL, critics, multi-teacher and distributed execution rejected.
- Bound sampled token IDs, old/current actor log-probabilities, teacher
  sampled-token log-probabilities, policy version, tokenizer and estimator
  identity to fresh trajectories and caches.
- Matched the pinned upstream estimator, loss, metrics, gradient and tiny
  optimizer step; added profile-specific resume, export, materialization and
  doctor validation.

### Measured runtime evidence

- Published a Qwen3 PG systems record over 32 prompts and eight strict updates
  on one RTX 4080, including exact interruption/resume and a launchable
  materialized artifact bundle. This is not a task-quality comparison.
- Promoted SmolLM2-360M/1.7B to a full direct-GKD recipe over 32 prompts and
  eight updates, with 1.4961 GiB peak reserved VRAM, exact resume, PEFT reload
  and materialized export checks.
- Added a measured Ubuntu 26.04 WSL2 path on the same RTX 4080 covering plan,
  bounded probe, rollout, teacher scoring, one update and PEFT reload.

### Portable hardware records

- Added torch-free `hardware record` and `hardware validate` commands plus a
  generated strict schema that preserves measured, estimated and unknown
  states across profile, model, batching, memory, timing, resume and artifact
  evidence.
- Community records remain unreviewed and are never uploaded automatically;
  maintainer-measured publication requires explicit review and consent.

No frozen benchmark or task-level result changed, and this release does not
claim distributed execution, full verl compatibility or quality superiority.

## [0.9.1] - 2026-08-13

### Semantic contract repair

- Made actor and teacher dtype, quantization and attention settings explicit in
  the verl-shaped profile, system plan, probe identity and native `RunConfig`.
- Added pinned existing-student-adapter validation and trainable PEFT loading,
  including base/tokenizer identity, safetensors payload checks and exported
  lineage.
- Separated logical strict-OPD batches from physical rollout and actor-update
  trajectory/token ceilings. `ppo_mini_batch_size` is now truthfully
  informational for direct GKD.
- Added a shared placement capability model. Unknown-size quantized roles now
  require proof instead of selecting impossible swap; executable plans cannot
  violate a known static runtime placement constraint.
- Published mutation-based field-effect evidence for all 68 executable,
  non-informational compatibility claims.

### Documentation

- Corrected CUDA onboarding to install the matching PyTorch wheel before
  `miniverl[train,cuda]` and removed the invalid QLoRA-plus-swap recommendation.
- Distinguished verl `forward_kl_topk` top-k IDs/log-probabilities and
  diagnostics from miniVERL's explicit `bucketed_topk_tail` K+1 objective.
- Split the current pure-OPD runtime and scale-out path from the legacy
  environment/PPO reward scaffold, centered the landing page on the measured
  v0.9 developer workload and archived the historical project log.

No frozen benchmark, task-level result, model revision, algorithm or
distributed-execution claim changed.

## [0.9.0] - 2026-08-13

### Measured developer workload

- Published a checksummed RTX 4080 systems workload over 32 distinct consumed
  prompts, 64-token responses and eight QLoRA updates. The data-bound figure
  reports steady-state phase time, labelled throughput and 3.1914 GiB peak
  reserved VRAM without a task-quality or method-comparison claim.
- Fixed Parquet resume to reconstruct the saved row/epoch cursor. A real
  Qwen3 interruption after update four now reproduces uninterrupted
  trajectories, adapter and optimizer tensors byte for byte; all training
  state fields match apart from the intentionally run-specific resolved-config
  digest.
- Qualified the Apache-2.0 SmolLM2-360M/1.7B pair with a pinned one-update
  compatibility smoke covering tokenizer identity, rollout, teacher scoring,
  actor update and PEFT reload. This is not a second full recipe or quality
  benchmark.

### Immutable plan/run workflow

- Added deterministic `plan --out plan.json` artifacts that bind the source
  YAML, ordered overrides, accepted compatibility matrix, immutable model
  revisions, scanned Parquet hashes/schema/rows, exact native config and
  weight-free physical recommendations.
- Added `run --plan` with fail-closed plan, source, data and minor-version
  validation. The plan digest is carried by run manifests, teacher caches and
  checkpoints; direct `run --config` remains supported.

### Bounded hardware probe

- Implemented explicit `plan --probe` calibration with sequential role loading,
  tiny rollout/teacher-score/selected-position-backward phases, exact cache
  identity, zero optimizer updates and post-release CUDA-memory verification.

### Transactional scale-out materialization

- Added `miniverl bridge materialize` for exact local or downloaded student and
  teacher snapshots. It rejects moving revisions and unsafe trees, validates
  model/tokenizer/PEFT/Parquet/top-k inputs under the pinned verl commit, hashes
  every copied or merged file, and publishes through a rollback-safe staged
  directory replacement.
- New pure-OPD exports correctly keep both base-model loadability flags false.
  `launch.sh` replaces the fail-closed template only after the pinned upstream
  config merge and bounded sequential model-load smoke pass; distributed
  execution remains explicitly untested.
- Teacher-adapter merging requires an explicit flag, never mutates the source
  base, and records base, adapter, software, output and licensing provenance.

### Verl config UX

- Added safe trailing Hydra-style overrides and repeatable plain/JSON
  `--overrides-file` inputs with deterministic base → files → `--set` →
  trailing precedence. The compiled report preserves every duplicate, source,
  prior value and final effective value without evaluating shell text or
  interpolation.
- Added value-bound approval metadata for the packaged profile. External
  configs, or packaged profiles whose high-risk values drift, require explicit
  `--accept-local-reinterpretations` before `run`; unsupported algorithm and
  distributed fields still fail closed.
- Classified all 82 leaves in an Apache-2.0-provenanced fixture derived from
  the pinned official verl v0.8.0 Qwen3 OPD example. Harmless logging and
  compile hints are informational, while policy-gradient and enabled FSDP
  offload semantics remain unsupported.

### Fixed

- Made the PyPI release verifier require the durable product, project and
  policy links that the long description actually promises, instead of two
  historical research links intentionally removed from the v0.8.1 landing
  page. Every remaining repository link must still be release-pinned and
  reachable.

## [0.8.1] - 2026-08-12

### Product surface

- Reframed the English, Chinese and PyPI landing pages around the bounded
  single-GPU verl-style OPD workflow, with a responsive architecture diagram,
  a concise research index and a dedicated migration guide for verl users.
- Made the packaged Qwen3 profile use familiar upstream-shaped `vllm` engine
  values while retaining explicit `locally_reinterpreted` classifications for
  the sequential local-HF runtime.
- Aligned the banner, package metadata, CLI and citation summary without
  expanding the supported algorithm or implying distributed execution.

## [0.8.0] - 2026-08-12

### Single-GPU verl v0.8 OPD runtime

- Added the typed `verl-opd-v0.8-single-gpu-v1` configuration profile and an
  offline compiler plus weight-free `plan` and executable `run` commands.
  Resolved YAML and repeatable dotted overrides compile into deterministic
  field-by-field compatibility reports.
- Unsupported policy-gradient OPD, task-reward mixtures, KL penalties,
  multi-generation, multi-teacher and distributed dimensions fail closed.
  Engine/resource fields are labelled as local reinterpretations rather than
  upstream-exact behavior.
- Added first-class bounded verl Parquet prompts, padded local-HF rollout,
  exact response-only selection, current-policy/teacher/cache bindings and the
  pinned verl `forward_kl_topk` loss with token-mean scalar/metric/gradient
  conformance.
- Added a packaged Qwen3-0.6B/1.7B NF4 recipe and standard PEFT export. One RTX
  4080 runtime-conformance run completed its first update in 12.0224 seconds at
  3.1758 GiB peak reserved VRAM. No alignment-quality comparison was run.
- Added OPD v2 import/export. Imports publish a canonical prompt profile without
  inventing an environment or reward; exports preserve PEFT, teacher identity,
  Parquet bytes and pure OPD overrides without a reward scaffold. Missing base
  snapshots and teacher-adapter materialization remain explicit launch blockers.

## [0.7.1] - 2026-08-11

### Product correction

- Reordered the English, Chinese, PyPI and documentation landing pages around
  the installable single-GPU runtime, its hardware boundary, the pinned verl
  artifact bridge and measured systems evidence. Research studies and their
  negative results remain intact under Research Notes rather than preceding
  the quickstart.
- Updated package metadata, CLI help and diagnostics to describe the current
  single-GPU alignment and distillation runtime without claiming the planned
  verl-style OPD execution layer already exists.
- Added `miniverl evidence show/validate alignment-external-v1` and
  `miniverl pilot --builtin-study alignment-external-v1`. The wheel now carries
  the typed result, schema, preregistration and all 512 task-evidence rows with
  byte-bound validation, so the primary pip journey no longer depends on a Git
  checkout.
- Carried the corrected post-v0.7.0 evidence digest labels into a new immutable
  stable release without changing the v0.7.0 tag or any frozen scientific
  result.

## [0.7.0] - 2026-08-10

### External Alignment Gate result

- The first preregistered external-alignment study terminated at starting-
  checkpoint selection. Two declared lineages and eight candidates all
  measured 0/64 retained JSONNav utility against the unchanged 20% floor. No
  checkpoint was selected; teacher qualification, continuation SFT/DPO/KD/OPD
  training and the reserved final test did not run. This is a study-design and
  precondition finding, not a post-training method comparison.
- Added a schema-valid early-stop result and 512 privacy-safe JSONNav
  selection rows. `miniverl pilot --study-result ...` returns
  `do_not_continue_this_study` and `insufficient_evidence` without turning one
  stopped study into a universal method recommendation.
- Preserved the original fallback selection artifact and published a corrected
  lineage-only projection plus correction manifest. The primary and fallback
  selection manifests are disclosed as separately generated but byte- and
  task-identical, not independent samples.
- Granite Guardian values are explicitly unqualified diagnostics. Granite and
  PairRM qualification, PairRM method preference and teacher qualification are
  `not_run`; the necessary retained-utility failure does not depend on them.
- Added a generated checkpoint gate matrix and study-flow diagram with mobile
  layouts, plus the external-study page, English/Chinese release framing and
  browser visual coverage.

### Foundation and artifact hardening

- The bridge validates bundle trees before opening content; refuses symlinks,
  reparse points, non-regular or escaping entries and bounded-tree violations;
  and distinguishes complete, incomplete and uninspected privacy checks.
- Dataset extension sidecars bind source digest and row count, conversion
  revalidates source identity before publication, and row provenance uses
  bounded contiguous runs.

### Fixed

- A reward scaffold saved with a UTF-8 byte-order mark is no longer reported as
  a syntax error. CPython strips the BOM when it reads a source file, so such a
  scaffold imports normally; the static checker handed the leading `U+FEFF` to
  `ast.parse`, which rejected it. The failure was fail-closed — a legitimate
  file was refused, nothing unsafe was accepted — and a BOM still cannot hide a
  top-level call. Found while verifying the published v0.6.3 wheel on Windows.

## [0.6.3] - 2026-08-05

### Security

- `miniverl bridge doctor` no longer imports the inspected bundle's reward
  scaffold. Up to 0.6.2 it used `importlib` `exec_module`, so any top-level
  statement in an untrusted bundle ran with the user's privileges as soon as
  they asked for a diagnosis, while the report described the result as a
  "side-effect-free import". The scaffold is now parsed with `ast.parse` and
  verified statically, reporting `not_present`, `syntax_valid`,
  `interface_shape_verified` or `trusted_dynamic_import_verified`. Every
  definition-time position is audited, not only top-level statements: class
  bases, `metaclass=` and other class keywords, parameter and return
  annotations, annotated-assignment annotations, type-parameter bounds,
  decorators and call-valued defaults all run when a module is imported, so
  `class Hidden(exploit())` is rejected rather than reported as verified.
  Keyword-only parameters are checked too, so a required keyword-only
  `extra_info` — which raises `TypeError` on verl's three-argument call — no
  longer passes. Source bytes, AST nodes, AST depth and finding count are
  bounded, imports are listed with `import_runtime_safety: not_verified`, and
  relative bundle-local imports are refused.
- `bridge doctor` separates what a bundle *says* from what this process
  *recomputed*. `SHA256SUMS` ships inside the bundle it describes, so anyone who
  edits a claim can reseal it; matching hashes prove internal consistency only.
  Bundle testimony now appears under `bundle_declared_claims` with a
  `provenance_trust` level of `unsigned_self_consistent`, the top-level flags
  reflect only locally recomputed results, and `--require-verl` performs the
  upstream OmegaConf parse and structured merge locally instead of comparing an
  installed commit id.
- Portable metadata privacy ran one absolute-path regex, so a manifest holding
  an API key, a bearer token or a database URL with inline credentials passed.
  Structured JSON/YAML is now walked so a finding can name a JSON path,
  unstructured text reports a line number, the scan is bounded by file size,
  total bytes and finding count, and matched text is still never reported. The
  status is named `heuristic_passed`/`heuristic_failed` because it is a
  detector, not de-identification proof.
- An extension sidecar that exists but does not validate now fails the
  conversion instead of being treated as absent. Schema version, exact
  namespace, a `rows` mapping, canonical integer keys within the source row
  count, JSON-compatible values, unknown top-level fields and optional
  `source_sha256`/`source_rows` binding are all checked; sidecars published by
  0.6.0–0.6.2 still read.
- `convert-dataset` streams record batches through a `ParquetWriter` instead of
  calling `read_table(...).to_pylist()`, which materialized the entire dataset
  before converting a row. The output schema is derived once from the source
  schema, so an optional nested field that appears only in a later row group
  cannot produce a second incompatible schema. Strict conversion remains
  complete-or-nothing and now stops before reading the next row group.
- `scripts/check_text_integrity.py` fails CI on GBK-mangled UTF-8 punctuation,
  U+FFFD and unintended byte-order marks. The release checklist's `— not
  applicable` and the changelog's `base → SFT` arrows are repaired.
- Added `miniverl bridge doctor --trust-and-import-reward-code` for bundles you
  produced yourself. It warns before executing anything, reports
  `untrusted_code_executed: true`, and does not claim to be a sandbox.
- `import-verl` and `convert-dataset` reject any overlap between an input file
  and their intended output family before taking a reservation, covering exact,
  relative, symlink, hard-link and Windows case aliases. Previously `--out`
  could name the source config: a successful import overwrote it and a rejected
  import deleted it while publishing the rejection report. `--overwrite`
  replaces a previous output family and never authorizes destroying an input.

### Added

- Adapter safetensors are validated past the header. dtype and shape byte
  arithmetic, offset ordering, contiguity and full coverage of the data segment
  are checked, then every tensor is materialized through the official reader.
  Levels are `not_present`, `header_only`, `payload_structure_validated` and
  `tensor_materialization_validated`. A file declaring a 4x4 F32 tensor with no
  payload previously passed. The structural pass needs no optional dependency,
  so a torch-free install reaches `payload_structure_validated` and reports
  `official_reader_status: dependency_missing` instead of implying the file is
  broken; `--require-adapter-payload` demands the strongest level and is
  therefore not satisfied without the official reader.
- `miniverl convert-dataset --allow-rejected-rows` opts into a partial dataset.
  Conversion is otherwise complete-or-nothing, and a partial report carries
  `complete_dataset_conversion: false`, `lossless_for_accepted_rows: true` and
  the output-row-to-source-row index map.
- `release-state.yaml` and `scripts/release_state.py` give every public
  stable/development version claim one canonical source, gated in CI.
- Dedicated 390px layouts for `consumer-runtime-v1-pareto`,
  `cost-quality-pareto`, `fresh-vs-frozen` and `recovery-success`.

### Changed

- Conflicting miniVERL extension data across `miniverl_extensions`, the
  conversion sidecar and `extra_info.miniverl` now fails the conversion instead
  of silently preferring one source. Canonical-equal duplicates are accepted and
  recorded. Diagnostics name the row and the locations, never the values.
- Parquet bounds are enforced while reading. The dataset scan streams row groups
  through `iter_batches` restricted to string-bearing columns and stops at
  `max_rows`/`max_bytes`; schema validation reads the footer only. Both
  previously materialized the whole table first. The scan reports
  `files_inspected`, `row_groups_read`, `rows_scanned`, `rows_total` and
  `bytes_scanned`.
- The bridge output guarantee is stated as transactional publication with
  in-process rollback rather than atomic multi-file publication, which it never
  provided across `kill -9`, kernel panic or power loss.
- `CITATION.cff` describes the current scope: an auditable single-GPU alignment
  and distillation runtime with comparable SFT, DPO, KD and OPD arms,
  shared-backbone role switching and a bounded verl artifact bridge.

### Fixed

- The mobile readability exemption list is empty: every public figure is
  enforced at the 11px floor at 390px.
- The docs version selector, both READMEs, `PYPI.md` and the quality record no
  longer disagree about which release is stable.

## [0.6.2] - 2026-08-05

### Added

- `miniverl bridge doctor --require-tokenizer-load` fails unless the bundle's
  tokenizer actually loads from its local snapshot with `local_files_only=True`
  and `trust_remote_code=False`, and its versioned structural identity,
  vocabulary size and special tokens match the source run where the manifest
  records them. The network is never contacted.
- `miniverl bridge doctor --scan-dataset-text` adds a bounded heuristic scan of
  string-like Parquet fields for URL userinfo, private-key blocks, access-key
  ids, bearer tokens, credential assignments, absolute local paths and
  repeatable `--sentinel` values. It reports detector category, split, column
  and row index only, never the matched text, and records whether it ran
  `full` or `sampled`.
- `--overwrite` on `import-verl` and `convert-dataset`, required before any
  existing output family is replaced.
- Dedicated mobile layouts for both Alignment Lab charts, selected through
  `<picture>` at narrow viewports, plus a regression fixture that reproduces
  the v0.6.1 metric-coverage header collision.

### Changed

- Metric coverage is now an accessible, responsive HTML table with one
  column-level scope statement instead of an SVG whose long headers collided
  and whose two rightmost columns repeated one identical value in six rows.
- `bridge doctor` reports tokenizer verification as `not_present`,
  `metadata_only`, `loadable_local_snapshot` or `structural_identity_verified`
  instead of treating filename and digest presence as compatibility.
- `bridge doctor` reports `portable_metadata_privacy`,
  `dataset_content_privacy` and `model_weight_privacy` separately.
  `not_inspected` is never translated into `passed`.
- The browser visual gate measures each figure's real rendered bounding box in
  the viewport under test, inspects every visible SVG `<text>` node rather than
  only tagged ones, detects header-to-header and header-to-data collisions and
  label-to-mark occlusion, checks responsive tables and card labels, and fails
  when visible chart text renders below 11 px at 390 px.

### Fixed

- Unresolved `${...}` interpolation can no longer reach a runnable recipe. One
  recursive audit covers source fields, explicit command-line choices and the
  generated recipe. Informational fields may stay unresolved but are labelled
  `unresolved_informational_only` and never enter executable output.
- Bridge outputs are stem-specific and transactional. `--out foo.yaml`
  publishes only `foo.yaml` or `foo.template.yaml` plus one
  `foo.import-report.json`; dataset conversion publishes its Parquet, sidecar
  and report as one family. Each invocation takes an exclusive per-stem
  reservation, refuses to start on a collision, stages every file and rolls
  back on failure, so one invocation's report can never be paired with
  another's artifact.
- Raw-HTML figure sources in Markdown pages now use the relative paths MkDocs
  requires; the visual gate fails when a figure does not actually load.

## [0.6.1] - 2026-08-03

### Added

- Deterministic real-browser documentation gates across five representative
  pages and four viewports, with overflow, SVG bounds, label collision,
  readability, table and responsive-bridge assertions plus screenshot
  artifacts.
- Responsive desktop and mobile bridge diagrams that separate the verified
  local runtime, portable bundle and pinned upstream smoke from explicitly
  untested distributed execution.

### Changed

- The Alignment Lab case study now uses three data-bound forest/matrix figures
  that show every measured seed, preserve not-applicable query ratios and make
  the limited sandbox-safety coverage explicit without changing frozen data.
- The documentation uses pinned Material 9.7.7 with stable/development paths,
  search, dark/light modes, copy controls and a task-oriented landing page.
- The English and Chinese READMEs are shorter product guides with one scoped
  evidence summary and direct Align, Distill locally and Scale out paths.

### Fixed

- `import-verl` now classifies field semantics, fails closed with a
  non-executable template when data, teacher, objective or schedule choices are
  unresolved, accepts finite scientific-notation strings and validates every
  runnable recipe before atomic publication.
- `export-verl` now reports artifact completeness, upstream parse/load smoke,
  reward implementation, launchability, distributed execution and algorithm
  parity independently; its fail-closed scaffold emits `launch.template.sh`
  and is never described as ready to launch.

### Verified

- Official verl `v0.8.0` commit `7aed6b230776f963fa09509c10d9c3a767d1102c`
  still passes the bounded parse/load smoke. Distributed execution and
  miniVERL OPD-to-PPO semantic parity remain untested and unclaimed.
- Every frozen calculator, RecoveryBench, Consumer Runtime, Alignment Lab and
  bridge-smoke JSON/JSONL artifact remains byte-identical.

## [0.6.0] - 2026-08-03

### Added

- A miniVERL-defined compatibility Level-3 bridge for the fail-closed
  `single-gpu-online-distillation-v1` profile, pinned to official verl `v0.8.0`
  commit `7aed6b230776f963fa09509c10d9c3a767d1102c`.
- `import-verl`, bidirectional prompt-Parquet conversion, `export-verl` standard
  artifact bundles and `bridge doctor`, with exact pin, PEFT/safetensors,
  tokenizer, data, reward-scaffold, privacy and hash checks.
- Five versioned, evidence-bound community recipe records plus
  `benchmark --export-community`, schema/privacy/digest validation, a static
  documentation site and launch materials.

### Changed

- The README now describes miniVERL as single-GPU prototyping for one
  documented subset of verl-style online post-training and distinguishes
  artifact/config interoperability from untested distributed execution.

### Verified

- A Python 3.12 smoke installed the exact pinned verl source (observed package
  version `0.8.0.dev0`), parsed the official and exported OmegaConf shapes,
  loaded standard PEFT/safetensors and both Parquet splits, imported the safe
  reward scaffold, and verified privacy plus every bundle hash. Ray,
  FSDP/Megatron, vLLM/SGLang and distributed training were not run.

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

[Unreleased]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.6.3...v0.7.0
[0.6.3]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/DaoyuanLi2816/mini-verl/compare/v0.5.0...v0.6.0
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
