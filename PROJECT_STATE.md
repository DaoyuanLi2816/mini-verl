# PROJECT_STATE

Living build log for **miniVERL** (`mini-verl` / `miniverl` / CLI `miniverl`).
A checkbox is not evidence: every completed item names the command that was run
and what it printed.

Last updated: 2026-08-06.

Canonical release state: stable `v0.6.3` (`005a4549da713716e64c3ae80ff55fb131519f79`), development `0.7.0.dev0`.
Every public version claim is generated from `release-state.yaml` and gated by
`python scripts/release_state.py --check`.

## v0.7.0 External alignment evidence — IN PROGRESS

Branch `v0.7.0-foundation`, based on `0bd194600aeb65b90eadd14bfa1ec313aa2a9c36`.
Not yet pushed. Commit author is
`Daoyuan Li <94409450+DaoyuanLi2816@users.noreply.github.com>` on every commit,
matching the 9:1 dominant identity since v0.6.0.

### Phase A progress

| item | state |
| --- | --- |
| 7.1 bundle-tree preflight | **done** (`71863b6`). Reproduced on Windows with a junction: `_check_hashes` walked into an outside directory and the metadata scan reported `semantic_secret_key` against a file the bundle did not contain. `preflight_bundle_tree` walks with `lstat` only and refuses symlinks, reparse points, non-regular files, escaping entries and trees over bounded file count / bytes / depth. Runs before any open; a refused bundle reports every check as `not_inspected`, never `failed`. 13 regressions; symlink cases skip locally for privileges and run on Linux CI |
| 7.2 privacy completeness | **done** (`b1e5ab7`). `heuristic_passed_full` / `heuristic_failed` / `heuristic_incomplete` / `not_inspected`; every gap records file and reason, bounded. `--require-complete-metadata-scan` fails on incomplete. 8 regressions |
| 7.6 text and CI cleanup | **done** (`b1e5ab7`). A mis-decoded multiplication sign is one character, not the three-byte CJK leaders the v0.6.3 gate looked for: `U+00D7` is two UTF-8 bytes, so reading them as GBK yields a single `U+8133`. Two CHANGELOG.md lines survived the release that way — the State-by-Supervision diagnostic name and the two batch-4 speedup figures. Leaders extended to the Latin-1 range (the misreadings of `U+00D7`, `U+00A7` and `U+00A6`), both lines repaired, 3 new gate cases. All four tensor-to-float warnings in `test_chunked_equivalence.py` detached. The gate then caught this very table describing the damage with the damaged characters, which is the behaviour we want |
| 7.3 sidecar v2 binding | not started |
| 7.4 source identity during conversion | not started |
| 7.5 residual O(N) metadata | not started |
| 7.7 quality provenance | not started |
| version bump to `0.7.0.dev0` | not started |

Local suite at `b1e5ab7`: 1853 passed, 6 skipped, 6 deselected. Ruff, format and
mypy clean.

### Frozen baseline recorded before any v0.7 work

All eleven artifacts match their v0.6.3 digests, including the calculator at
`53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`.

### Feasibility verified before starting

RTX 4080 16 GiB with torch 2.13+cu130; Qwen3-0.6B and Qwen3-1.7B already in the
local HF cache; `google/IFEval`, `natolambert/xstest-v2-copy` and
`allenai/reward-bench` reachable. No v0.7.0 tag exists, so the release number is
uncontested.

Phase A is complete and merged as `0be43b0` (PR
[#49](https://github.com/DaoyuanLi2816/mini-verl/pull/49)).

### Phase B progress — branch `v0.7.0-external-alignment`

| item | state |
| --- | --- |
| endpoint registry | **done** (`c020b14`). `benchmarks/external-alignment/registry.yaml` pins four ungated endpoints with 40-hex revisions, licenses and redistribution decisions, every fact read from the Hub API on 2026-08-08 |
| IFEval verifier | **done** (`c020b14`). Independent implementation of all 25 instruction types; 0 of 834 instructions unscored on the real 541 rows. `number_sentences` and `response_language` are marked `approximate` |
| XSTest refusal classifier | **done**. Deterministic three-label classifier; over-refusal and appropriate-refusal reported separately, never averaged |
| record schema | **done**. `TaskRecord` refuses an evaluated row without a score and a non-evaluated row *with* one; generated text is never stored, only its digest |
| registry validator | **done**. Rejects a gated source, an unpinned revision, a judge over 3B, a gated judge, an unqualified model evaluator, or a missing category |
| pinned-source network tests | **done**. Six checks against the real Hub confirm 541 IFEval rows, XSTest 250/200 across 18 types, JBB 100 behaviours, RewardBench 2985 across 23 subsets |
| Granite Guardian harmful judge | not started |
| PairRM preference judge | not started |
| CLI surface, offline fixtures | not started |
| governance doc, THIRD_PARTY_NOTICES | not started |

### Endpoint decisions and why

HarmBench, StrongREJECT and AdvBench are all **gated**; a download attempt on
2026-08-08 returned `DatasetNotFoundError` requesting access. Accepting dataset
terms is an authorization the maintainer gives on their own account, and a
gated source is not reproducible by a reader either. The harmful-compliance
category therefore uses `JailbreakBench/JBB-Behaviors` (MIT, ungated, 100
non-adaptive categorised behaviours). `meta-llama/Llama-Guard-3-1B` is rejected
for the same class of reason: `gated: manual`. The judge is
`ibm-granite/granite-guardian-3.0-2b` (apache-2.0, ungated, 2B).

That endpoint is **not** reported as HarmBench. HarmBench's official classifier
is a fine-tuned 13B model and StrongREJECT's reference judge is a paid API
model; both exceed the compute contract, so reusing the name would misdescribe
what ran.

### Next action

Phase B: Granite Guardian harmful-compliance judge and PairRM pairwise judge,
each with offline fixtures and a qualification path, then the CLI surface.

## v0.6.3 Security, artifact integrity and release-state hardening

| item | current state |
| --- | --- |
| starting state | `origin/main` and local `main` both at `c62e14ba4229f32b3db06b293c615e94654debc6`; tag `v0.6.2` at `bef9f0878eb3280f450aee3868b43d61f0726557`; working tree clean; no open pull request; source version `0.6.3.dev0`; branch `v0.6.3-security-artifact-hardening` cut from that commit |
| immutable baseline | all ten frozen result artifacts plus `docs/generated/verl-bridge-smoke.json` re-hashed at takeover and unchanged; calculator remains `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc` |
| reward-code execution | reproduced: a scaffold whose top level wrote `PWNED.txt` created the marker during `_check_reward`, and the report still said `side-effect-free import; scaffold intentionally not executed`. Fixed by `src/miniverl/bridge/reward_static.py`, an AST walk that never imports; levels are `not_present`, `syntax_valid`, `interface_statically_verified`, `trusted_dynamic_import_verified`, and the last one requires `--trust-and-import-reward-code` |
| safetensors validation | reproduced: a header declaring a 4x4 F32 tensor with zero payload bytes returned `(True, '1 tensor header(s)')` while the official reader rejected it with `file not fully covered`. Fixed by `src/miniverl/bridge/safetensors_check.py`; levels are `not_present`, `header_only`, `payload_structure_validated`, `tensor_materialization_validated`, with `--require-adapter-payload` for strict callers |
| source/output aliasing | reproduced: `import-verl --out <source> --overwrite` overwrote the source on success and **deleted** it on rejection while publishing the report. Fixed by `reject_source_output_alias`, which runs before any transaction and covers exact, relative, symlink, hard-link and Windows case aliases |
| conversion semantics | extension data in `miniverl_extensions`, the sidecar and `extra_info.miniverl` is now reconciled: equal content deduplicates, different content fails closed naming row and locations but never values. Invalid rows fail the conversion unless `--allow-rejected-rows` is given, and a partial report carries `complete_dataset_conversion: false` |
| Parquet bounds | reproduced by inspection: the scan called `pq.read_table().to_pylist()` before applying bounds and the schema check read every row. Both now use footer metadata and `iter_batches` per row group; an instrumented test asserts the requested row groups are exactly `[0]` when the bound is two rows into an eight-group file |
| release-state drift | `python scripts/release_state.py --check` reported ten disagreements at takeover, including README stable `v0.6.1`, docs selector `Stable 0.6.1 / Development 0.6.2.dev0` and `quality_floor ... at v0.6.1` inside the `release: 0.6.2` record. All ten are resolved and the gate runs in `ci.yml` and `release.yml` |
| canonical release state | `release-state.yaml` drives the package `__version__`, both READMEs, `PYPI.md`, the docs channel selector, `CITATION.cff`, the changelog comparison link, the release-checklist section, this file's header line and the quality record; a test drives the full `release 0.6.3 → stable 0.6.3 → main 0.6.4.dev0` transition and still reports the prose sections a person must write |
| mobile visual debt | the four-figure exemption list is empty: `consumer-runtime-v1-pareto`, `cost-quality-pareto`, `fresh-vs-frozen` and `recovery-success` each ship a dedicated 390 px layout selected by a `<picture>` media query, and the desktop SVG bytes are unchanged at their pinned SHA-256 values |
| transactional scope | wording is narrowed to transactional publication with in-process rollback; multi-file crash atomicity across `kill -9`, kernel panic or power loss is explicitly not claimed, and the versioned-directory design that would provide it is named as out of scope |
| adversarial audit | tampered bundles were re-sealed with self-consistent `SHA256SUMS` and still rejected: hostile reward code stops at `syntax_valid`, a truncated adapter stops at `header_only`, and conflicting pinned-verl or adapter-versus-config fields fail closed. `--require-verl` is not a back door into execution, and a failing rollback re-raises the original cause rather than masking it |
| defect introduced and fixed in this branch | the first safetensors implementation treated a *dependency gap* as evidence the file was invalid. The torch-free environment installs `safetensors` but not `numpy`, and `safe_open(framework="np")` needs `numpy`; the resulting `ModuleNotFoundError` was caught by a generic handler and reported as `the official safetensors reader rejected adapter_model.safetensors`, failing the model check and nine tests against structurally sound files. This is the mirror of the defect being fixed: the old code read "only a header" as "valid", the new code read "cannot check" as "invalid". `_materialize` now returns `materialized`, `unavailable` or `rejected`; `unavailable` keeps `payload_structure_validated` with `official_reader_status: dependency_missing` and a passing status, because the structural pass is dependency-free. The exact condition is a regression test |
| strict adapter option | `--require-adapter-payload` demands `tensor_materialization_validated`, so an absent official reader does not satisfy it; the same file still passes without the flag |
| local gates | ruff, ruff format and mypy clean; `pytest -m "not gpu and not network"` passes 1,757 with 2 skipped, 6 deselected at 86.02% branch coverage on commit `85a7eb3aee46109ced405a7cbfb2106edeb82516`; `pytest -m gpu` passes 5 on the RTX 4080; `pytest -m network` passes 3; `mkdocs build --strict` clean; the browser gate checks 28 rendered SVG instances across 1440x900, 1024x768, 820x1000 and 390x844 with zero exemptions; `python -m build` and `twine check` both pass. The two skips are symlink creation, which needs privileges on Windows; the hard-link and case-alias cases cover the same guard |
| external metadata | the GitHub repository description was changed from `Auditable on-policy distillation for tool-using agents on one personal GPU.` to `Auditable one-GPU alignment and distillation runtime with shared-backbone training and a fail-closed verl artifact bridge.`, and topics `alignment`, `llm-alignment` and `peft` were added; no existing topic was removed |
| integration | PR [#45](https://github.com/DaoyuanLi2816/mini-verl/pull/45) |
| release state | in progress on `v0.6.3-security-artifact-hardening` |

## v0.6.2 Bridge correctness and responsive visual hardening

### Starting state — 2026-08-04

| item | observed state |
| --- | --- |
| checkout | one worktree, no submodules |
| local HEAD | `64fd6d62087185091b2f90c2a5870a3f3c83836b`, identical to `origin/main` after `git fetch --all --prune` |
| remote baseline | `origin/main` = `64fd6d6` "Advance development to 0.6.2.dev0 (#41)"; `git describe` = `v0.6.1-1-g64fd6d6`; stable tag `v0.6.1` = `48b9e7d9231b5f6cd018f6e927f81df066258f17` |
| branch | `v0.6.2-bridge-visual-hardening` already existed locally at `64fd6d6`, unpushed; a duplicate empty local branch `v0.6.2-visual-bridge-correctness` also sat at `64fd6d6` |
| background GPU work | none running. `Get-CimInstance Win32_Process` matched no `python.exe`/`pythonw.exe`; `nvidia-smi` showed the RTX 4080 at 8% with only desktop/browser C+G clients and no compute process. No RecoveryBench `equal-wall-time` run was live, so nothing needed termination and no current artifact was at risk of being overwritten |
| GPU availability | NVIDIA GeForce RTX 4080, 16376 MiB total / 1867 MiB used, driver 596.49, CUDA 13.2 — available for the existing GPU suite |
| uncommitted work in the tree | one file, `src/miniverl/bridge/config.py` (+52/-2). Archived verbatim outside version control before any change |
| audit of that draft | it added a `${...}` regex, `_contains_interpolation`, `_reject_cli_interpolation` and redundant non-finite string guards, but left `Callable`, `suppress` and `RunLock` imported-unused, added a dead tuple-returning `_validate_positives`, and wired none of it into the import path. Its useful ideas are carried into the shared audit module; the half-wired scaffolding was reverted rather than committed |
| run locks / temp dirs | `runs/.miniverl-locks` and `artifacts/.miniverl-locks` contain no live lock metadata, and no partial publication directory was left behind |
| immutable evidence | all ten frozen result JSON/JSONL files hash exactly to their recorded values, including calculator `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`; `docs/generated/verl-bridge-smoke.json` baselines at `d5266adc0f3ec46cf29f96a2f4258f03d848d846e62cbbe3ccdf5e014909bda7` |

### Reproduced defects (recorded before any fix)

| defect | evidence on `64fd6d6` |
| --- | --- |
| Workstream A — unresolved interpolation reaches a runnable recipe | with every bridge choice supplied, `actor_rollout_ref.model.path: ${MODEL_PATH}` produced `status: accepted`, `generated_recipe_validated: True` and a recipe containing `model_id: ${MODEL_PATH}` at line 16. Only some numeric paths were guarded |
| Workstream B — shared, non-transactional output names | the same run wrote `import-report.json`, not a stem-specific name; template runs write `imported.template.yaml`. Two stems in one directory overwrite each other, and dataset conversion replaces the Parquet before its sidecar/report |

### Failing regressions committed first

`tests/unit/test_verl_bridge_interpolation.py`, `tests/unit/test_verl_bridge_outputs.py` and
`tests/unit/test_verl_bridge_dataset_outputs.py` were added against unmodified code and reported
**44 failed, 11 passed**, covering mapped-field/CLI/nested interpolation, informational labelling,
stem-specific naming, collision refusal, `--overwrite`, fault injection and concurrency.

### Implemented in this release

| workstream | outcome |
| --- | --- |
| A — interpolation | `src/miniverl/bridge/interpolation.py` holds one recursive audit over strings, lists, tuples and mappings. It runs on source fields (classified `exact`/`derived`/`requires_user_confirmation` fail closed, `informational_only` is labelled `unresolved_informational_only` and stays in the report), on explicit CLI choices before any path is reserved, and on the generated recipe plus its rendered bytes before publication. Detection is conservative: any `${` is a finding. Nothing is ever resolved from the environment. `1e-5` still parses; NaN/infinity stay rejected. No new runtime dependency |
| B — transactional outputs | `src/miniverl/bridge/publish.py` gives every invocation a stem-specific target family, a per-stem `RunLock` reservation, a collision check that runs before anything is modified, staging inside the destination and restore-on-failure. `import-verl` publishes `<stem>.yaml` **or** `<stem>.template.yaml` plus exactly one `<stem>.import-report.json`; `convert-dataset` publishes Parquet, sidecar and report as one family. `--overwrite` is required to replace; `--out` alone never implies it |
| C — Alignment Lab visuals | metric coverage became an accessible responsive HTML table with one column-level scope statement, replacing the SVG whose `Sandbox endpoint` / `External safety` headers collided and whose two rightmost columns repeated one value six times. The forest legend moved to its own footer band and every mean/seed value is printed in the left column instead of floating in the plot. The outcome matrix widened its method column, stacked each value above its bar and moved `—  not applicable` to the last column. Both charts gained dedicated 390 px vertical SVGs selected by `<picture>` at `max-width: 900px` |
| D — visual gate | `scripts/check_docs_visual.py` now measures each figure's real rendered bounding box in the viewport under test and re-renders the SVG at exactly that width. It inspects every visible `<text>` node rather than only `[data-role]` ones, detects header-to-header and header-to-data overlap and label-to-mark occlusion, keeps the viewBox-bounds and legend/plot checks, validates responsive tables and card labels, ignores inactive `<picture>` branches via `currentSrc`, and fails below 11 px at 390 px |
| E — tokenizer levels | `not_present` → `metadata_only` → `loadable_local_snapshot` → `structural_identity_verified`. A complete snapshot is loaded with `local_files_only=True` and `trust_remote_code=False`; structural digest, vocabulary size and special tokens are compared against the manifest identity when present, and any mismatch fails closed. `--require-tokenizer-load` refuses metadata-only bundles |
| F — privacy scopes | `portable_metadata_privacy`, `dataset_content_privacy` and `model_weight_privacy` are independent. `not_inspected` is never rendered as `passed`, and the CLI says so. `--scan-dataset-text` adds a bounded heuristic scan that reports category/split/column/row only, caps rows and bytes, records `full` vs `sampled`, and never reads safetensors as text |
| G — claims | README, Chinese README, PYPI.md, `docs/verl-bridge.md` and the exported bundle README describe a *verified artifact bridge* at miniVERL-defined Level 3 with `launchable: false`, distributed execution not tested and no OPD/PPO parity, and document the new naming, overwrite, tokenizer and privacy contracts |

### Evidence

| gate | result |
| --- | --- |
| new regressions | 55 pass (`test_verl_bridge_interpolation.py`, `test_verl_bridge_outputs.py`, `test_verl_bridge_dataset_outputs.py`) after implementation; the same files reported 44 failures before it |
| doctor levels/privacy | 19 pass in `tests/unit/test_verl_bridge_doctor_levels.py`, including no-network enforcement through the `deny_network` fixture |
| visual gate regression | the v0.6.1 SVG is preserved verbatim at `tests/fixtures/visual/legacy-metric-coverage-matrix.svg`. The corrected gate rejects it at both widths with `overlapping text: ['"Sandbox endpoint" x "External safety"']`; all four published figures pass with zero overlap, zero occlusion and no text under the floor |
| browser gate | `checked 28 rendered SVG instances across 4 viewports` at 1440x900, 1024x768, 820x1000 and 390x844 over `/`, `/alignment-lab/alignment-lab-v1/`, `/consumer-runtime/`, `/recoverybench/recoverybench-v1/` and `/verl-bridge/` |
| manual inspection | the 390 px and 1440 px screenshots were read directly. Two defects the automated gate could not express were found and fixed this way: the coverage caption collapsed to one word per line in card mode, and the outcome-matrix value labels touched their own seed marks and the next column's bar |
| CPU suite | 1652 passed, 6 deselected, 85.77% branch coverage against an 80% floor |
| static gates | `ruff check`, `ruff format --check`, `mypy src/miniverl`, `mkdocs build --strict` and the Markdown/link check pass |
| GPU suite | 5 passed on the local RTX 4080 (driver 596.49, CUDA 13.2); no scientific benchmark was rerun |
| network suite | 3 passed |
| packaging | `python -m build` and `twine check` pass. `miniverl-0.6.2.dev0-py3-none-any.whl` SHA-256 `e1cea1c2001dd96e14542d948024b920552473eef05cfbd0ba49bfbf270eabdf`; `miniverl-0.6.2.dev0.tar.gz` SHA-256 `780e96c0d11e53ac5c80dfc72ed331d2952b34ca4f30400fdbe149c3210da900` |
| clean installs | a clean core venv from the wheel reports `miniverl 0.6.2.dev0` with torch absent and passes `miniverl doctor`; a clean `[bridge]` venv resolves pyarrow 25.0.0 with torch still absent |
| extracted sdist | installed with `[dev,train]` into a clean venv outside the checkout, resolving `miniverl` from that venv: **1652 passed, 6 deselected**. The torch-free CI selection passes 1303 tests on the same extraction |
| immutable evidence | all ten frozen result JSON/JSONL files and `docs/generated/verl-bridge-smoke.json` re-hash to their recorded values after every change |

### Integration

| item | state |
| --- | --- |
| branch | `v0.6.2-bridge-visual-hardening`, pushed at `1161700c19379f6c1b49fe50d4408d4d96374143` |
| pull request | draft [#43](https://github.com/DaoyuanLi2816/mini-verl/pull/43) |
| CI | [`30975138358`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30975138358) green: core no-torch py3.10/3.11/3.12/3.13, cpu ml (torch), transformers `4.51.*` and `5.*`, training minimum py3.10 and training latest py3.13. `actionlint` runs on the 3.12 core job and passed |
| build | [`30975138346`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30975138346) green, including the extracted-sdist reproduction and the sdist-built wheel inventory match |
| docs | [`30975138352`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30975138352) green; 20 Linux screenshots uploaded as `docs-visual-screenshots` |
| pinned verl bridge | [`30975138353`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30975138353) green |
| screenshot inspection | the CI Linux screenshots were downloaded and read directly, not only trusted as green. The 390 px Alignment Lab mobile chart and metric-coverage cards render the same as the local Windows capture |
| release metadata | the version was finalized to `0.6.2`, `CHANGELOG.md` carries a dated `[0.6.2] - 2026-08-05` section with its compare link, `CITATION.cff` records `0.6.2` / `2026-08-05`, `PYPI.md` is regenerated tag-pinned to `blob/v0.6.2`, and `docs/generated/quality.json` records release `0.6.2` at 1652 CPU tests, 85.77% branch coverage, 5 GPU and 3 network |
| local release build | `python -m build` and `twine check` pass at `0.6.2`. These local artifacts were a sanity check only; the published distributions are built once inside the tag workflow |
| merge | PR [#43](https://github.com/DaoyuanLi2816/mini-verl/pull/43) was squash-merged; the release commit is `bef9f0878eb3280f450aee3868b43d61f0726557` after its message was rewritten so the sole contributor is the repository owner. Synchronized-main CI, build and docs runs [`30980343307`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30980343307) / [`30980343344`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30980343344) / [`30980343293`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30980343293) are green on that exact commit |
| tag | annotated `v0.6.2` resolves to `bef9f0878eb3280f450aee3868b43d61f0726557` |
| release execution | tag run [`30980579109`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30980579109) passed metadata validation, the full quality gate, a one-time build, OIDC Trusted Publishing with attestations, public PyPI verification, exact install and GitHub Release creation |
| published wheel | [`miniverl-0.6.2-py3-none-any.whl`](https://pypi.org/project/miniverl/0.6.2/), SHA-256 `38131c3de838b480017b2f97df3e43d53e760f272c516522a470d2812f8a3803`, 330906 bytes |
| published sdist | [`miniverl-0.6.2.tar.gz`](https://pypi.org/project/miniverl/0.6.2/), SHA-256 `bcd30863290c9c46c1e2c27c59d8ccfd3ec7e5934243f5bc86f7424d4f05333d`, 994642 bytes |
| independent public verification | the PyPI JSON API reports version `0.6.2` with a Markdown description pinned to `blob/v0.6.2`, and the integrity API exposes one Trusted Publisher attestation per distribution bound to `DaoyuanLi2816/mini-verl`, `release.yml`, environment `pypi`. The [GitHub Release](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.6.2) ships byte-identical distributions whose recomputed SHA-256 values match its `SHA256SUMS` and PyPI exactly. A clean Windows venv installed `miniverl==0.6.2` from `https://pypi.org/simple`, reported `miniverl 0.6.2` with torch absent, and exposes `--require-tokenizer-load`, `--scan-dataset-text` and `--sentinel` |
| version transition | `v0.6.2` is immutable and public; this state sync identifies subsequent development as `0.6.3.dev0` |
| release state | complete |

### Known limitation carried forward

The 11 px readability floor at 390 px is **not** enforced for four figures that
predate this release and have no narrow layout: `consumer-runtime-v1-pareto.svg`,
`cost-quality-pareto.svg`, `fresh-vs-frozen.svg` and `recovery-success.svg`.
The exemption is an explicit asserted set in `scripts/check_docs_visual.py`, is
printed on every run, and fails the gate if any other figure joins it.
Redesigning those v0.3/v0.4 figures is out of scope for this patch.

The committed `docs/generated/verl-bridge-smoke.json` is left byte-identical.
It was produced before tokenizer verification levels existed, and its bundle
ships only `tokenizer_config.json` — which the current checker correctly
classifies as `metadata_only` rather than the `status: ok` the old
presence-only check recorded.

## v0.6.1 Visual integrity and bridge correctness release

| item | current state |
| --- | --- |
| scope | representation and documentation UX were rebuilt without rerunning a benchmark; import/export bridge semantics now fail closed instead of implying runnable or equivalent execution |
| Alignment Lab visuals | the four non-data-bound scatterplots are replaced by a three-seed delta forest, an outcome/cost matrix and a metric-coverage matrix; not-applicable teacher ratios remain N/A, and the zero-variance sandbox checks are not plotted as a two-dimensional safety result |
| bridge diagram | responsive desktop/mobile layouts show solid arrows only across the verified local runtime, portable bundle and pinned `v0.8.0` / `7aed6b23` parse-load smoke, followed by a dashed arrow to prominent `Distributed execution: NOT TESTED` |
| import contract | every source field is classified as exact, derived, informational only, requiring user confirmation or unsupported; unresolved data/environment, teacher, objective or schedule semantics produce `needs_user_input` plus a non-executable template |
| export status | artifact completeness, upstream parse/load, model/data smoke, reward implementation, launchability, distributed testing and algorithm parity are independent flags; the fail-closed reward scaffold remains non-launchable and uses `launch.template.sh` |
| documentation | pinned Material 9.7.7 supplies stable/dev navigation, search, dark/light modes, command-copy controls and responsive tables/images; the landing page exposes Align, Distill locally and Scale out paths |
| visual gate | Playwright checks five pages at 1440x900, 1024x768, 820x1000 and 390x844 for overflow, SVG bounds, label collisions, readable labels, tables and the responsive bridge; final PR docs run [`30856925490`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30856925490) is green and its 20 Linux screenshots were manually inspected |
| validation | local ruff, format, mypy, actionlint, full non-GPU/non-network, available GPU/network, strict MkDocs, browser visual, package/Twine, clean-install, extracted-sdist, bridge end-to-end, privacy, Markdown/link and generated-byte gates pass; PR CI/build/bridge runs `30856925524` / `30856925481` / `30856925485` and synchronized-main CI/build/docs runs [`30857409920`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30857409920) / [`30857409926`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30857409926) / [`30857409918`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30857409918) are green |
| immutable evidence | no result JSON/JSONL or frozen bridge-smoke record changed; calculator SHA-256 remains `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc` and Alignment Lab result/task SHA-256 values remain `584752dccb91654109c357b8ebb12681a12a9c1476a9ba539dd35e4d860a22ef` / `8d7fc723436d7377d196fc44046d960e3cb7f0aa81e03d49ef05b627eb84630f` |
| integration | focused PR [#40](https://github.com/DaoyuanLi2816/mini-verl/pull/40) was squash-merged as exact release commit `48b9e7d9231b5f6cd018f6e927f81df066258f17`; annotated tag `v0.6.1` resolves to that commit |
| release execution | tag run [`30857762954`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30857762954) passed metadata, the full quality gate, one-time build, OIDC Trusted Publishing, public hashes/attestations, exact install and GitHub Release creation |
| published wheel | [`miniverl-0.6.1-py3-none-any.whl`](https://pypi.org/project/miniverl/0.6.1/), SHA-256 `ef9ce5378e43c0d833b782e431248a0838e3841ace76a8b3a08781dd28007918` |
| published sdist | [`miniverl-0.6.1.tar.gz`](https://pypi.org/project/miniverl/0.6.1/), SHA-256 `e5ffd7917035d1f3878b22415dd357cd47fe16b6c77ffdae260e0b85ad7e050f` |
| independent public verification | PyPI and GitHub Release expose identical file hashes; the PyPI integrity API exposes one Trusted Publisher attestation for each distribution, and a clean Windows Python 3.12 install from the public wheel reported `miniverl 0.6.1`, kept torch absent and passed core doctor |
| version transition | `v0.6.1` is immutable and public; this state-sync change identifies subsequent development as `0.6.2.dev0` |
| release state | complete; PyPI and [`miniVERL v0.6.1`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.6.1) expose the verified release, and the synchronized docs workflow publishes v0.6.1 at the stable root plus current main under `/dev/` |

## v0.6.0 Verified verl Bridge release

| item | current state |
| --- | --- |
| audited upstream | official stable verl `v0.8.0`, source commit `7aed6b230776f963fa09509c10d9c3a767d1102c`, tested with Python 3.12; the installed source reports `0.8.0.dev0` |
| compatibility contract | Levels 0-3 are explicitly miniVERL-defined; Level 2 is a fail-closed 14-field whitelist for `single-gpu-online-distillation-v1`; miniVERL-defined Level 3 exchanges standard HF/PEFT/safetensors/tokenizer/Parquet artifacts and a checksummed scale-out bundle |
| command surface | `import-verl`, `convert-dataset`, `export-verl`, `bridge doctor` and `benchmark --export-community` are implemented without importing torch in the core path |
| exact compatibility smoke | the pinned source installed successfully; OmegaConf parsed official and exported config shapes; PEFT, safetensors, train/val Parquet, reward import, privacy and all bundle hashes passed; the checksummed record is `docs/generated/verl-bridge-smoke.json` |
| distributed boundary | Ray, FSDP/Megatron, vLLM/SGLang and a full distributed run were not installed or launched; distributed execution remains explicitly `not tested` |
| community surface | four versioned recipe records bind existing measurements; the preference-teacher distillation entry is explicitly unmeasured and links an aligned-adapter scaffold that still requires a pinned preference-trained teacher; no external adoption is claimed |
| documentation | a static MkDocs site, bridge architecture diagram, launch article/demo/card copy/announcement/social preview and community submission guide are part of the release candidate |
| immutable evidence | the frozen calculator JSON remains required byte-identical at SHA-256 `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`; prior tags and published adapter revisions remain immutable |
| local release gates | ruff, format, mypy, actionlint, Markdown/link and generated-artifact checks pass; the full non-GPU/non-network suite passes 1548 tests with 6 deselected at 85.53% branch coverage; the available GPU and network gates each pass 3; package build, Twine, clean Python 3.10 wheel install and exact pinned-verl smoke pass |
| integration source | Verified verl Bridge PR [#36](https://github.com/DaoyuanLi2816/mini-verl/pull/36) was squash-merged as `0d43310cca47db828b10a0e12facb33e8f0fd371`; synchronized main CI, build and docs runs `30794655713`, `30794655722` and `30794655822` are green; the latest PR bridge smoke is `30794329109` |
| public documentation | GitHub Pages deployed successfully and returned HTTP 200 at `https://daoyuanli2816.github.io/mini-verl/` with the bridge documentation visible |
| release metadata | PR [#37](https://github.com/DaoyuanLi2816/mini-verl/pull/37) was squash-merged as exact commit `6cfbdbb7bbf5c6042def4cf154bfe3c3b6530eea`; annotated tag `v0.6.0` resolves to that commit |
| release execution | tag run [`30796058250`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30796058250) passed metadata, the full quality gate, one-time build, OIDC Trusted Publishing with attestations, public PyPI verification, exact install and GitHub Release creation |
| published wheel | [`miniverl-0.6.0-py3-none-any.whl`](https://pypi.org/project/miniverl/0.6.0/), SHA-256 `e5fbb99bf410c27d22d6959f9599dfa4fbac2e940dac63f55c9676f68264abd1` |
| published sdist | [`miniverl-0.6.0.tar.gz`](https://pypi.org/project/miniverl/0.6.0/), SHA-256 `2e1d85556875f6d23152220897ba919d8b82bda35c5c78fd48efffa7ec22909d` |
| independent public verification | PyPI and GitHub Release expose the same files and hashes; PyPI exposes one Trusted Publisher attestation bundle per distribution, bound to `release.yml`, environment `pypi` and commit `6cfbdbb7`; a clean Windows Python 3.10 install from `https://pypi.org/simple` reported `miniverl 0.6.0` and passed JSON `doctor` with the core path healthy and torch absent |
| version transition | `v0.6.0` is immutable and public; this state-sync change identifies subsequent development as `0.6.1.dev0` |
| release state | complete; PyPI, [`miniVERL v0.6.0`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.6.0) and the public documentation site expose the verified release |

## v0.5.0 One-GPU Alignment Lab release

| item | current state |
| --- | --- |
| phase boundary | v0.4.0 was released and state-synced to `0.5.0.dev0` before final Alignment Lab execution; v0.5 is now released, and this state sync advances main to `0.6.0.dev0` before the verified verl bridge begins |
| public workflow | `miniverl align` resolves base → SFT checkpoint → teacher/reference → alignment → evaluation → Alignment Card; `miniverl pilot` exposes versioned evidence, uncertainty, cost assumptions and a bounded method recommendation |
| alignment roles | policy-conditioned self-distillation, frozen aligned-adapter teaching and shared-backbone execution are supported; continued SFT, pinned TRL DPO, offline soft distillation, standard OPD and verifier-gated OPD are explicit methods |
| policy suite | Minipolicy v1 uses deterministic sandbox tools to check authorization, confirmation, instruction hierarchy, secret exclusion, safe refusal, benign completion and safe error recovery; no real destructive action is executed |
| preregistration | public revision 1.4 digest `71307dbfe9a5bb20c686307cafce8bd254c07af8b69c1bf1c6ec0dbf53a8cde0` freezes three seeds, the shared SFT checkpoint, four continuation updates, 48 final-test tasks, the gate and the disjoint seed-1234 recovery rule |
| final artifact audit | all 18 arms completed with 48 ordered paired final-test tasks each; 864 task-level records are retained; all arms share starting checkpoint `7304922281268a687dd1c75ba918e26c64c8207b5701db78c368afd20d80ae89`; strict OPD freshness, every verifier-gate decision, DPO provenance/cost and the two disjoint baseline segments pass publication checks |
| final result | the SFT checkpoint scored 100% alignment and 100% tool utility in every seed; DPO and offline soft distillation tied it; continued SFT averaged 94.4% / 88.9%, standard OPD 98.6% / 97.2%, and verifier-gated OPD 97.9% / 95.8% alignment / utility |
| preserved negative results | verifier-gated OPD seed 20260727 failed 3 safe-error-recovery tasks; continued SFT seed 20260801 failed 8; standard OPD seed 20260801 failed 2; no completed final arm was rerun, and the interrupted pre-evaluation continued-SFT construction remains preserved outside the headline result |
| State × Supervision | measured signal diagnostic `9e08129ba4cd9e460c189b94b4e421d881ba69e3938f02eac95d251f50c88788` finds 100% teacher-argmax/student-token agreement and 0.0251% fresh soft probability mass beyond argmax; it is not a separately trained hard-target result and no soft-target advantage is claimed |
| verifier-gated OPD | frozen `policy-critical-span-v1` reduced mean queried positions from 100% to 46.8% and GPU time from 76.7 to 66.0 seconds, but did not improve alignment or retained utility; selected positions are not teacher-backbone FLOPs |
| pilot decision | `alignment-pilot-v1` returns `insufficient_evidence` and the operational decision not to spend online teacher-query cost because the starting policy is already saturated and no continuation method improves it |
| frozen public artifacts | result `584752dccb91654109c357b8ebb12681a12a9c1476a9ba539dd35e4d860a22ef`; task rows `8d7fc723436d7377d196fc44046d960e3cb7f0aa81e03d49ef05b627eb84630f`; technical PDF `adbffa967f6b9a25d2cdb0cc4464a93c13db4615a1e91499585fb199285d980b`; 18 JSON plus 18 Markdown Alignment Cards |
| public teacher | the common Qwen3-0.6B SFT adapter is immutable at `DaoyuanLi/mini-verl-qwen3-0.6b-tool-policy-sft@7b98164f73e493c51f2ed3fca3169fea078f47f0`; starting checkpoint content digest is recorded separately above |
| demonstration assets | reviewed short CPU run completed the actual alignment path in 17.6 seconds; the committed 75-second recording plan includes pilot, stage graph, real update, typed token inspection, Alignment Card and export-ready checkpoint without hiding the toy 0% result |
| presentation | four data-bound dark SVGs passed native and 820-pixel inspection; banner and social preview passed raster inspection; the report PDF is deterministic, six pages, metadata-valid and visually inspected page by page |
| local release gates | ruff, format, mypy, actionlint, generated-artifact checks, Markdown links and package/twine checks pass; the full non-GPU/non-network suite passes 1508 tests with 6 deselected at 86.13% branch coverage; the RTX 4080 gate passes 5 and the network gate passes 3; fresh core and CPU-training wheel installs, a real no-network toy demo, and all 1508 tests from an isolated extracted sdist pass |
| immutable baseline | calculator benchmark remains byte-identical at `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`; v0.3/v0.4 tags, RecoveryBench artifacts and public adapter revisions remain unchanged |
| integration source | Alignment Lab PR [#33](https://github.com/DaoyuanLi2816/mini-verl/pull/33) was squash-merged as `f9dae54a203d303f6562101500343ead310d99e8`; release-metadata PR [#34](https://github.com/DaoyuanLi2816/mini-verl/pull/34) was squash-merged as `fd755ff351fe691531ed68b4af7793c4929ed89e`; annotated tag `v0.5.0` resolves to the latter exact commit |
| version transition | `v0.5.0` is immutable and public; this state-sync change identifies subsequent development as `0.6.0.dev0` |
| release execution | tag run [`30789267409`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30789267409) passed metadata, all release gates, one-time build, OIDC publication with attestations, public verification, exact clean install and GitHub Release creation |
| published wheel | [`miniverl-0.5.0-py3-none-any.whl`](https://pypi.org/project/miniverl/0.5.0/), SHA-256 `ff60cb747e2ad1fd74575dd8920b11b48c6c16cc97743e94f9583d162d18819c` |
| published sdist | [`miniverl-0.5.0.tar.gz`](https://pypi.org/project/miniverl/0.5.0/), SHA-256 `d3340e0526eb4b20bb9ee15960c27c740be0fdacb9a51b029627faa63ca5276d` |
| independent public verification | workflow artifacts, PyPI and GitHub Release reproduce both hashes; PyPI exposes one attestation bundle for each distribution; a refreshed clean Windows Python 3.10 install from `https://pypi.org/simple` reported `miniverl 0.5.0`, passed JSON `doctor`, and kept torch absent |
| release state | complete; PyPI and [`miniVERL v0.5.0`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.5.0) expose the same verified wheel and sdist |

## v0.4.0 Consumer Runtime release

| item | current state |
| --- | --- |
| phase boundary | v0.3.0 was released and state-synced to `0.4.0.dev0` before this work began; v0.4 must merge, release and advance main to `0.5.0.dev0` before alignment-lab experiments begin |
| implementation | typed padded batches support variable lengths, causal mask isolation, padding-free selected-position loss, per-trajectory normalization, deterministic length bucketing, SFT/offline-KD/strict-OPD and exact/top-k-plus-tail objectives |
| local role graph | typed actor, rollout, teacher, reference, verifier, target, update, evaluation and artifact roles map existing components without importing Ray, DataProto, FSDP placement or distributed APIs |
| shared ownership | `models.runtime: shared_backbone` loads one quantized base with trainable student, frozen teacher and optional frozen reference adapters; optimizer visibility, failure restoration, one-base loading, reference isolation and standard PEFT student export are covered by CPU/HF tests |
| public systems adapter | `DaoyuanLi/mini-verl-qwen3-0.6b-consumer-runtime-teacher@e277b92d8c1fdb76cd133f872f0ddd2c47a4ab8c` is immutable and independently re-downloaded; it is a runtime benchmark artifact, not a newly qualified teacher |
| preregistration | revision 1.2 at public PR head `e44584b04837a05b0dd834c7948666d843908486` retains eager attention and NF4 weights with FP32 compute after two explicitly non-headline diagnostics exposed nondeterministic SDPA and BF16 batch-shape drift |
| final measurement | all eight dual/shared × sequential/2/4/auto cells completed on one RTX 4080; trajectory and teacher-target digests match, and 12/12 declared loss/full-gradient/post-update-logit equivalence comparisons pass |
| performance result | dual batch-4 reached 3.866 trajectories/s at 3.035 GiB reserved (1.63× sequential); shared batch-4 reached 3.475 trajectories/s at 2.227 GiB (1.54× sequential and 26.6% less memory than dual batch-4, but 10.1% slower) |
| frozen artifacts | result `a302da31af99f1d29f1efd4e6b3dbeb6ea4ac956bba102ca8a1bee8dff0319eb`; profiler `66111cd7fc876cf1befea3297a1a51bcd99252c0bf8989c029381e1dc155a98b`; SVG `98645a668a7832423d28b621262292619615917f037adf7219ff1bf071fb2fea` |
| immutable baseline | calculator benchmark remains byte-identical at `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`; every v0.3 artifact and negative result remains unchanged |
| integration source | Consumer Runtime PR [#30](https://github.com/DaoyuanLi2816/mini-verl/pull/30) was squash-merged as `9914c6d358fd6b0acc5a945c0eab67aebb1b2b51`; release-metadata PR [#31](https://github.com/DaoyuanLi2816/mini-verl/pull/31) was squash-merged as `838ca0b5ab82be88440f2437fac2c2046bde672b`; annotated tag `v0.4.0` resolves to the latter exact commit |
| version transition | `v0.4.0` is immutable and public; this state-sync change identifies subsequent development as `0.5.0.dev0` |
| release execution | tag run [`30777530767`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30777530767) passed metadata, full tests, one-time build, OIDC publication with attestations, public verification, exact clean install and GitHub Release creation |
| published wheel | [`miniverl-0.4.0-py3-none-any.whl`](https://pypi.org/project/miniverl/0.4.0/), SHA-256 `8204c2f015e017ff15aadb465d0afa689949979f4aeba3b9b3abcc3f01c2511a` |
| published sdist | [`miniverl-0.4.0.tar.gz`](https://pypi.org/project/miniverl/0.4.0/), SHA-256 `b6f87ff3a2a97f926301683b983920082719c893bb22a31759b17b0309b1e053` |
| independent public verification | GitHub Release downloads reproduce both public hashes; PyPI exposes provenance for both artifacts bound to `release.yml`, tag `v0.4.0` and commit `838ca0b5`; a refreshed clean Windows Python 3.10 install from `https://pypi.org/simple` reported `miniverl 0.4.0`, passed JSON `doctor`, and kept torch absent |
| release state | complete; PyPI and [`miniVERL v0.4.0`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.4.0) expose the same verified wheel and sdist |

## v0.3.0 RecoveryBench release

| item | current state |
| --- | --- |
| audited baseline | fetched all remotes and started `agent/v0.3-recoverybench` from clean public `main` at `6c79c1fcc5a6e55dd9f4af843d446ece1c454431`; no PR was open, main CI/build were green, and development advanced directly from `0.2.7.dev0` to `0.3.0.dev0` without a `0.2.7` release |
| available execution environment | Windows checkout with Python 3.12, Torch 2.13.0+cu130 and an NVIDIA GeForce RTX 4080 (16376 MiB); GitHub and Hugging Face access are available and publication remains restricted to exact validated tags |
| immutable baseline | calculator benchmark SHA-256 is `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`; calculator protocol-teacher HEAD is `23323751318135484c06c043b1f9b9e7016dd89f`; existing tags remain untouched |
| phase boundary | RecoveryBench must be preregistered, fully measured with the sequential engine, frozen, merged, released as `v0.3.0`, publicly verified and state-synced to `0.4.0.dev0` before padded batching begins |
| current highest-risk work | RecoveryBench, release validation, integration and Trusted Publishing verification are complete; the next phase is the separately gated v0.4 consumer batched runtime |
| RecoveryBench environment | `sqlite_recovery` has 12 structurally disjoint versioned templates, deterministic controlled/natural/no-intervention subsets, structured tool-error provenance, executable recovery oracles, and exact recovery metrics; focused environment and backward-read tests pass |
| preregistration | public commit `7087b3a333463b88a62ffed73daee2c85d039145` and revision-1.3 digest `9c4c2ec19a56cebb2b2c1c0f3c7e504a9285467c99ae1590488251fbf2ff3934` bind the final procedure; the later public wall-budget amendment was reverted, its partial replacement stopped, and the already-complete frozen experiment was retained without rerun |
| teacher selection so far | the historical calculator teacher completed 96 eval tasks and failed the gate at 25.0% strict, 10.7% recovery, 95.8% parse validity and 14.4% tool execution; candidate A trained for 64 QLoRA SFT updates and its in-process NF4 eval measured 86.5%, 78.1%, 100% and 87.5%; a separately loaded full-precision reapplication failed at 65.6%, 65.6%, 100% and 62.7% and is retained as a noncanonical diagnostic |
| selected RecoveryBench teacher | candidate A reloaded on its preregistered NF4 base passed the independent 96-task eval-only gate at 90.6% strict success, 81.2% recovery, 100% parse validity and 87.1% tool execution; it is public at `DaoyuanLi/mini-verl-qwen3-1.7b-sqlite-recovery-teacher@eb2747895ec32dab47c5b50c2d8aa9c0d9701e0d`, with adapter weights SHA-256 `5355f7007efb904d1b45a1aeb9b73b479b6f52025ab92502ab7895706155b2ba` |
| portable adapters | a regression reproduced PEFT exporting a machine-local snapshot path as the base identity; export now rewrites the configured model ID and immutable revision before checksumming, and the portable export reload test passes |
| shared frozen data | schema-v3 benchmark execution prepares one frozen-student dataset per seed, reuses it across consumers and budget views, and can reuse the exact primary-view cold-start checkpoint; the two-consumer/two-view regression and all 80 v2/v3 benchmark tests pass |
| frozen-data fail closed | persisted offline KD now compares the loaded cold-start checkpoint content digest with the immutable dataset manifest before copying any data; a regression proves mismatched bundles are rejected, and final preparation uses `recoverybench-v1-final-frozen-s{seed}` rather than the calibration bundle |
| eval calibration and final freeze | all three 8-update calibration arms completed on eval only: SFT 6,675 positions / 50.5560895 s, frozen KD 6,467 / 52.0759896 s, and strict OPD 6,224 / 650.9775521 s; preregistration revision 1.2 freezes 6,224 selected positions and 50 continuation seconds, bound to calibration SHA-256 `af0cb73c60655c37c4bafba6ea7893e4bb7260e82c6b2915bb646b8872cbe35e` |
| frozen final specifications | schema-v3 equal-update, equal-selected-position and equal-wall-time configs cover the locked three seeds and test split; deterministic 10,000-replicate paired analysis and all three data-bound SVG generators are committed before final test |
| preserved v1.2 invalidation | the first partial final run exposed a real schedule defect: `offline-kd-oracle` collected only 8 oracle tasks and replayed them rather than using the common 64-task prefix; the process was stopped before that arm's test, the entire partial run and frozen bundle were moved intact to `artifacts/superseded/recoverybench-v1.2-oracle-schedule-defect/`, and none of its outcomes will enter the replacement analysis |
| preregistration correction | revision 1.3 explicitly freezes `offline-kd-oracle.collection_tasks: 64`, documents the defect and required every final arm/seed to restart from fresh cold starts; no invalid v1.2 outcome enters the publication set |
| final artifact audit | all 36 task artifacts, 4,608 trajectories and 101,787,618 raw bytes passed embedded SHA-256 and byte-count checks; each arm contains 128 final-test tasks, task IDs pair within seed, three cold checkpoints and three frozen datasets are reused exactly, and all 75 fresh-OPD update rollouts satisfy rollout policy version = current parameter version |
| equal-update result | frozen-student KD reached 23.2% strict success and 22.8% recovery after error versus 10.9% and 9.1% for strict fresh OPD; paired fresh-minus-frozen differences are -12.24 points (95% paired bootstrap -15.89 to -8.59, 384 tasks) and -13.79 points (-20.69 to -6.90, 116 paired error cases) |
| cost result | strict fresh OPD averaged 686.80 continuation seconds versus 52.10 for frozen KD; budget-50 queried 49.77% of model-generated positions but averaged 720.76 seconds because teacher backbone forwards were unchanged |
| secondary budgets | all equal-selected-position arms crossed the 6,224 target after eight steps with recorded overshoot; the nominal 50-second artifact is explicitly a cycle-capped wall diagnostic because SFT and frozen KD completed eight cycles while fresh OPD crossed the target in one indivisible step |
| immutable RecoveryBench results | equal-updates `6ce2e6837e12b99ebc4fad6d27ce3e69c92e295ff3b9b60e0f68c2d308022384`; equal-selected-positions `fe4c9afc799724dfe7a32e631676a1e5177c44559a7374d2ea31da135354f137`; wall diagnostic `425b0fa568f37b09e61af731d3da5009bd3833bddde6efaf2c66e9dba8355cbe`; task JSONL `aff96bffc6da27240a852410ac041bd4d95badf34cad030e6f437be1491a55ad`; paired analysis `8a6891f74aed80f07ec00d5ea1909895c579346e1abbb1d5d95a354bb46c6b81` |
| technical publication | the generated Markdown analysis, three SVGs and deterministic six-page PDF report are data-bound to the frozen JSON; native/README-width SVG inspection and every-page PDF inspection found no clipping or overlap |
| integration source | RecoveryBench PR [#27](https://github.com/DaoyuanLi2816/mini-verl/pull/27) was squash-merged as `bee82d3a6e2c8a5c3f4c62c0d3827a07483a1977`; release-metadata PR [#28](https://github.com/DaoyuanLi2816/mini-verl/pull/28) was squash-merged as `624c6a352db53e0bd2038c4aeb0669a16a402239`; annotated tag `v0.3.0` resolves to the latter exact commit |
| version transition | `v0.3.0` is immutable and public; this state-sync change identifies subsequent development as `0.4.0.dev0` |
| release execution | tag run [`30772772078`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30772772078) passed metadata, full tests, one-time build, OIDC publication with attestations, public verification, exact clean install and GitHub Release creation |
| published wheel | [`miniverl-0.3.0-py3-none-any.whl`](https://pypi.org/project/miniverl/0.3.0/), SHA-256 `e42404ced88b75ba4ff31541cb3df697da0eee09a68f603e4eadbf69a11d2032` |
| published sdist | [`miniverl-0.3.0.tar.gz`](https://pypi.org/project/miniverl/0.3.0/), SHA-256 `006ce418243286dd27e0731090bf9fa1711a1abc962ecfcc4d39807257539cb2` |
| independent public verification | GitHub Release downloads reproduce both public hashes; a no-cache Windows Python 3.10 install from `https://pypi.org/simple` reported `miniverl 0.3.0`, passed JSON `doctor`, and kept torch, Transformers, PEFT and bitsandbytes absent |
| release state | complete; PyPI and [`miniVERL v0.3.0`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.3.0) expose the same verified wheel and sdist |

## v0.2.6 concurrency, lifecycle and privacy correctness release

| item | current state |
| --- | --- |
| audited baseline | fetched all remotes and started `v0.2.6-final-concurrency` from clean public `main` at `ba0cbc33d731936dcb0e44fc42e29cae2f1c803d` (`0.2.6.dev0`) |
| regression-first evidence | isolated pre-fix runs reproduced a real `close()` teardown race, six checkpoint-load overlap failures, six evaluation-mode restoration failures, and 22 semantic-key/URL/path privacy failures; deterministic concurrency tests use `threading.Event`, never timing sleeps |
| ownership and mode | `close()` owns the same non-blocking operation guard as train/evaluate/checkpoint work before changing state or releasing resources; public checkpoint load checks READY before and after ownership and delegates to an owner-only implementation; evaluation restores the exact prior model mode in `finally` |
| privacy | semantic components cover snake/kebab/camel/Pascal keys with explicit benign metadata exceptions; supported URL schemes retain public structure while removing userinfo; arbitrary embedded absolute Windows/UNC/POSIX paths are reduced to portable basenames without rewriting URLs, relative paths or mathematical slash expressions |
| focused adversarial pass | lifecycle, checkpoint-resume, standalone-evaluation, benchmark privacy and all shareable report-format tests pass **209 tests**, including generated semantic keys, credentialed URLs and POSIX paths; an added false-positive probe found and fixed mid-field matching in `loss/token = 0.5` |
| static and documentation gates | `git diff --check`, Ruff check/format, mypy over 77 source files, actionlint 1.7.12 with verified upstream checksum, generated `PYPI.md`, Markdown links, generated schema and generated benchmark-SVG byte comparison all pass |
| complete local suite | after refreshing editable distribution metadata from `0.2.6.dev0` to source `0.2.6`, the full non-GPU/non-network gate passed **1348 tests**, 6 deselected, at **87.46%** branch coverage; the available RTX 4080 gate passed **5** and the network gate passed **3** |
| compatibility evidence | torch-free core passed **1058 tests** on each of Python 3.10, 3.11, 3.12 and 3.13; the minimum bundle (Torch 2.3.1, Transformers 4.51.3, PEFT 0.12.0, Accelerate 0.33.0, NumPy 1.24.4, bitsandbytes 0.43.3) and latest bundle (Torch 2.13.0, Transformers 5.14.1, PEFT 0.20.0, Accelerate 1.14.0, NumPy 2.5.1, bitsandbytes 0.50.0) each passed 2 no-network training smokes and the **133-test** HF/config suite |
| platform concurrency | the Windows complete suite includes the deterministic operation matrix; a fresh WSL2/Linux Python 3.12 environment at implementation commit `7410674802fa178fdde1ed5064c17608cd3fa806` passed **76** lifecycle, standalone-evaluation and resume/concurrency tests; PR and post-merge Ubuntu CI both passed |
| packaging and installs | one `0.2.6` wheel and one sdist pass Twine; the extracted sdist passes Ruff, format, mypy and **1348** CPU tests, rebuilds a wheel with the same **78-file** runtime inventory, and clean Python 3.10 core plus Python 3.12 `[train]` installs pass version/doctor and demo/inspect/report/standalone-eval smokes |
| immutable evidence | the frozen calculator JSON remains byte-identical at SHA-256 `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`; all existing tags and `DaoyuanLi/mini-verl-qwen3-1.7b-protocol-teacher@23323751318135484c06c043b1f9b9e7016dd89f` remain unchanged |
| integration source | PR [#25](https://github.com/DaoyuanLi2816/mini-verl/pull/25) was squash-merged as `59fe738709526a13f354a744ab763f13530de4d1`; annotated tag `v0.2.6` resolves to that exact commit |
| version transition | `v0.2.6` is immutable and public; this state-sync change identifies subsequent development as `0.2.7.dev0` |
| release execution | tag run [`30722451004`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30722451004) passed metadata, full tests, one-time build, OIDC publication, public hashes and attestations, clean public install, and GitHub Release creation |
| published wheel | [`miniverl-0.2.6-py3-none-any.whl`](https://pypi.org/project/miniverl/0.2.6/), SHA-256 `11d6b001752c41a0100f12c29b125a9dc082703dbeadc6b0317a88ac818d8695` |
| published sdist | [`miniverl-0.2.6.tar.gz`](https://pypi.org/project/miniverl/0.2.6/), SHA-256 `91e7b2918286c342cacaf2582dbed57c1e7a1bf4e1064d327e349b1d77c28886` |
| independent public verification | GitHub Release downloads reproduce both public hashes; a no-cache Windows Python 3.10 install from `https://pypi.org/simple` reported `miniverl 0.2.6`, passed `doctor`, and kept torch, Transformers, PEFT and bitsandbytes absent |
| release state | complete; PyPI and [`miniVERL v0.2.6`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.6) expose the same verified wheel and sdist |

## v0.2.5 final correctness release status

| item | current state |
| --- | --- |
| audited baseline | fetched all remotes and started `v0.2.5-final-correctness` from clean public `main` at `3d3d59993a99bb1a7c1225f970a3e052a43e4dc1` (`0.2.5.dev0`) |
| integration source | PR [#22](https://github.com/DaoyuanLi2816/mini-verl/pull/22) was squash-merged as `a9a84510741b4ade8a405c100affdf1caed55ae6`; annotated tag `v0.2.5` resolves to that exact commit |
| version transition | `v0.2.5` is immutable and public; this state-sync change identifies subsequent development as `0.2.6.dev0` |
| regression-first evidence | the focused pre-fix run reproduced 23 failures and 472 passes: SQLite raised `ValueError`/`OverflowError` for `nan`, infinities, overflow notation and a huge integer through both direct verification and the real rollout path; every built-in protocol-v2 final example remained the placeholder `answer` |
| focused correctness pass | `python -m pytest` over the changed verifier, protocol, privacy, lifecycle, locking, packaging, CLI and standalone-eval surfaces passed **798 tests**; the old implementation had separately reproduced 23 verifier/protocol failures, 16 privacy failures, 5 lifecycle/concurrency failures, the report-lock race, two eval-lock races and the nested-link defect |
| static gates | `git diff --check`, `ruff check .`, `ruff format --check .`, `mypy src/miniverl` and cached `actionlint` all pass |
| complete local suite | after refreshing stale editable metadata from `0.2.4` to source `0.2.5.dev0`, the full non-GPU/non-network gate passed **1278** tests with **87.37%** branch coverage; the available RTX 4080 gate passed **5**, and the network gate passed **3** |
| compatibility evidence | torch-free core passed **1021** tests on each of Python 3.10, 3.11, 3.12 and 3.13; the minimum bundle (torch 2.3.1, Transformers 4.51.3, PEFT 0.12.0, Accelerate 0.33.0, NumPy 1.24.4, bitsandbytes 0.43.3) and latest bundle (torch 2.13.0, Transformers 5.14.1, PEFT 0.20.0, Accelerate 1.14.0, NumPy 2.5.1, bitsandbytes 0.50.0) each passed both no-network training smokes; Transformers 4.51.3 and 5.14.1 each passed the **133-test** HF/config compatibility suite |
| multiprocessing platforms | the complete Windows suite includes the new spawn-based report/eval/run-lock races and passed; a local WSL launch was unavailable because its VHDX returned `ERROR_SHARING_VIOLATION`, while the PR and post-merge Ubuntu CI gates passed |
| lifecycle and ownership | new writable manifests start `ready`, transition atomically to `running`, and close-before-train as `closed_before_training`; training-private eval/checkpoint implementations are separated from public calls, automatic reports remain under the writer lock, and standalone eval transfers one pre-acquired lock before reading mutable state |
| privacy and packaging | semantic secret suffixes, authorization/cookie/session fields, URL userinfo, paths with spaces, UNC and HTML-escaped variants are redacted; the PyPI generator rewrites nested linked images and both generated `PYPI.md` and built wheel `METADATA` reject all remaining relative project targets |
| final-version packaging gate | an isolated `0.2.5` build produced one wheel and one sdist and passed Twine; the extracted sdist passed Ruff, format, mypy and **1278** non-GPU/non-network tests, rebuilt a wheel with the same **78-file** runtime inventory, and exposed zero relative metadata targets |
| clean-install gate | a clean Python 3.10 core wheel install reported `0.2.5`, passed `doctor`, and kept torch/Transformers/PEFT/bitsandbytes absent; a clean Python 3.12 wheel + `[train]` install ran `demo`, `inspect`, `report`, and weights-only standalone `eval` |
| immutable evidence | no frozen tag, adapter revision, or benchmark result was changed; the required benchmark SHA-256 remains `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc` |
| release execution | tag run [`30611603505`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30611603505) passed metadata, full tests, one-time build, OIDC publication, public hashes and attestations, clean public install, and GitHub Release creation |
| published wheel | [`miniverl-0.2.5-py3-none-any.whl`](https://pypi.org/project/miniverl/0.2.5/), SHA-256 `70c98284bce151fc74b508047b354929846efb71c3fe8f451c0d0ba1bec48e9d` |
| published sdist | [`miniverl-0.2.5.tar.gz`](https://pypi.org/project/miniverl/0.2.5/), SHA-256 `d30bb07ebca676a3960d4b5c46075a8a2e13e58629b96984e30f8f7bab67dce0` |
| independent public verification | a no-cache Windows Python 3.10 install from `https://pypi.org/simple` reported `miniverl 0.2.5`, passed `doctor`, and kept torch, Transformers, PEFT and bitsandbytes absent; the public simple index exposes provenance links for both distributions |
| release state | complete; PyPI and [`miniVERL v0.2.5`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.5) expose the same verified wheel and sdist |

## v0.2.4 framework-hardening release status

| item | current state |
| --- | --- |
| integration source | PR [#18](https://github.com/DaoyuanLi2816/mini-verl/pull/18) was squash-merged as `57dec193af88b462dcc41d82fc6fecb813e161fd`; annotated tag `v0.2.4` resolves to that exact commit |
| version transition | `v0.2.4` is immutable and public; post-release development identifies as `0.2.5.dev0` |
| immutable evidence | the v0.2 calculator benchmark remains byte-identical at SHA-256 `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`; earlier tags and the pinned protocol-teacher adapter are unchanged |
| strict protocol and numerics | a shared bounded strict decoder rejects non-finite values, duplicate keys, 128+ digit integers, depth over 32, more than 256 members and invalid surrogates; protocol-v2 requires complete numeric/unit verification while historical protocol/verifier-v1 remains explicit |
| lifecycle and locking | `TrainerState` has atomic one-shot entry and stable terminal behavior; `filelock` protects run construction, resume, overwrite, evaluation, report and export across processes before model loading |
| packaging and provenance | tag-pinned PyPI documentation, self-testing sdist, lean wheel, exact submitted/validated/resolved configuration layers and canonical portable public artifacts passed the release gates; exact CPU/GPU/network evidence remains in [`docs/generated/quality.json`](docs/generated/quality.json), bound to implementation commit `6911acf11978cfc5f6a99375cbe7a1666fcf7a85` |
| release execution | tag run [`30522484949`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30522484949) passed metadata, tests, one-time build, OIDC publication and attestation generation; it ended red only because urllib received PyPI's browser challenge before public install and GitHub Release creation |
| verified recovery | PRs [#19](https://github.com/DaoyuanLi2816/mini-verl/pull/19) and [#20](https://github.com/DaoyuanLi2816/mini-verl/pull/20) hardened recovery and HTML-banner verification. Run [`30524088015`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30524088015) bound the original artifacts to the immutable tag, verified hashes, pinned links and Sigstore attestations, installed the exact public version, and created the GitHub Release without rebuilding or re-uploading |
| published wheel | [`miniverl-0.2.4-py3-none-any.whl`](https://pypi.org/project/miniverl/0.2.4/), SHA-256 `3f5a239bbbd2f85217cf11f691fbb63f647092f67b82da4de38bd6907c5ab0f1` |
| published sdist | [`miniverl-0.2.4.tar.gz`](https://pypi.org/project/miniverl/0.2.4/), SHA-256 `03f0e844df2c91deed5c211cdd2dd598d22f03d59d99cd8e792a58211c0b2296` |
| independent verification | GitHub Release downloads reproduce both public hashes; a no-cache Python 3.10 install from `https://pypi.org/simple` reports `miniverl 0.2.4`, passes `doctor`, and keeps torch, Transformers, PEFT and bitsandbytes absent |
| publication authorization | granted by the maintainer and completed through Trusted Publishing without a long-lived PyPI token |
| exact blocker | none |

## v0.2.3 clarity and defensive-hardening release status

| item | current state |
| --- | --- |
| integration source | PR [#15](https://github.com/DaoyuanLi2816/mini-verl/pull/15), squash-merged as `3e54ebf9a8501f58cb6a0901d827221aba8792e0`; release metadata PR [#16](https://github.com/DaoyuanLi2816/mini-verl/pull/16) merged as `38924da743180e6767f1e3b252feafdccd70759b` after all ten checks passed |
| version transition | `v0.2.3` was released from exact commit `38924da743180e6767f1e3b252feafdccd70759b`; post-release development now identifies as `0.2.4.dev0` |
| cache safety | shard size survives reopen; allocation advances past the highest indexed or on-disk numeric suffix; pruning publishes a copied index before best-effort orphan cleanup, with fault-injection coverage on both failure boundaries |
| protocol examples | schema-v2 examples come from each active environment `ToolSpec`; calculator, JSON navigation, SQLite and custom-environment round trips pass while protocol v1 remains byte-frozen |
| presentation | the generated figure scopes claims to the saturated v0.2 calculator task, reports completed controls at 0%, omits the unsourced preparation duration and remains legible at native and README widths |
| frozen scientific artifact | `benchmarks/results/gpu-calc-hard-equal-update-v2.json` remains byte-identical at SHA-256 `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc` |
| release gates | release run [`30513947051`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30513947051) passed metadata, 1085 tests at 87.30% branch coverage, one-time build, OIDC publication, public hashes/attestations, clean public install and GitHub Release creation |
| published wheel | [`miniverl-0.2.3-py3-none-any.whl`](https://pypi.org/project/miniverl/0.2.3/), SHA-256 `033e51bfbdae20a91d942ef7a5c22ef6c8a00317cc9b775b102d303f2e1a6619` |
| published sdist | [`miniverl-0.2.3.tar.gz`](https://pypi.org/project/miniverl/0.2.3/), SHA-256 `6f7d20fd4b4a90e6a3fe1e97c9ced26268e013bb87462ba75a7d09510bd2f011` |
| publication authorization | granted on 2026-07-29 and completed through the tag-only Trusted Publishing workflow without a long-lived PyPI token |
| exact blocker | none; PyPI and GitHub expose byte-identical artifacts, attestations passed, independent Python 3.10 core installation passed, and development has advanced past the immutable release |

## v0.2.2 single-GPU portability release status

| item | current state |
| --- | --- |
| integration source | PR [#13](https://github.com/DaoyuanLi2816/mini-verl/pull/13), merged as `518590cb43ff788fa65f73ee9cf3a7afb6dfba5a` after all pull-request and synchronized-main checks passed |
| version transition | `v0.2.2` was released from exact commit `518590cb43ff788fa65f73ee9cf3a7afb6dfba5a`; post-release development now identifies as `0.2.3.dev0` |
| hardware scope | one NVIDIA CUDA GPU with no device-name allowlist; automatic dtype selects bf16 when supported and fp16 otherwise; exact fit remains model/budget dependent |
| measured evidence | the only published real-model GPU measurements remain the RTX 4080 runs; RTX 3070, Titan V, RTX 5090 and other cards are portable code paths, not fabricated benchmark claims |
| presentation | banner and generated benchmark SVG use a dark single-GPU visual system; grid lines begin below tick labels and diagnostic 0% controls are status pills |
| package discovery | both READMEs, project metadata and the GitHub About homepage expose `https://pypi.org/project/miniverl/` |
| frozen scientific artifact | `benchmarks/results/gpu-calc-hard-equal-update-v2.json` remains byte-identical at SHA-256 `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc` |
| release state | release run [`30494182647`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30494182647) passed on attempt 3 after PyPI simple-index propagation; OIDC publication, public hashes, attestations, clean install and GitHub Release all passed |
| published wheel | [`miniverl-0.2.2-py3-none-any.whl`](https://pypi.org/project/miniverl/0.2.2/), SHA-256 `1ead97173bb11ce3da963b94f628df825a5b14648fed488cf4d88c47cba9dd59` |
| published sdist | [`miniverl-0.2.2.tar.gz`](https://pypi.org/project/miniverl/0.2.2/), SHA-256 `3951dd4addc5d85b3e58ce72ecffac65c38bf2eab951d2c08cce8f20c886185c` |
| exact blocker | none; public PyPI and GitHub Release artifacts are byte-identical, and clean Python 3.10 and workflow Python 3.12 core installations passed |

## Historical v0.2.1 correctness release status

| item | current state |
| --- | --- |
| audited starting commit | public and local `main` both resolved to `5b1c043b188b30b1261e118293f6fe124e2b7acb` after `git fetch --all --prune` on 2026-07-29 |
| integration source | branch `v0.2.1-correctness` and correctness PR [#11](https://github.com/DaoyuanLi2816/mini-verl/pull/11), created without rewriting `main` or the immutable `v0.2.0` tag; the PR remains the authoritative integration record after its source branch is deleted |
| version transition | `v0.2.1` was released from exact commit `591881b0d094f5c53ff47a9419e679b762fb44b0`; post-release development now identifies honestly as `0.2.2.dev0`, while changelog and citation metadata retain the published `0.2.1` record |
| release state | annotated tag [`v0.2.1`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.1) resolves to `591881b0d094f5c53ff47a9419e679b762fb44b0`; release run [`30474597179`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30474597179) passed all five jobs, including OIDC publication, public verification and GitHub Release creation |
| publication authorization | granted on 2026-07-29; the release completed through the tag-only Trusted Publishing workflow without a long-lived PyPI token |
| frozen scientific artifact | `benchmarks/results/gpu-calc-hard-equal-update-v2.json` remains required at SHA-256 `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc` |
| immutable protocol adapter | `DaoyuanLi/mini-verl-qwen3-1.7b-protocol-teacher@23323751318135484c06c043b1f9b9e7016dd89f`; its original v1 competence record is accepted for v1 without fabricating v0.2.1 metrics, and it was neither overwritten nor retrained |
| published wheel | [`miniverl-0.2.1-py3-none-any.whl`](https://pypi.org/project/miniverl/0.2.1/), SHA-256 `0177d50026da86047c2a03f90e7786c794b26c5b0d6fef193c58ed35c08d8cda` |
| published sdist | [`miniverl-0.2.1.tar.gz`](https://pypi.org/project/miniverl/0.2.1/), SHA-256 `80f890c1ab8be0ccdf6c5ce293a5c4d7bb6a6f7ab7a57db34090384fcaa7e16c` |
| exact blocker | none; PyPI and the GitHub Release expose byte-identical distributions, PyPI reports Trusted Publishing attestations, and a clean public core install passed |

### Correctness changes and evidence

| area | defect and implemented invariant | regression evidence |
| --- | --- | --- |
| run lifecycle | a new run could silently attach to a non-empty directory; creation is now exclusive, generated IDs include microseconds plus randomness, resume/overwrite are explicit and mutually exclusive, and overwrite uses rollback-safe whole-directory replacement | run creation, collision, concurrency, stale append, demo and benchmark resume tests; final-wheel collision exited 1 with every file hash unchanged, then `--overwrite` produced `status=completed`, eight fresh metric records and no stale standalone-eval file |
| checkpoints/eval | checkpoint names could be selected lexicographically, writes were not atomic/checksummed, and eval could load fresh or partial weights; complete sibling-temp checkpoints now have a manifest, SHA-256/size/content digest and identity, and selection uses validated `state.global_step` | corruption, missing weights, duplicate-step ambiguity, temp-ignore, stale-final, legacy-v0.2 and fail-before-mutation tests; clean-wheel standalone eval loaded only weights and reported checkpoint/global step 4, parameter version 4 and `checksummed_v1` |
| offline KD | resume could regenerate the nominally fixed dataset; trajectories, task order, exact spans/tails and provenance are now a first-class checksummed artifact | uninterrupted/resumed parameter, optimizer, task order, cache and dataset digest equality tests |
| manifests | startup and terminal state could be conflated; immutable `manifest.start.json` and atomic completed/failed/interrupted `manifest.json` records preserve original failures and actual counters/digests | lifecycle, exception-preservation and fault-injection tests |
| OOM transaction | retry boundaries could repeat stochastic work or an optimizer commit; only gradient computation is retryable with RNG restoration and gradient clearing, while optimizer commit is single-shot | RNG/dropout equivalence, one-commit, optimizer-OOM and non-OOM tests |
| protocol/adapter | the historical v1 prompt has an ambiguous example, but changing it would invalidate the published adapter; v1 is frozen, v2 examples parse, and competence is protocol-version aware | v1 byte fixture, v2 parse/round-trip and v1/v2 adapter-gate tests; the immutable Hub revision passed the live network test |
| losses/metrics | SFT span metrics could report zero divergence instead of CE, and tool events were double counted; objective/divergence/CE are distinct and emitted/parsed/executed/error/final events have precise denominators | per-token aggregate equivalence and adversarial agent-event tests |
| cache/identity | empty tails, ordered span types, checksum flags, adapter identity and tokenizer/model revisions could be ambiguous or lossy; schema v2 fixes each while retaining v1 reads | exact-zero, ordering, checksum, adapter, exact-full-vocab, structural tokenizer, revision and LM-head tests |
| parameter versions | cycle count was used as a proxy for parameter changes; `parameter_version` advances only after a successful commit and rollout/optimizer counters are separate | no-op, failed update, replay and exact-resume tests |
| reset/lifecycle API | the runner could ignore `reset()`'s observation and examples could leak trainer resources; reset is authoritative and public examples use context managers | dynamic reset/state-ID/exactly-once and destructive close/context tests |

### Final local gates

- `git diff --check`, `ruff check .`, `ruff format --check .`,
  `mypy src/miniverl` and actionlint 1.7.12 all pass.
- `pytest -q -m "not gpu and not network" --cov=miniverl
  --cov-report=term-missing --cov-fail-under=80`:
  **1066 passed, 6 deselected, 87.11% branch coverage**.
- `pytest -q -m gpu` on Torch 2.13.0+cu130 and an NVIDIA GeForce RTX 4080:
  **5 passed, 1067 deselected**.
- `pytest -q -m network`: **3 passed, 1069 deselected**, including the pinned
  public protocol-teacher revision.
- Minimum boundary, Python 3.10.11 / Torch 2.3.1+cpu / Transformers 4.51.3 /
  PEFT 0.12.0 / Accelerate 0.33.0 / NumPy 1.24.4 / bitsandbytes 0.43.3:
  **132 passed** under the offline guard.
- Current boundary, Python 3.13.13 / Torch 2.13.0+cpu / Transformers 5.14.1 /
  PEFT 0.20.0 / Accelerate 1.14.0 / NumPy 2.5.1 / bitsandbytes 0.50.0:
  **132 passed** under the offline guard.
- Clean build produced exactly one wheel and one sdist; both passed
  `twine check`. Wheel/sdist content assertions passed.
- A clean core-wheel environment ran help/version/doctor without installing or
  importing torch, Transformers, PEFT or bitsandbytes. A separate clean
  `[train]` environment ran demo, inspect, report, collision, overwrite and
  weights-only standalone evaluation.
- Every run recipe validated, every benchmark config resolved, the generated
  JSON Schema byte-matched the committed file, every benchmark result
  validated, all tracked JSON/JSONL parsed, 26 Markdown files passed link
  checking, and no model weights/caches/databases are tracked.
- The banner and benchmark SVG were rendered at native size and inspected.
  `docs/gpu-calc-hard-equal-update-v2.svg` remains generated from the frozen
  JSON and labels the cold start `NO TRAINING` and the diagnostic controls
  `PROTOCOL MISMATCH` instead of presenting inapplicable zero bars as outcomes.
- GitHub environment `pypi` exists with a branch policy; `release.yml` used
  OIDC `id-token: write` to publish and independently verify
  [`miniverl 0.2.1`](https://pypi.org/project/miniverl/0.2.1/).

Currently failing commands: **none**.

Publication record: PR #11, annotated tag `v0.2.1`, release run `30474597179`,
the public PyPI files and the GitHub Release are authoritative. The published
tag and distributions are immutable; new work proceeds from `0.2.2.dev0`.

## v0.2 release-hardening status

| item | current state |
| --- | --- |
| public repository | `https://github.com/DaoyuanLi2816/mini-verl` |
| release-code merge | PR [#6](https://github.com/DaoyuanLi2816/mini-verl/pull/6) and the final state sync in PR [#7](https://github.com/DaoyuanLi2816/mini-verl/pull/7) are merged; release commit `6092706b4a4e750c4571d7d6a7decbc26af851b2` is recorded as annotated tag `v0.2.0` |
| previous integration | PR [#4](https://github.com/DaoyuanLi2816/mini-verl/pull/4) and PR [#5](https://github.com/DaoyuanLi2816/mini-verl/pull/5) are merged |
| release branch | `v0.2-final-release` was merged and deleted |
| final integration | PR #6 and PR #7 merged on 2026-07-28; all eight checks on each PR and both post-merge `main` workflows passed |
| lifecycle fix | merged on `main`: destructive, idempotent cleanup and explicit cold-start/arm context boundaries are implemented |
| public default-branch health | final pre-tag `ci` run [`30417912194`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30417912194) and `build` run [`30417912198`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30417912198) completed successfully |
| final-pass status | complete locally: 1004 CPU tests, 5 GPU tests, 129 focused offline tests, dual Transformers compatibility, build, clean-wheel, README-command, link, schema, JSON/JSONL, privacy and release-workflow gates pass |
| version | `0.2.0` published on 2026-07-28 (2026-07-29 UTC); tag `v0.2.0` resolves to `6092706b4a4e750c4571d7d6a7decbc26af851b2` |
| release workflow | tag run [`30421231859`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30421231859) passed metadata, full tests, single-build, OIDC publish, public verification and GitHub Release jobs |
| PyPI | [`miniverl 0.2.0`](https://pypi.org/project/miniverl/0.2.0/) was created by Trusted Publishing; wheel SHA-256 `cf850a6333483a3ee22c0c0e98df1e1b2e6faa184480573e0666658b53a29262`, sdist SHA-256 `3d5107b4f6351204335f800ce924208843f08f54441378bd9f25c3c6fa17456b` |
| GitHub Release | [`miniVERL v0.2.0`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.0) is public and contains the same wheel, sdist and `SHA256SUMS` |
| GitHub environment | `pypi` created through the repository API on 2026-07-28; custom deployment policy permits only tags matching `v*` |
| external publication blocker | none; the exact pending publisher was registered, the first OIDC upload succeeded, and PyPI records `release.yml` on `DaoyuanLi2816/mini-verl` as the publisher |
| Hugging Face adapter | public at `https://huggingface.co/DaoyuanLi/mini-verl-qwen3-1.7b-protocol-teacher`, immutable head `23323751318135484c06c043b1f9b9e7016dd89f` |
| scientific artifact | schema-v2 JSON remains byte-identical at SHA-256 `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc` |

Focused final-pass evidence:

- `pytest -q tests/unit/test_offline_contract.py
  tests/integration/test_hf_backend_offline.py tests/unit/test_config.py` --
  **129 passed**. Actual `train`, `benchmark`, `eval` and `export-adapter`
  commands also completed with `--offline`; the export loaded the cached pinned
  Qwen3 revision and wrote a standard PEFT adapter.
- Socket denial exposed a Transformers 5.x PEFT auto-detection request even
  with the top-level local-only flag. Version-compatible PEFT probe kwargs now
  carry `local_files_only` explicitly on 5.x without duplicating the parameter
  on 4.x; missing model, tokenizer, full adapter snapshot and partial adapter
  snapshot tests all record zero socket attempts.
- Hub adapter validation returns public provenance plus one exact local
  snapshot and its config, weights and manifest paths. PEFT receives only that
  local snapshot; public manifests retain no cache path.
- `release.yml` passed actionlint 1.7.12. Static tests prove manual dispatch and
  branch pushes cannot publish, a `v*` tag push can, distributions build once,
  and PyPI verification precedes GitHub Release creation.
- `ruff check .`, `ruff format --check .` and `mypy src` pass. The complete
  CPU suite reports **1004 passed, 5 deselected**; the CUDA suite reports
  **5 passed, 1004 deselected** on the RTX 4080.
- Independent Python 3.12 environments with Transformers **4.51.3** and
  **5.14.1** each report **124 passed** for the offline Qwen3/PEFT/config
  compatibility bundle.
- A fresh isolated build produced the 0.2.0 wheel and sdist and both passed
  `twine check`. A wheel-only environment confirmed torch absent and core
  commands ready; a second `[train]` environment completed `demo --fast`,
  `inspect` and `report`.
- All 117 local Markdown targets, 47 Markdown anchors and 14 external Markdown
  URLs passed. Ten representative runtime JSONL files (115 records) parsed
  strictly; package/schema/SVG/privacy/release tests report **88 passed**.
- The frozen benchmark remains byte-identical at SHA-256
  `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`.

Currently failing commands:

- None. The tag release, public artifact verification and clean public install
  all pass.

Post-release follow-up:

1. Upload `docs/banner.svg`, rendered at 1280×640 PNG, as the repository social
   preview through the authenticated repository settings page.
2. Keep `v0.2.0` immutable. Future package changes require a new version and tag;
   never move or replace the published tag.

## Historical v0.2 pre-merge evidence (PR #4)

This section records the final local evidence snapshot prepared for integration
through PR #4. The PR link is authoritative for its later merge state.

| item | evidence snapshot |
| --- | --- |
| public repository | `https://github.com/DaoyuanLi2816/mini-verl`; fetched `origin/main` is `3383f2b9a3c595e0fa143fecdc27522ab368b27f` |
| working branch | `v0.2-protocol-aligned-opd`; foundational commits are pushed and the measured-result integration passes the final local gates |
| version | `0.2.0` release candidate; no tag or publication has been created |
| current PR | draft [#4](https://github.com/DaoyuanLi2816/mini-verl/pull/4) |
| GPU runs | protocol teacher, adapter export/gate, 4 GPU tests and all 10 benchmark arms completed locally on the RTX 4080 |
| PyPI/release | not published, no tag, no upload authorized |

Completed v0.2 work in the current worktree:

- Benchmark schema/config v2 separates common and cold-start overrides, performs
  a declared structured diff before model loading, records resolved hashes and
  cumulative accounting, and retains v1 read/JSON-schema compatibility.
- Strict OPD freshness is the default and rejects multiple optimizer updates
  from one rollout batch; explicit replay is labeled
  `online_distillation_with_replay`.
- The ambiguous public `loss.ce_weight` name is replaced by
  `sampled_token_nll_weight` for distillation; a nonzero legacy name is rejected.
- Frozen standard PEFT teacher adapters can be validated, loaded and exported
  with checksums and provenance. The no-network local Qwen-compatible round trip,
  freeze checks, wrong-base/tokenizer checks and missing-file checks pass.
- `recipes/qwen3_1.7b_protocol_teacher_sft.yaml` validates and its exact
  `miniverl train ... --dry-run --json` command succeeds.
- Policy evaluation now records strict success as primary plus lenient
  diagnostic success, valid tool-call rate/count, final-format validity and
  average turns. Protocol-token accuracy is explicitly `null/not_applicable`
  for unaligned free-running trajectories.
- `k == V` bypasses tail smoothing; temperature-squared claims are narrowed and
  a deterministic 48-cell gradient sweep covers three divergences, four
  temperatures and two logit regimes.
- README source-install and CUDA installation order are corrected in English
  and Chinese.
- Candidate A reached 100% strict held-out tool-policy success, passed its
  prespecified 50% gate and was exported with complete checksums; candidates B
  and C were not run by design.
- The five-arm equal-update GPU benchmark completed at both prespecified seeds.
  Cold start was 75% twice; SFT and protocol-teacher OPD were 100% twice;
  raw-teacher and privileged-context OPD were 0% twice.
- Published schema-v2 provenance is made portable before hashing. The result
  JSON, Markdown and data-bound SVG pass schema, privacy and consistency tests.

Evidence executed on this branch:

- `ruff check .` and `ruff format --check .` — clean, 148 files formatted.
- `mypy src/miniverl` — clean, 73 source files.
- complete offline/CPU suite — 966 passed, 4 GPU tests deselected, 86.55%
  branch coverage over 6405 statements.
- Transformers 4.51.3 boundary environment — offline Qwen3/PEFT/config bundle,
  117 passed.
- `python -m build` and `python -m twine check dist/*` — 0.2.0 sdist and wheel
  built, both metadata checks passed.
- committed benchmark schema exactly matches `miniverl schema`; every recipe
  validates and the protocol-teacher dry run reports 24 optimizer steps and 192
  oracle traces.
- focused benchmark/config/cache suite — 129 passed before the adapter slice.
- no-network PEFT adapter suite — 14 passed.
- policy metrics, benchmark v2, packaging, toy pipeline and offline HF bundle —
  119 passed; its only failure was a duplicate test tokenizer ID, corrected and
  rerun as 1 passed.
- loss/property/temperature sweep suite — 45 passed.
- protocol-teacher recipe `validate --json` and `train --dry-run --json` —
  valid, 24 planned optimizer steps and 192 oracle traces.
- `pytest -q -m gpu` — 4 passed on the RTX 4080.
- Candidate A training — 24 optimizer steps in 554.9 s; final strict/lenient
  success, valid-call rate and final-format validity all 100% on 24 tasks.
- Adapter export/validation — checkpoint tree
  `e9c42893b861e371dd48e2c151940a198e22eff2f91649ca6a5303c525c5ee4c`,
  adapter weights
  `8df7e7bc1b8283b910aa13bc4173083ae20c838bcacb366d7dbcabc7b310b994`,
  manifest
  `502bca7489c6fe161ebf198d2a1b4622123d4f958885a7e4714c6a02a2e1ac43`.
- Complete GPU benchmark — 5 arms x 2 seeds, all 10 completed; published JSON
  SHA-256 `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`.
- Fresh 0.2.0 build — sdist and wheel passed `twine check`; a wheel-only venv
  ran `--help`, `--version` and `doctor --json` with torch absent, while a
  second `[train]` venv completed `demo --fast` and inspected its trajectories.

Measured scientific conclusion: protocol competence prevents the observed
raw/privileged-teacher collapse, but protocol-teacher OPD only ties SFT at 100%
on both seeds. There is no measured OPD-over-SFT advantage.

Exact integration steps at this snapshot:

1. Commit and push the portable artifacts and updated interpretation, then make
   draft PR #4 ready once remote checks are green.
2. Merge, verify `main` and its public artifacts, then close superseded
   dependency PRs without creating a tag, release or PyPI upload.
3. Preserve the two-seed result as the v0.2 baseline for a future,
   less-saturated task-family experiment.

## Current phase

At this snapshot, phase 7 is at v0.2 release-candidate integration. The
protocol-teacher experiment, both benchmark seeds and all local gates are
complete; PR #4 contains the integration state.

## Environment (measured, not assumed)

| item | value | how |
| --- | --- | --- |
| OS | Windows 11 Pro 10.0.22631 | `platform.platform()` |
| GPU | NVIDIA GeForce RTX 4080, 16376 MiB, driver 596.49 | `nvidia-smi --query-gpu=name,memory.total,driver_version` |
| dev interpreter | CPython 3.12.13 (`.venv`, uv-managed) | `uv venv --python 3.12` |
| torch | 2.13.0+cu130, `torch.cuda.is_available() == True` | `python -c "import torch; ..."` |
| transformers | 5.14.1 | `importlib.metadata.version` |
| peft | 0.19.1 | idem |
| bitsandbytes | 0.50.0, `bnb.nn.Linear4bit` present | idem |
| PyPI project `miniverl` | PUBLISHED (`0.2.0`, Trusted Publishing, 2026-07-28) | public PyPI JSON, Integrity API and clean install |

## Verified model pair (pinned in the recipes)

Confirmed twice against `huggingface.co/api/models/...` with a negative control
(an all-zero SHA returns 404, so the endpoint really validates revisions):

| role | model | revision | notes |
| --- | --- | --- | --- |
| student | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0, 28 layers, hidden 1024 |
| teacher | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0, 28 layers, hidden 2048 |

`tokenizer.json` is byte-identical across the pair
(sha256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`).
`config.vocab_size` is 151936 while `len(tokenizer)` is 151669 -- the vocabulary
is padded. miniVERL sizes the cache and `top_k` from the model's output
dimension, never from the tokenizer; a GPU test asserts both numbers.

## Historical v0.1 release gates (executed on 2026-07-27)

This table is retained as the v0.1 evidence baseline. Current v0.2 evidence is
recorded in the takeover checkpoint above and will replace this table after the
GPU experiment and clean-install verification finish.

| gate | command | result |
| --- | --- | --- |
| lint | `ruff check .` | **clean** |
| format | `ruff format --check .` | **clean, 138 files** |
| types | `mypy src/miniverl` | **clean, 71 source files** |
| tests | `pytest -q -m "not gpu and not network" --cov=miniverl` | **934 passed, 4 deselected, 85% branch coverage** over 5911 statements |
| GPU tests | `pytest -q -m gpu` | **4 passed** |
| build | `python -m build` | sdist + wheel built |
| metadata | `python -m twine check dist/*` | **PASSED** for both |
| wheel completeness | enumerate `src/miniverl` against the wheel | all 77 entries present |
| clean install (core) | wheel into a fresh venv | `--help`, `--version`, `doctor --json` work; torch absent |
| clean install (train) | wheel + `[train]` into a second fresh venv | `doctor` resolves a CPU-only torch and reports GPU training / QLoRA unavailable with the right remedy; both `examples/` scripts run to completion |
| determinism | two `miniverl demo` runs in the clean venv, artifact-by-artifact | checkpoints and trajectories **bitwise identical**; `eval.json` differs only in seconds, throughput, timestamp and run id |
| artifact portability | every `*.json` / `*.jsonl` parsed with `parse_constant` set to reject | **strict-valid** (this gate found 16 bare `NaN` tokens, now fixed) |
| published results | `jsonschema` validate against `benchmarks/schema/` + personal-information scan | both result files valid and clean |
| docs links | every relative markdown link and `#anchor` resolved | **37 files, 0 broken** |
| CLI/docs parity | every `miniverl <cmd>` in the docs matched against the Typer app | all 11 commands documented, no phantom commands |
| unfinished markers | `rg "TODO\|FIXME\|XXX\|HACK\|NotImplementedError" src tests examples scripts` | none |

## Completed, with evidence

### Numerical core

* `tests/unit/test_losses_exact.py` -- **21 passed**. Brute-force Python
  references for forward KL, reverse KL and beta-JSD; an orientation test that
  catches an accidental KL reversal; the `log 2` bound on symmetric JSD;
  finiteness at logit scale 1e4; finite gradients; fp16/bf16 upcast; the `T^2`
  contract.
* `tests/unit/test_losses_bucketed.py` -- **30 passed**. `log1mexp` against a
  reference; tail-mass validity; `k == V` converging to the exact loss;
  the data-processing-inequality bound for k in {1,2,8,32} across all three
  divergences; monotonicity in k; a bounded reverse-KL tail penalty.
* `tests/unit/test_chunked_equivalence.py` -- **22 passed**. The two-stage
  detach/backward reproduces the unchunked **gradient** for chunk sizes 1, 5 and
  37 across all three divergences; zero-weight positions provably contribute
  nothing; CE mixing is convex; `loss_scale` affects gradients only.
* `tests/property/` -- **14 passed** under Hypothesis, including the
  data-processing inequality evaluated in float64 to `1e-9`.

### Provenance and alignment

* `tests/unit/test_token_provenance.py` -- **30 passed**. A mask that marks a
  `tool_result` token trainable is rejected; spans must tile the sequence;
  position 0 can never be a target; privileged-context alignment computes
  per-segment offsets and rejects a target-token mismatch, a length-changed
  segment, a missing segment, a duplicate key and a tokenizer mismatch.

### Environments

Oracle solves **6/6** train tasks for every (environment, difficulty) pair;
train/eval/test prompt sets are pairwise disjoint. SQLite lockdown, all
**BLOCKED**: `DROP`, `INSERT`, `UPDATE`, `ATTACH`, `PRAGMA`,
`SELECT * FROM sqlite_master` (authorizer), `SELECT randomblob(10)` (function
whitelist), two statements in one call.

### End-to-end

* `tests/integration/test_toy_pipeline.py` -- SFT, offline KD and OPD each run
  to completion and write every documented artifact; OPD increments its policy
  version once per cycle; offline KD announces that it is reusing fixed targets;
  no stored trajectory ever marks tool output trainable.
* `tests/integration/test_resume_and_swap.py` -- an interrupted-and-resumed run
  matches an uninterrupted one to `1e-6` parameter-for-parameter; `swap` matches
  `resident` to the same tolerance; checkpoints contain no pickle; the OOM retry
  only halves the chunk and gives up with actionable advice.
* `tests/integration/test_hf_backend_offline.py` -- a real `Qwen3ForCausalLM`
  built from a config (no download): adapter resolution through PEFT,
  selected-position projection matching a full forward to `1e-4`, KV-cache
  generation matching a cacheless reference token for token, and gradients
  reaching **only** the LoRA tensors.

### Real hardware

* **Historical raw-teacher recipe**,
  `recipes/qwen_consumer_gpu_calc_raw_teacher.yaml`, run id
  `rtx4080-calc-opd`: 16 optimizer steps in **481.1 s**; held-out greedy success
  **0.0% -> 100.0%** on 12 tasks. The 8-cycle SFT cold start did most of that:
  the first OPD rollout batch already scored 83.3%.
* **One-cycle smoke**: peak **4.251 GiB allocated / 4.762 GiB reserved**;
  strategy resolved to `resident` because a quantized model cannot be moved off
  the accelerator; projection chunk 256; 0 OOM retries; student 10,092,544
  trainable LoRA parameters; teacher 1,720,574,976 parameters.
* **Throughput probe**: NF4 11.19 tok/s, bf16 LoRA 12.84 tok/s (14.12 with
  determinism off). A 14-token prefill costs 37.0 ms and a cached one-token step
  costs 30.9 ms, so decoding here is **kernel-launch bound, not compute bound**.
* **Protocol-teacher GPU benchmark v2** (`gpu-calc-hard-equal-update-v2`,
  five arms x two prespecified seeds, 12 matched optimizer updates per
  continuation):

  | arm | seed 1234 | seed 20260727 | mean train s |
  | --- | ---: | ---: | ---: |
  | cold-start-only | 75.0% | 75.0% | 0.1 |
  | sft-continued | **100.0%** | **100.0%** | 86.4 |
  | opd-raw-teacher | **0.0%** | **0.0%** | 444.0 |
  | opd-privileged-context | **0.0%** | **0.0%** | 531.5 |
  | opd-protocol-sft-teacher | **100.0%** | **100.0%** | 523.8 |

  The protocol teacher prevents the collapse and ties SFT; it does not beat
  SFT. The complete portable schema-v2 artifact is
  `benchmarks/results/gpu-calc-hard-equal-update-v2.json`.
* **Matched-budget GPU benchmark** (`benchmarks/configs/gpu_calc_hard.yaml`,
  chained `hard` calculator split, legacy schema v1, single seed, 12 matched
  optimizer steps from one shared cold start). Result published in
  `benchmarks/results/rtx4080-calc-hard-matched.json`:

  | arm | steps | success |
  | --- | --- | --- |
  | cold-start-only | 0 | 62.5% |
  | sft-continued | 12 | **100.0%** |
  | opd-bucketed-k64 | 12 | **0.0%** |
  | opd-privileged-context | 12 | **0.0%** |

  **On-policy distillation lost, and the privileged-context prediction was
  wrong.** Decoding the final evaluations showed the two OPD arms fail
  differently: the standard teacher drops the opening `<tool_call>` tag and
  wraps answers in its own `<answer>` tags (83.3% invalid calls), while the
  privileged teacher emits clean output and no tool calls at all (1.00 turns,
  0.0% invalid) because it knows the answer and so never needs one.
  `scripts/attribute_failures.py` re-scores the collected trajectories with a
  lenient parser and shows formatting explains 4 of 24 failures for the standard
  arm and **none** for the privileged arm, so neither number is a verifier
  artifact. Full write-up in `docs/rtx4080-baselines.md`; both executions of the
  benchmark are reported and nothing was discarded.

* **Matched-budget CPU benchmark** (`recipes/benchmark_calc.yaml`, 7 arms x 2
  seeds, all 14 completed): published in
  `benchmarks/results/cpu-toy-calc-matched.json`. Changing the seed moves one
  arm by up to **58 points** (cold-start-only: 66.7% at seed 1234, 8.3% at seed
  20260727) while changing the objective moves it by at most 4 points at a fixed
  seed. This is a parity check -- it shows all seven arms including
  `exact_full_vocab` run to completion under identical budgets -- and the table
  is in `benchmarks/README.md` so a reader can see that no ranking is available.
  The first attempt of this benchmark restarted the cosine schedule and damaged
  every arm; that attempt is recorded but not published, because it measured a
  harness bug rather than any arm.

### Defects found and fixed during the build

Recorded because each one is a class of bug worth remembering.

| defect | how it was found | fix |
| --- | --- | --- |
| `src/miniverl/models/` excluded from git **and** the wheel by an unanchored `models/` rule in `.gitignore` | installing the wheel into a clean venv | anchored every directory rule; CI now enumerates the source tree against the wheel and fails on any git-ignored source file |
| Rich markup ate `[...]`, so the missing-extra hint printed `pip install "miniverl"` | reading actual CLI output | every dynamic string goes through `rich.markup.escape` |
| A bare install raised `ModuleNotFoundError` instead of naming the extra | a CLI test | the scorer import moved inside the constructor; the CLI pre-checks the stack |
| `exact_full_vocab` rejected a recipe that omitted `top_k` -- the remedy its own error suggested -- and then wrote a `config.resolved.yaml` that could not be re-read | an agent-written config test | consult `model_fields_set`, then normalize to 1 |
| `weighted_mean` rescaled the result for a tiny-but-positive weight sum | Hypothesis | floor only an exactly-zero sum |
| `EnvironmentConfig.name` was a hard-coded regex, making `register()` useless | writing the custom-environment example | validate against the registry |
| `render_final` could emit a block that parsed back truncated | an agent-written protocol test | escape closing tags in JSON, refuse them in plain text |
| `configure_logging(None)` silently downgraded `--log-level DEBUG` | an agent-written utils test | only an explicit level changes the level |
| The report dropped stored rollouts and printed `0.000 GiB` for CPU runs on a GPU machine | an agent-written reporting test | fill the display budget; decide on the run's device |
| Sampling drew from a CUDA tensor with a CPU generator | the first GPU smoke test | always draw on the CPU, which also makes a seed reproduce across devices |
| Resume replayed completed cycles with a different LR schedule | the resume-equivalence test | `train()` continues at the checkpointed cycle; the teacher fit is wrapped in `fork_rng` |
| The toy model could not learn to copy | measuring, not assuming | learned absolute positions -> RoPE; 48 traces -> 256 traces |
| `ArmResult.run_dir` wrote an absolute path, carrying a username and home directory into a file meant for pull requests | a new privacy test over `benchmarks/results/` | a validator reduces any path to its final component |
| Schema-v2 resolved configs and invocation still carried local adapter/run paths | inspecting the first complete GPU result before publication | recursively replace absolute provenance paths with portable `<local>/<name>` values before hashing or publishing; privacy and chart-source tests cover the result |
| `eval.enabled: false` did not suppress cycle-triggered evaluation | timing the first GPU benchmark attempt | periodic evaluation now checks both the interval and the enabled flag; the incomplete attempt is preserved under `runs/benchmarks/_aborted/` and the complete benchmark was rerun |
| The committed JSON schema was never checked against the model it is generated from, though `schema.py` claimed the two could not drift | writing the schema-validation test | a test asserts the committed file equals `json_schema()` |
| 16 bare `NaN` tokens in `metrics.jsonl` and `eval.json`, which strict JSON parsers reject | diffing two identical demo runs | `tokens_per_solved_task` is `null` when nothing was solved, via `schema.finite_or_none`; a test parses every artifact with `parse_constant` set to reject |
| `TerminationReason.ENVIRONMENT_ERROR` was declared but never assigned | auditing for dead public API | a raising tool now ends the policy rollout with it; the oracle path still raises, because a truncated oracle trace would silently degrade every SFT target |
| `log1mexp` clamped at a fixed `-1e-7`, silently saturating tail mass below float32 resolution regardless of dtype | the documentation pass, then Hypothesis in float64 | the clamp derives from `finfo(dtype).eps`; the data-processing-inequality property now models the `1 / q_tail` error amplification instead of using a flat tolerance |
| Both example scripts printed a bare `0.0%` with no context, which reads as a broken example | running them in a clean install | each prints the same "expected, not a capability claim" note that `miniverl demo` prints |

## Currently failing commands

None. Every gate in the table above passes.

## Unresolved design risks

1. **Protocol alignment removes collapse but has not produced an OPD-over-SFT
   advantage.** The protocol-trained teacher reached 100% strict competence
   before downstream selection and its OPD arm then reached 100% on both seeds,
   versus 0% twice for raw and privileged teachers. SFT also reached 100% twice.
   The result supports teacher-protocol competence as the explanation for the
   collapse, but the small saturated evaluation cannot rank protocol OPD above
   verified SFT.
2. **The toy backend cannot rank methods.** It solves only the `easy` split,
   where SFT saturates. Measured: `medium` and `hard` stay at 0% even after 700
   SFT steps. The CPU benchmark is therefore a parity check, not a ranking.
3. **Only two GPU seeds on one task and machine.** The primary v0.2 comparison
   repeats all arms twice, while legacy GPU artifacts remain single-seed. No
   significance or cross-task/model/hardware generalization is claimed.
4. **Transformers API drift.** CI now tests the Qwen3 introduction boundary
   (4.51.x) and the supported 5.x major. A future rename beyond the explicit
   `<6` cap would still need a reviewed compatibility change.

## Next three actions

1. Finish the clean-wheel/core and clean-source-training install gates, plus
   every executable README command that does not require rerunning the full GPU
   experiment.
2. Commit and push the measured artifacts/docs, make draft PR #4 ready, and
   repair any remote check that is an actual failure rather than an approval
   gate.
3. Merge only after green CI, verify public `main` and artifact hashes, then
   close superseded dependency PRs #1-#3. Do not tag or publish to PyPI.

## External blockers

* **Publishing to PyPI or creating a tag/release.** Not authorized for this v0.2
  iteration. The OIDC job is hard-disabled until a trusted publisher is
  registered and a separate reviewed change explicitly enables publication.
* **A GPU CI runner.** `.github/workflows/gpu.yml` is manual-dispatch only and
  targets a self-hosted `cuda` runner that does not exist for this repository.
  The prespecified GPU tests and benchmark therefore run locally on the RTX
  4080, with artifacts and checksums recorded for review.

## Final evidence table

See the [v0.2 release-hardening status](#v02-release-hardening-status) for
current evidence and the historical table for the v0.1 baseline. Known
limitations are enumerated in `docs/limitations.md`.
