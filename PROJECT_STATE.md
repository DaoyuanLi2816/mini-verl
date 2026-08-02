# PROJECT_STATE

Living build log for **miniVERL** (`mini-verl` / `miniverl` / CLI `miniverl`).
A checkbox is not evidence: every completed item names the command that was run
and what it printed.

Last updated: 2026-08-01.

## v0.3.0 RecoveryBench development

| item | current state |
| --- | --- |
| audited baseline | fetched all remotes and started `agent/v0.3-recoverybench` from clean public `main` at `6c79c1fcc5a6e55dd9f4af843d446ece1c454431`; no PR was open, main CI/build were green, and development advanced directly from `0.2.7.dev0` to `0.3.0.dev0` without a `0.2.7` release |
| available execution environment | Windows checkout with Python 3.12, Torch 2.13.0+cu130 and an NVIDIA GeForce RTX 4080 (16376 MiB); GitHub and Hugging Face access are available and publication remains restricted to exact validated tags |
| immutable baseline | calculator benchmark SHA-256 is `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`; calculator protocol-teacher HEAD is `23323751318135484c06c043b1f9b9e7016dd89f`; existing tags remain untouched |
| phase boundary | RecoveryBench must be preregistered, fully measured with the sequential engine, frozen, merged, released as `v0.3.0`, publicly verified and state-synced to `0.4.0.dev0` before padded batching begins |
| current highest-risk work | final experiment, artifact audit and data-bound publication are complete; remaining work is release validation, integration, Trusted Publishing verification and the required `0.4.0.dev0` state sync |
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
| immutable RecoveryBench results | equal-updates `6ce2e6837e12b99ebc4fad6d27ce3e69c92e295ff3b9b60e0f68c2d308022384`; equal-selected-positions `fe4c9afc799724dfe7a32e631676a1e5177c44559a7374d2ea31da135354f137`; wall diagnostic `425b0fa568f37b09e61af731d3da5009bd3833bddde6efaf2c66e9dba8355cbe`; task JSONL `76ab53202f8ad1eb332b056c9c840eb34816986883a568813edc0e0f502d3086`; paired analysis `c0e7b8c9e8da9a0d0a5d64a17a688c45e3dbbd1c3b68074249b31fc10f0baeca` |
| technical publication | the generated Markdown analysis, three SVGs and deterministic six-page PDF report are data-bound to the frozen JSON; native/README-width SVG inspection and every-page PDF inspection found no clipping or overlap |

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
