# Release checklist

This is the release gate and publication record for miniVERL. A checked item
names an invariant exercised on the stated source. Publication begins only
after the exact release commit and its remote checks are green.

## v0.4.0 Consumer Runtime release

- [x] Preregistration revisions 1.1 and 1.2 were public before the sole
      headline run; the eager-attention and FP32-compute amendments retain the
      earlier SDPA and BF16 diagnostics as non-headline negative evidence.
- [x] All eight dual/shared by sequential/2/4/auto cells completed with one
      trajectory digest and one teacher-target digest; all 12 preregistered
      loss, full-gradient and post-update-logit equivalence gates passed.
- [x] The frozen result, profiler and Pareto SVG have SHA-256 values
      `a302da31af99f1d29f1efd4e6b3dbeb6ea4ac956bba102ca8a1bee8dff0319eb`,
      `66111cd7fc876cf1befea3297a1a51bcd99252c0bf8989c029381e1dc155a98b`
      and `98645a668a7832423d28b621262292619615917f037adf7219ff1bf071fb2fea`.
- [x] Typed batching, mask isolation, all three objectives, shared adapter-role
      restoration, optimizer isolation, one-base loading, reference isolation
      and standard student export have focused CPU and tiny-HF coverage.
- [x] The legacy calculator JSON remains byte-identical at SHA-256
      `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`;
      v0.3 artifacts and negative results are unchanged.
- [x] Ruff check/format, mypy, actionlint, Markdown links, generated SVG byte
      comparison and `git diff --check` pass; native and 820-pixel SVG renders
      were visually inspected.
- [x] Full non-GPU/non-network suite: **1448 passed**, 6 deselected, **86.26%**
      branch coverage.
- [x] Available RTX 4080 GPU suite: **5 passed**, 1449 deselected.
- [x] Network suite: **3 passed**, 1451 deselected, including all immutable
      public adapters used by the current tests.
- [x] Wheel and sdist build and pass Twine; a clean Python 3.10 wheel install
      reports `miniverl 0.4.0.dev0` and passes the core command smoke tests.
- [x] Focused PR #30 is green and merged as `9914c6d`; synchronized main CI
      and build runs `30776659196` and `30776659178` are green.
- [x] Release metadata declares exact `0.4.0`; the intended annotated tag is
      exactly `v0.4.0` and will use the existing OIDC-only release workflow.

## v0.3.0 RecoveryBench release

- [x] Revision-1.3 preregistration predates the valid final run and remains
      byte-frozen at SHA-256
      `9c4c2ec19a56cebb2b2c1c0f3c7e504a9285467c99ae1590488251fbf2ff3934`.
- [x] All three schema-v3 budget results validate; 36 task files, 4,608
      trajectories, 128 task records per arm, task pairing, cold checkpoints,
      frozen datasets and all 75 fresh-policy update versions were audited.
- [x] Negative results, failed teacher candidates, the invalid v1.2 partial
      run and the post-run aborted wall-budget replacement are disclosed and
      excluded from headline analysis.
- [x] Equal-updates, selected-position and cycle-capped wall diagnostics are
      reported without changing the frozen experiment or analysis after seeing
      outcomes.
- [x] The five RecoveryBench publication artifacts and three SVGs have exact
      hash tests; SVG generator output byte-matches the committed figures.
- [x] The six-page technical PDF is data-bound, reproducible byte-for-byte and
      visually inspected on every page; the three SVGs and banner are visually
      clean at their intended sizes.
- [x] The legacy calculator JSON remains byte-identical at SHA-256
      `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`.
- [x] Ruff check/format, mypy, actionlint and `git diff --check` pass.
- [x] Full non-GPU/non-network suite: **1408 passed**, 6 deselected, **86.68%**
      branch coverage.
- [x] Available RTX 4080 GPU suite: **5 passed**, 1409 deselected.
- [x] Network suite: **3 passed**, 1411 deselected, including both immutable
      public teacher adapters.
- [x] Development wheel and sdist build; both pass Twine. Clean Python 3.10
      core and clean Python 3.12 training-extra installs pass their command and
      demo/report/cache smoke tests.
- [x] Extracted sdist passes PyPI-description, Ruff, format, mypy and all 1408
      CPU tests; its rebuilt wheel has the same 81-file package inventory.
- [x] Python 3.10.11 minimum boundary passes with Torch 2.3.1+cpu,
      Transformers 4.51.3, PEFT 0.12.0, Accelerate 0.33.0, NumPy 1.24.4 and
      bitsandbytes 0.43.3. Python 3.13.13 latest boundary passes with Torch
      2.13.0+cpu, Transformers 5.14.1, PEFT 0.20.0, Accelerate 1.14.0, NumPy
      2.5.1 and bitsandbytes 0.50.0.
- [x] Focused PR #27 is green and merged as `bee82d3`; synchronized `main` CI
      and build workflows are green.
- [x] Release metadata declares exact `0.3.0`; the intended annotated tag is
      exactly `v0.3.0` and will use the existing OIDC-only release workflow.

## After the tag

- [x] OIDC publication, public hashes/attestations, clean install and one
      GitHub Release were verified from the same distributions by tag run
      [`30772772078`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30772772078).
- [x] This post-release state-sync change advances development to
      `0.4.0.dev0`; it merges only after remote checks are green.

Published artifact identity:

- `miniverl-0.3.0-py3-none-any.whl`: SHA-256
  `e42404ced88b75ba4ff31541cb3df697da0eee09a68f603e4eadbf69a11d2032`
- `miniverl-0.3.0.tar.gz`: SHA-256
  `006ce418243286dd27e0731090bf9fa1711a1abc962ecfcc4d39807257539cb2`

## Historical v0.2.6 release record

### Version consistency

- [x] Tagged source `v0.2.6` declares package version `0.2.6`.
- [x] `CHANGELOG.md` has a dated `## [0.2.6]` section.
- [x] `CITATION.cff` version and release date match.
- [x] The intended annotated tag is exactly `v0.2.6`.
- [x] The package/project name remains `miniverl`.

### Correctness and lifecycle

- [x] Strict model-output JSON rejects non-finite values, duplicate keys,
      oversized integers, excessive depth/members and invalid surrogates before
      environment execution; protocol-v1 byte fixtures remain unchanged.
- [x] Calculator verifier-v2 requires a complete finite answer and compatible
      units while verifier-v1 remains identifiable for historical artifacts.
- [x] Calculator, JSON-navigation and SQLite direct/rollout/property tests turn
      arbitrary strings into bounded verification results; non-finite SQLite
      answers are malformed rather than process exceptions.
- [x] Deterministic Event-based regressions prove train/evaluate/checkpoint
      save/load/close exclusion, no-mutation losing operations, later and
      repeated close, and READY/terminal/evaluation-only lifecycle behavior.
- [x] Windows multiprocessing regressions prove same-run exclusion before model
      loading, report completion under lock, checkpoint selection under lock,
      bounded timeout, killed-owner recovery and different-run progress.
- [x] Public checkpoint load checks READY before and after exclusive ownership;
      load validation failure preserves model/progress/state and releases the
      guard, while exact resume behavior remains unchanged.
- [x] Evaluation restores the exact prior model mode after success and injected
      failures in rollout, trajectory write, diagnostics, metrics and events.
- [x] Numerical property tests for exact/bucketed objectives, weighted
      reductions, chunking, finite gradients, zero tails and OOM RNG
      equivalence remain green.

### Quality gates

- [x] `git diff --check`.
- [x] `ruff check .`.
- [x] `ruff format --check .`.
- [x] `mypy src/miniverl`.
- [x] Full non-GPU/non-network pytest suite passes with branch coverage above
      the required 80%.
- [x] All GPU tests pass on an NVIDIA GeForce RTX 4080.
- [x] All opt-in network tests pass.
- [x] Transformers 4.51.x and 5.x compatibility checks pass on the PR head.
- [x] The declared minimum Python 3.10 training dependency bundle passes the
      no-network toy/HF contract.
- [x] The current Python 3.13 training dependency bundle passes the same
      contract.
- [x] Core Python 3.10, 3.11, 3.12 and 3.13 checks pass on the PR head.
- [x] `actionlint` passes with the repository's `cuda` self-hosted label
      declared in `.github/actionlint.yaml`.
- [x] No unfinished implementation markers remain under `src`, `tests`,
      `examples` or `scripts`.

### Packaging and clean installs

- [x] A clean `python -m build` produces one `0.2.6` wheel and one sdist.
- [x] `python -m twine check dist/*` passes.
- [x] The wheel contains the report template and no tests.
- [x] The sdist contains the full shipped test surface, including scripts and
      workflow fixtures, and its complete non-GPU/non-network suite passes from
      an extracted directory outside the checkout.
- [x] A wheel rebuilt from the extracted sdist has the same runtime-package
      inventory as the repository wheel.
- [x] A clean core-only wheel install runs `--help`, `--version` and
      `doctor --json` with torch, Transformers, PEFT and bitsandbytes absent
      and unimported.
- [x] A clean install of the same wheel with `[train]` runs demo, inspect,
      report and weights-only standalone eval.
- [x] Reusing the demo output without `--overwrite` fails without changing a
      file; explicit overwrite produces a fresh completed run.

### Artifacts, documentation and hygiene

- [x] Every shipped run recipe validates; every benchmark config resolves.
- [x] The committed benchmark JSON Schema is byte-identical to generated
      output and all published results validate.
- [x] The frozen
      `benchmarks/results/gpu-calc-hard-equal-update-v2.json` SHA-256 remains
      `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`.
- [x] The benchmark SVG remains generated from and bound to that exact JSON.
- [x] All tracked JSON/JSONL parses strictly.
- [x] README and documentation Markdown links pass link checking.
- [x] `PYPI.md` byte-matches its generator, stable repository links use
      `v0.2.6`, and built wheel metadata contains no relative or local links.
- [x] Shareable HTML/Markdown/JSON/YAML and benchmark exports contain no fake
      credential/path sentinel; benign tokenizer/count metadata remains intact.
- [x] No model weights, checkpoints, caches or databases are tracked.
- [x] The banner and benchmark SVG were rendered and visually inspected.
- [x] Benchmark grids begin below their tick labels; the generated dark SVG is
      still bound to the immutable source JSON.
- [x] The supported single-GPU recipe uses automatic bf16/fp16 selection and
      carries no GPU model or VRAM-tier tag.
- [x] The base-vs-`[train]` installation split and v1 scientific confound are
      stated explicitly.
- [x] `PROJECT_STATE.md`, the compatibility policy, support claims and
      dependency boundaries describe the validated implementation.
- [x] No RecoveryBench or unrelated feature expansion was added.

### Trusted publishing readiness

- [x] PyPI reports project `miniverl`; `v0.2.6` is the current public version.
- [x] GitHub environment `pypi` exists and has a deployment branch policy.
- [x] `release.yml` requests `id-token: write`, uses the `pypi` environment and
      publishes only on a tag push.
- [x] The maintainer registered the pending publisher for
      `DaoyuanLi2816/mini-verl`, workflow `release.yml`, environment `pypi`.
- [x] The immutable `v0.2.0` through `v0.2.6` tags, frozen calculator JSON and
      pinned public protocol-teacher adapter are unchanged.

### After the tag

- [x] Annotated tag `v0.2.6` resolves to exact validated merge commit
      `59fe738709526a13f354a744ab763f13530de4d1`.
- [x] Tag workflow
      [`30722451004`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30722451004)
      passed metadata, tests, one-time build, OIDC publication, attestations,
      public verification and GitHub Release creation.
- [x] Public PyPI and GitHub Release wheel/sdist hashes match workflow
      artifacts.
- [x] This green-gated state-sync change advances development to
      `0.2.7.dev0`.

Publication completed on 2026-08-01 UTC:

- [`miniverl 0.2.6`](https://pypi.org/project/miniverl/0.2.6/) exposes Trusted
  Publishing provenance for both distributions.
- [`miniVERL v0.2.6`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.6)
  contains the byte-identical wheel, sdist and `SHA256SUMS`.
- An independent no-cache Windows Python 3.10 install from the public PyPI
  index reported `miniverl 0.2.6`, passed `doctor`, and kept torch,
  Transformers, PEFT and bitsandbytes absent.

Published artifact identity:

- `miniverl-0.2.6-py3-none-any.whl`: SHA-256
  `11d6b001752c41a0100f12c29b125a9dc082703dbeadc6b0317a88ac818d8695`
- `miniverl-0.2.6.tar.gz`: SHA-256
  `91e7b2918286c342cacaf2582dbed57c1e7a1bf4e1064d327e349b1d77c28886`

## Historical v0.2.5 record

- [x] Annotated tag `v0.2.5` resolves to exact validated merge commit
      `a9a84510741b4ade8a405c100affdf1caed55ae6`.
- [x] Tag workflow
      [`30611603505`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30611603505)
      passed metadata, tests, one-time build, OIDC publication, attestations,
      public verification and GitHub Release creation.
- [x] Public PyPI and GitHub Release wheel/sdist hashes match workflow
      artifacts.
- [x] This green-gated state-sync change advances development to
      `0.2.6.dev0`.

Publication completed on 2026-07-31 UTC:

- [`miniverl 0.2.5`](https://pypi.org/project/miniverl/0.2.5/) exposes Trusted
  Publishing provenance for both distributions.
- [`miniVERL v0.2.5`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.5)
  contains the byte-identical wheel, sdist and `SHA256SUMS`.
- An independent no-cache Windows Python 3.10 install from the public PyPI
  index reported `miniverl 0.2.5`, passed `doctor`, and kept torch,
  Transformers, PEFT and bitsandbytes absent.

Published artifact identity:

- `miniverl-0.2.5-py3-none-any.whl`: SHA-256
  `70c98284bce151fc74b508047b354929846efb71c3fe8f451c0d0ba1bec48e9d`
- `miniverl-0.2.5.tar.gz`: SHA-256
  `d30bb07ebca676a3960d4b5c46075a8a2e13e58629b96984e30f8f7bab67dce0`

## Historical v0.2.4 record

- [x] Annotated tag `v0.2.4` resolves to exact validated commit
      `57dec193af88b462dcc41d82fc6fecb813e161fd`.
- [x] Tag run
      [`30522484949`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30522484949)
      passed metadata, tests, one-time build, OIDC publication and attestation
      generation. Recovery run
      [`30524088015`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30524088015)
      verified the original artifacts, public hashes/attestations and clean
      install, then created the GitHub Release without rebuilding or uploading.
- [x] Public PyPI and GitHub Release wheel/sdist hashes match the original
      workflow artifacts.
- [x] Development advances to `0.2.5.dev0` through a green state-sync PR.

Publication completed on 2026-07-30:

- [`miniVERL v0.2.4`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.4)
  contains the byte-identical wheel, sdist and `SHA256SUMS`.
- An independent no-cache Windows Python 3.10 install from the public PyPI
  index reported `miniverl 0.2.4`, passed `doctor`, and kept torch,
  Transformers, PEFT and bitsandbytes absent.

Published artifact identity:

- `miniverl-0.2.4-py3-none-any.whl`: SHA-256
  `3f5a239bbbd2f85217cf11f691fbb63f647092f67b82da4de38bd6907c5ab0f1`
- `miniverl-0.2.4.tar.gz`: SHA-256
  `03f0e844df2c91deed5c211cdd2dd598d22f03d59d99cd8e792a58211c0b2296`

## Historical v0.2.3 record

Publication completed on 2026-07-29:

- [x] Annotated tag `v0.2.3` resolves to exact validated commit
      `38924da743180e6767f1e3b252feafdccd70759b`.
- [x] Release run
      [`30513947051`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30513947051)
      passed metadata, tests, build, OIDC publication, public verification and
      GitHub Release creation.
- [x] Public PyPI hashes and Trusted Publishing attestations match the workflow
      artifacts; clean workflow Python 3.12 and independent Windows Python 3.10
      core installs reported `miniverl 0.2.3`.
- [x] The
      [`miniVERL v0.2.3`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.3)
      GitHub Release contains the byte-identical wheel, sdist and `SHA256SUMS`.
- [x] Development advanced to `0.2.4.dev0`.

Published artifact identity:

- `miniverl-0.2.3-py3-none-any.whl`: SHA-256
  `033e51bfbdae20a91d942ef7a5c22ef6c8a00317cc9b775b102d303f2e1a6619`
- `miniverl-0.2.3.tar.gz`: SHA-256
  `6f7d20fd4b4a90e6a3fe1e97c9ced26268e013bb87462ba75a7d09510bd2f011`

## Historical v0.2.2 record

Publication completed on 2026-07-29:

- [x] Annotated tag `v0.2.2` resolves to exact validated commit
      `518590cb43ff788fa65f73ee9cf3a7afb6dfba5a`.
- [x] Release run
      [`30494182647`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30494182647)
      passed metadata, tests, build, OIDC publication, public verification and
      GitHub Release creation.
- [x] Public PyPI hashes and Trusted Publishing attestations passed; clean
      Python 3.10 and workflow Python 3.12 core installs reported
      `miniverl 0.2.2`.
- [x] The
      [`miniVERL v0.2.2`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.2)
      GitHub Release contains the byte-identical wheel, sdist and `SHA256SUMS`.
- [x] Development advanced to `0.2.3.dev0`.

Published artifact identity:

- `miniverl-0.2.2-py3-none-any.whl`: SHA-256
  `1ead97173bb11ce3da963b94f628df825a5b14648fed488cf4d88c47cba9dd59`
- `miniverl-0.2.2.tar.gz`: SHA-256
  `3951dd4addc5d85b3e58ce72ecffac65c38bf2eab951d2c08cce8f20c886185c`

The first two install-verification attempts saw PyPI's JSON/file APIs before
the public simple index had propagated. Attempt 3 passed unchanged. The
post-release workflow now retries that final public-index install so future
releases do not report this expected propagation window as a package defect.

## Historical v0.2.1 record

Publication completed on 2026-07-29:

- [x] Create annotated tag `v0.2.1` on exact validated commit
      `591881b0d094f5c53ff47a9419e679b762fb44b0`.
- [x] Verify release run
      [`30474597179`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30474597179)
      tests and builds the distributions once.
- [x] Verify OIDC publication, public PyPI hashes and Trusted Publishing
      attestations for
      [`miniverl 0.2.1`](https://pypi.org/project/miniverl/0.2.1/).
- [x] Verify the
      [`miniVERL v0.2.1`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.1)
      GitHub Release contains the same wheel, sdist and
      `SHA256SUMS`.
- [x] Install public `miniverl==0.2.1` in a clean Windows environment and run
      `miniverl --version` plus `miniverl doctor --json`; the core verdict
      passed and Torch remained absent.
- [x] Open the post-release state-sync change and advance development to
      `0.2.2.dev0`.

Published artifact identity:

- `miniverl-0.2.1-py3-none-any.whl`: SHA-256
  `0177d50026da86047c2a03f90e7786c794b26c5b0d6fef193c58ed35c08d8cda`
- `miniverl-0.2.1.tar.gz`: SHA-256
  `80f890c1ab8be0ccdf6c5ce293a5c4d7bb6a6f7ab7a57db34090384fcaa7e16c`

Independent downloads from PyPI and the GitHub Release reproduced both
digests.

## Historical v0.2.0 record

Tag workflow
[`30421231859`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30421231859)
published `v0.2.0` from commit
`6092706b4a4e750c4571d7d6a7decbc26af851b2` on 2026-07-28
(2026-07-29 UTC).

- Wheel SHA-256:
  `cf850a6333483a3ee22c0c0e98df1e1b2e6faa184480573e0666658b53a29262`
- Sdist SHA-256:
  `3d5107b4f6351204335f800ce924208843f08f54441378bd9f25c3c6fa17456b`
- PyPI:
  [`miniverl 0.2.0`](https://pypi.org/project/miniverl/0.2.0/)
- GitHub:
  [`miniVERL v0.2.0`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.0)
