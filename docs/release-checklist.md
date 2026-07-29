# Release checklist

This is the release gate and publication record for `v0.2.2`. A checked item
names an invariant that was exercised on the release source. The tag workflow
repeated the mechanical gates and refused inconsistent metadata or an
unchecked pre-tag item. Publication began only after explicit maintainer
authorization.

## Version consistency

- [x] `src/miniverl/__init__.py` is `0.2.2`.
- [x] `CHANGELOG.md` has a dated `## [0.2.2]` section.
- [x] `CITATION.cff` version and release date match.
- [x] The future annotated tag is exactly `v0.2.2`.
- [x] The package/project name remains `miniverl`.

## Correctness and lifecycle

- [x] New-run, collision, concurrent creation, overwrite, resume and stale
      JSONL-append regressions pass.
- [x] Atomic checkpoint save, state-based selection, checksum corruption,
      missing weights, identity mismatch, legacy v0.2 reads and weights-only
      standalone evaluation regressions pass.
- [x] Exact uninterrupted/resumed OPD and offline-KD tests pass; offline resume
      reuses the persisted dataset, order and teacher cache.
- [x] Completed, failed and interrupted manifest tests pass with actual progress
      and OOM state.
- [x] RNG-restored gradient OOM retry, dropout equivalence, single optimizer
      commit, optimizer-OOM and non-OOM propagation tests pass.
- [x] Protocol v1 byte stability, parser-valid v2 examples, render/parse
      round-trips and protocol-aware adapter competence gates pass.
- [x] Per-token objective/CE/divergence accounting and precise emitted,
      parsed, executed, error and final-answer metric tests pass.
- [x] Exact-zero tails, ordered span types, checksum policy, adapter identity,
      full-vocabulary exactness and lossy-dtype rejection tests pass.
- [x] Structural/revision tokenizer identity, output-vocabulary compatibility,
      padded vocabulary and fail-before-mutation model-state tests pass.
- [x] No-op, success, failure, replay and resume parameter-version tests pass.
- [x] Dynamic reset observation/state provenance and exactly-once reset tests
      pass.

## Quality gates

- [x] `git diff --check`.
- [x] `ruff check .`.
- [x] `ruff format --check .`.
- [x] `mypy src/miniverl`.
- [x] Full non-GPU/non-network pytest suite passes with branch coverage above
      the required 80%.
- [x] All GPU tests pass on an NVIDIA GeForce RTX 4080.
- [x] All opt-in network tests pass.
- [x] Transformers 4.51.x and 5.x compatibility tests pass.
- [x] The declared minimum Python 3.10 training dependency bundle passes the
      no-network toy/HF contract.
- [x] The current Python 3.13 training dependency bundle passes the same
      contract.
- [x] `actionlint` passes with the repository's `cuda` self-hosted label
      declared in `.github/actionlint.yaml`.
- [x] No unfinished implementation markers remain under `src`, `tests`,
      `examples` or `scripts`.

## Packaging and clean installs

- [x] A clean `python -m build` produces one `0.2.1` wheel and one sdist.
- [x] `python -m twine check dist/*` passes.
- [x] The wheel contains the report template and no tests.
- [x] The sdist contains recipes, tests, docs, license and both READMEs.
- [x] A clean core-only wheel install runs `--help`, `--version` and
      `doctor --json` with torch, Transformers, PEFT and bitsandbytes absent
      and unimported.
- [x] A clean install of the same wheel with `[train]` runs demo, inspect,
      report and weights-only standalone eval.
- [x] Reusing the demo output without `--overwrite` fails without changing a
      file; explicit overwrite produces a fresh completed run.

## Artifacts, documentation and hygiene

- [x] Every shipped run recipe validates; every benchmark config resolves.
- [x] The committed benchmark JSON Schema is byte-identical to generated
      output and all published results validate.
- [x] The frozen
      `benchmarks/results/gpu-calc-hard-equal-update-v2.json` SHA-256 remains
      `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`.
- [x] The benchmark SVG remains generated from and bound to that exact JSON.
- [x] All tracked JSON/JSONL parses strictly.
- [x] README and documentation Markdown links pass link checking.
- [x] Public artifacts contain no real absolute user path, username, hostname
      or secret.
- [x] No model weights, checkpoints, caches or databases are tracked.
- [x] The banner and benchmark SVG were rendered and visually inspected.
- [x] Benchmark grids begin below their tick labels; the generated dark SVG is
      still bound to the immutable source JSON.
- [x] The supported single-GPU recipe uses automatic bf16/fp16 selection and
      carries no GPU model or VRAM-tier tag.
- [x] The base-vs-`[train]` installation split and v1 scientific confound are
      stated explicitly.
- [x] `TODO.md`, `PROJECT_STATE.md`, support claims and dependency boundaries
      describe the release-candidate implementation.
- [x] No RecoveryBench or unrelated feature expansion was added.

## Trusted publishing readiness

- [x] PyPI reports project `miniverl` and current public version `0.2.1`.
- [x] GitHub environment `pypi` exists and has a deployment branch policy.
- [x] `release.yml` requests `id-token: write`, uses the `pypi` environment and
      publishes only on a tag push.
- [x] The maintainer registered the pending publisher for
      `DaoyuanLi2816/mini-verl`, workflow `release.yml`, environment `pypi`.
- [x] The immutable `v0.2.0` and `v0.2.1` tags and public artifacts are unchanged.

## After the tag

Complete and check these in the post-release state-sync change:

- [ ] Record the exact `v0.2.2` tag commit and release workflow run.
- [ ] Verify public PyPI hashes, attestations and a clean core installation.
- [ ] Verify the GitHub Release contains the identical distributions.
- [ ] Advance development to `0.2.3.dev0`.

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
