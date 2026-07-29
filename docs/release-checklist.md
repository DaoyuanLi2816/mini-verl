# Release checklist

This is the release gate and publication record for `v0.2.0`. A checked box
means the command was executed on the v0.2 source or the named invariant was
inspected. The `release` workflow re-runs the mechanical gates and fails a tag
if any item before *After the tag* is left unchecked.

## Version consistency

- [x] `src/miniverl/__init__.py` `__version__` is the release version.
- [x] `CHANGELOG.md` has a `## [<version>]` section with a date.
- [x] `CITATION.cff` `version:` matches.
- [x] The git tag will be `v<version>`.

Checked by: `.github/workflows/release.yml`, job `validate-and-test`.

## Code quality

- [x] `ruff check .`
- [x] `ruff format --check .`
- [x] `mypy src/miniverl`
- [x] `pytest -q -m "not gpu and not network"` with the coverage gate
      (the exact v0.2 count and coverage are recorded in `PROJECT_STATE.md`)
- [x] `pytest -q -m gpu` on a machine with CUDA
- [x] `rg -n "TODO|FIXME|XXX|HACK|NotImplementedError" src tests examples scripts`
      has no unfinished implementation markers; its sole match is the
      intentional compatibility catch for older model APIs in
      `models/adapters.py`.

## Packaging

- [x] `python -m build` produces both an sdist and a wheel.
- [x] `python -m twine check dist/*` passes.
- [x] The wheel contains `miniverl/reporting/templates/report.html.j2` and does
      **not** contain `tests/`.
- [x] The sdist contains `recipes/`, `tests/`, `LICENSE` and `README.md`.
- [x] A fresh virtual environment installing only the wheel can run
      `miniverl --help`, `miniverl --version` and `miniverl doctor --json`, with
      torch absent.
- [x] A second fresh environment with `[train]` can run `miniverl demo --fast`
      end to end.

## Documentation

- [x] Every command in `README.md` has been executed; placeholder forms were
      instantiated with the concrete toy/Qwen run, benchmark and adapter paths
      recorded in `PROJECT_STATE.md`.
- [x] `README.zh-CN.md` describes the same behaviour as `README.md`.
- [x] Every number in the README and in `docs/` is measured, or marked
      "not measured" / "not run".
- [x] The verl disclaimer is present and accurate.
- [x] Badges point at workflows that exist. The PyPI badge was added only after
      the public `0.2.0` publication was independently verified.
- [x] `docs/limitations.md` is current and does not undersell anything.

## Artifacts and hygiene

- [x] `git status --porcelain` is empty at the release-candidate commit.
- [x] No model weights, checkpoints, Hugging Face caches, databases, secrets or
      absolute user paths are committed
      (`git ls-files | rg "\.(safetensors|bin|pt|ckpt|sqlite|db)$"` is empty).
- [x] `benchmarks/schema/benchmark-result.schema.json` matches
      `miniverl schema`.
- [x] Every file in `benchmarks/results/` validates against that schema.

## Publishing (completed)

The workflow stores no secret or long-lived PyPI token. Its publish, public
verification and GitHub Release jobs each independently require a `push` event
whose ref starts with `refs/tags/v`; `workflow_dispatch` validates, tests,
builds and uploads a workflow artifact but can never publish.

The exact pending publisher was registered with project `miniverl`, owner
`DaoyuanLi2816`, repository `mini-verl`, workflow `release.yml` and environment
`pypi`. The tag was pushed only after that account-side registration was
confirmed.

Tag run
[`30421231859`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30421231859)
completed successfully on 2026-07-28 (2026-07-29 UTC):

1. `v0.2.0` resolved to release commit
   `6092706b4a4e750c4571d7d6a7decbc26af851b2`.
2. The full quality gate passed and the wheel and sdist were built exactly once.
3. OIDC Trusted Publishing created
   [`miniverl 0.2.0`](https://pypi.org/project/miniverl/0.2.0/) without a token.
4. Public PyPI metadata, file hashes and repository-identity attestations passed,
   followed by a clean public install and CLI exercise.
5. Only then did the workflow create
   [`miniVERL v0.2.0`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.0)
   with the verified wheel, sdist and `SHA256SUMS`.

### Published artifact identity

- `miniverl-0.2.0-py3-none-any.whl`:
  SHA-256 `cf850a6333483a3ee22c0c0e98df1e1b2e6faa184480573e0666658b53a29262`
- `miniverl-0.2.0.tar.gz`:
  SHA-256 `3d5107b4f6351204335f800ce924208843f08f54441378bd9f25c3c6fa17456b`

Independent downloads from both PyPI and the GitHub Release reproduced those
digests. PyPI reports Trusted Publishing and an attestation publisher of
`release.yml` on `DaoyuanLi2816/mini-verl`.

## After the tag

- [x] Set the GitHub repository description to
      `On-policy distillation for tool-using LLM agents on one consumer GPU.`
- [x] Set the topics: `llm`, `on-policy-distillation`, `knowledge-distillation`,
      `agentic-rl`, `tool-use`, `qlora`, `consumer-gpu`, `post-training`, `qwen`,
      `llm-agents`, `verl`.
- [ ] Upload `docs/banner.svg` (rendered to PNG) as the social preview image.
- [x] Create the GitHub release from the verified release distributions and
      generated notes.

The remaining social-preview upload is a repository-settings action. A
1280×640 PNG was rendered and inspected; the `release` workflow only inspects
the sections above.
