# Release checklist

This is the release-candidate gate for `v0.2.0`. A checked box means the command
was executed on the v0.2 source or the named invariant was inspected. The
`release` workflow re-runs the mechanical gates and fails a tag if any item
before *After the tag* is left unchecked.

## Version consistency

- [x] `src/miniverl/__init__.py` `__version__` is the release version.
- [x] `CHANGELOG.md` has a `## [<version>]` section with a date.
- [x] `CITATION.cff` `version:` matches.
- [x] The git tag will be `v<version>`.

Checked by: `.github/workflows/release.yml`, job `validate`.

## Code quality

- [x] `ruff check .`
- [x] `ruff format --check .`
- [x] `mypy src/miniverl`
- [x] `pytest -q -m "not gpu and not network"` with the coverage gate
      (the exact v0.2 count and coverage are recorded in `PROJECT_STATE.md`)
- [x] `pytest -q -m gpu` on a machine with CUDA
- [x] `rg -n "TODO|FIXME|XXX|HACK|NotImplementedError" src tests examples scripts`
      returns nothing

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

- [ ] Every command in `README.md` has been executed.
- [x] `README.zh-CN.md` describes the same behaviour as `README.md`.
- [x] Every number in the README and in `docs/` is measured, or marked
      "not measured" / "not run".
- [x] The verl disclaimer is present and accurate.
- [x] Badges point at workflows that exist. **No PyPI badge until the package is
      actually published.**
- [x] `docs/limitations.md` is current and does not undersell anything.

## Artifacts and hygiene

- [ ] `git status --porcelain` is empty.
- [x] No model weights, checkpoints, Hugging Face caches, databases, secrets or
      absolute user paths are committed
      (`git ls-files | rg "\.(safetensors|bin|pt|ckpt|sqlite|db)$"` is empty).
- [x] `benchmarks/schema/benchmark-result.schema.json` matches
      `miniverl schema`.
- [x] Every file in `benchmarks/results/` validates against that schema.

## Publishing (externally blocked)

The OIDC publication job exists but is deliberately disabled with `if: false`;
this repository stores no secrets. Live verification on 2026-07-28 found:

- `https://pypi.org/pypi/miniverl/json` returns HTTP 404, so no project exists
  on which a trusted publisher could already be registered;
- `gh api repos/DaoyuanLi2816/mini-verl/environments` returns
  `{"total_count":0,"environments":[]}`, so the required GitHub `pypi`
  environment is not configured.

The absence of either external object means a workflow file alone is not
trusted-publisher configuration. Do not remove `if: false`, create `v0.2.0`, or
create a GitHub Release until these steps are completed:

1. On PyPI, create the `miniverl` project and add a **trusted publisher**:
   - Owner: `DaoyuanLi2816`
   - Repository: `mini-verl`
   - Workflow: `release.yml`
   - Environment: `pypi`
2. In a separately reviewed change, remove the hard-disabled condition from
   `publish-pypi` in `release.yml`.
3. Create the GitHub `pypi` environment named exactly as above.
4. Push the tag. The `validate` job must pass before `publish-pypi`.
5. Verify the public PyPI page and a fresh wheel install before creating the
   GitHub Release.

Until the publisher is registered and publication is explicitly authorized,
the release workflow validates and builds but cannot upload.

### Name availability

`miniverl` was available on PyPI when this release was prepared
(`https://pypi.org/pypi/miniverl/json` returned HTTP 404 on 2026-07-28).
**Re-check immediately before publishing.** If it has been taken, publish as
`mini-verl-opd` and keep the display name `miniVERL`, the import package
`miniverl`, the CLI command `miniverl` and the repository name `mini-verl`
unchanged; only `[project] name` in `pyproject.toml` changes.

## After the tag

- [ ] Set the GitHub repository description to
      `On-policy distillation for tool-using LLM agents on one consumer GPU.`
- [ ] Set the topics: `llm`, `on-policy-distillation`, `knowledge-distillation`,
      `agentic-rl`, `tool-use`, `qlora`, `consumer-gpu`, `post-training`, `qwen`,
      `llm-agents`, `verl`.
- [ ] Upload `docs/banner.svg` (rendered to PNG) as the social preview image.
- [ ] Create the GitHub release from `CHANGELOG.md`.

These four are repository-settings actions that cannot be performed from a
commit, so they stay unchecked here on purpose; the `release` workflow only
inspects the sections above.
