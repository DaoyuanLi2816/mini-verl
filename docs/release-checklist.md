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

Checked by: `.github/workflows/release.yml`, job `validate-and-test`.

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

The workflow stores no secret or long-lived PyPI token. Its publish, public
verification and GitHub Release jobs each independently require a `push` event
whose ref starts with `refs/tags/v`; `workflow_dispatch` validates, tests,
builds and uploads a workflow artifact but can never publish.

Live verification on 2026-07-28 found:

- `https://pypi.org/pypi/miniverl/json` returns HTTP 404;
- the GitHub `pypi` environment exists and its custom deployment policy permits
  only tags matching `v*`;
- no `v0.2.0` tag, PyPI release or GitHub Release exists;
- **PyPI pending publisher not yet registered** is the sole external
  publication blocker.

PyPI's
[pending-publisher documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
supports creating a new project on the first OIDC upload. A project page does
not need to exist first, and no token-based bootstrap upload is needed. The
exact bootstrap is:

1. Sign in to the intended PyPI maintainer account.
2. Open the account-level **Publishing** page.
3. Add a pending GitHub publisher with:
   - PyPI project name: `miniverl`
   - GitHub owner: `DaoyuanLi2816`
   - Repository: `mini-verl`
   - Workflow filename: `release.yml`
   - Environment: `pypi`
4. Confirm the GitHub environment is still named exactly `pypi`.
5. Only with explicit publication authorization, push `v0.2.0`.
6. Verify that the first tagged OIDC publication created the project and
   converted the pending publisher to a normal publisher, following the
   [OIDC publishing flow](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

The pending publisher does **not** reserve `miniverl` before that first upload.
Re-check name availability immediately before registration and tagging.
Package metadata `[project].name` must remain exactly `miniverl`, and every
owner/repository/workflow/environment claim above must match exactly. Until the
pending publisher is verifiably registered, do not create the tag.

### Name availability

`miniverl` was available on PyPI when this release was prepared
(`https://pypi.org/pypi/miniverl/json` returned HTTP 404 on 2026-07-28).
That 404 is not a reservation. **Re-check immediately before publishing** and
stop rather than silently changing the distribution name if it has been taken.

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
