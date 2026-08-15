# Repository settings audit

This is a maintainer checklist, not a claim about settings that code cannot
inspect.

- Default branch: `main`.
- Require a pull request and the current CPU/build/docs/pinned-verl checks.
- Do not require a second approval; this is a single-maintainer project.
- Restrict tag creation for release tags to the maintainer.
- Keep PyPI trusted publishing bound to `release.yml` and environment `pypi`.
- Enable private vulnerability reporting.
- Keep Actions permissions read-only by default and grant write scopes per job.
- Register the private RTX 4080 runner only after following the
  [runner safety guide](release-qualification.md); do not allow fork-triggered
  execution. Apply exactly `self-hosted`, `cuda` and `rtx4080`; keep the runner
  offline except during a maintainer-dispatched qualification.
- Keep candidate construction on `ubuntu-latest`; the private runner downloads
  the same-run artifact and must never rebuild release distributions.

Confirm branch rules, tag rules, private vulnerability reporting and runner
registration in GitHub settings before each major release. Repository files
cannot prove those external controls are enabled.
