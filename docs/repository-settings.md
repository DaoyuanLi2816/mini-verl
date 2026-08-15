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
  execution.

Confirm branch rules, tag rules, private vulnerability reporting and runner
registration in GitHub settings before each major release. Repository files
cannot prove those external controls are enabled.

