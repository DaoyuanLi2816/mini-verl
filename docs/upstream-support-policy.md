# Upstream verl support policy

miniVERL supports closed compatibility profiles, not “latest verl” and not
arbitrary upstream configuration. The machine-readable matrix is
[`generated/upstream-support-matrix.json`](generated/upstream-support-matrix.json).

## Lifecycle

- **active**: documented local execution, conformance and current release
  qualification are maintained.
- **maintenance**: identity and readers remain stable; security and clear
  compatibility defects may be fixed, but no new source-field coverage is
  implied.
- **legacy**: retained for artifact inspection or migration only; no runnable
  support claim.
- **unsupported**: no profile identity exists and generic YAML must fail closed.

Every profile pins an upstream release and full commit. Once published, its
identity, field meanings, objective and upstream pin never move in place.
Promotion of another upstream version requires a separately named profile,
complete field matrix, upstream conformance, native runtime effects,
export/materialization checks, and exact-commit RTX 4080 qualification. A
template or partially compiling YAML is not support.

The active direct-GKD and sampled-k1 profiles both target official verl
`v0.8.0` at `7aed6b230776f963fa09509c10d9c3a767d1102c`. No v0.9-or-newer profile is
currently supported. Existing bridge artifacts remain independent-project
handoffs and do not imply upstream endorsement.

