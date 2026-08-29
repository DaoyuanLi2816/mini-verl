# v0.10.1 focused security review

This review covers input and artifact boundaries touched by the current local
runtime and release qualification. Model-code and third-party-library isolation
remain in the consolidated [execution limitations](limitations.md#execution-boundary).

| surface | enforced boundary | regression evidence |
| --- | --- | --- |
| YAML and overrides | `safe_load`, typed fields, no unresolved interpolation, no shell evaluation | config, interpolation and hostile-tag tests |
| Parquet and nested metadata | typed prompt rows; bridge inspection has row/byte bounds | dataset streaming, semantics and metadata privacy tests |
| bundle trees | regular files under one root; symlink, junction/reparse, traversal, count and nominal-size rejection | hostile-bundle and preflight tests |
| qualification ZIP | exact GitHub run/SHA, bounded entries and expanded bytes, no traversal or symlink, declared hashes | qualification extraction and binding tests |
| reward scaffold | fail-closed AST inspection; dynamic import requires the explicit trusted path | reward static and definition-time tests |
| checkpoints/caches | finite JSON plus safetensors, checksums and atomic publication; no pickle | cache, checkpoint, resume and transaction tests |
| JSON | non-finite numbers rejected; strict parsers bound depth, members and size where model text is accepted | protocol, strict-JSON and qualification tests |
| output publication | input/output conflict checks, exclusive run locks and transactional directory replacement | bridge publish, run-lock and lifecycle tests |
| privacy | portable projections remove home/user/environment/credential text; qualification JSON rejects private paths | privacy completeness, hardware and qualification tests |

Repository code uses argv-based subprocess calls and does not add a
`shell=True` execution path. The pinned conformance tests execute checked-out
upstream source only inside their explicit test environment. Model repositories
still become executable when a user opts into `trust_remote_code: true`; the
default remains false.

The review found one public maintenance defect: `SECURITY.md` still named the
obsolete 0.2.4 line. The canonical release-state checker now owns that current
stable claim. The new external qualification download also introduced a new
archive boundary, so extraction was implemented fail-closed with dedicated
tests rather than relying on `extractall`.
