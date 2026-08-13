# Immutable execution plans

For a serious run, freeze the resolved config and Parquet inputs before loading
model weights:

```bash
miniverl plan --config verl-opd.yaml \
  --accept-local-reinterpretations \
  --out plan.json --offline
miniverl run --plan plan.json --offline
```

`plan --out` is still weight-free. It scans the actual Parquet source and
atomically writes a deterministic JSON artifact containing:

- the source YAML byte digest and every ordered override;
- the full compatibility report and explicit reinterpretation acceptance;
- immutable student and teacher revisions;
- every Parquet file hash plus schema, content and row-count identities;
- the exact validated native `RunConfig`, loss semantics and physical
  recommendations;
- declared tokenizer revisions. Structural tokenizer identities remain
  `declared_not_loaded` until the bounded hardware probe loads them.

The plan's canonical digest excludes only its own digest fields. `run --plan`
validates the schema and miniVERL minor-version boundary, recomputes that digest,
rechecks the YAML and every Parquet byte, rescans the data manifest, and consumes
the sealed native config. It never recompiles with new defaults. Model
construction starts only after all checks pass.

The same plan digest is recorded in the run manifest, teacher-cache index,
checkpoint state and checkpoint identity. A cache or checkpoint from another
plan is refused. Direct `run --config` remains available for quick experiments,
but `--plan` is mutually exclusive with config and override options.

## Deliberate limits

An immutable plan does not download or hash model snapshots, allocate CUDA, or
claim measured memory. Those identities become measured only through the
explicit probe/materialization stages. Moving model revisions, unsupported verl
semantics and unaccepted local reinterpretations prevent plan publication.
