# Compatibility policy

miniVERL is a `0.x` research package: a minor release may add validated fields
or tighten unsafe input, but it must not silently reinterpret a supported
artifact. Release notes identify migrations, and the current reader either
loads an older artifact with its historical semantics or rejects it with a
regeneration hint.

## Stable surfaces within the v0.2 line

- Package name `miniverl`, command name `miniverl`, documented command names,
  and the small Python entry points `RunConfig` and `OPDTrainer`.
- Existing run artifact names. The v0.2.4 config files are additive;
  `config.original.yaml` remains the checkpoint-resume compatibility layer.
- Trajectory schema 1, cache schema 2 plus the explicitly constrained legacy
  schema-1 reader, checkpoint manifest schema 1, and benchmark schemas 1/2.
- Historical protocol-v1 and verifier-v1 semantics. New runs default to the
  stricter v2 semantics rather than rewriting old evidence.

## Versioned but extensible

Run manifests are descriptive records rather than executable inputs. New
v0.2.4 runs use manifest schema 3 for explicit config-layer provenance; report,
inspect and export continue to read older v0.2 manifests by feature detection.
Unknown fields are preserved where a schema allows them, while an unknown
schema version in a training-critical artifact is rejected.

Deprecations emit a warning in at least one minor release before removal and
name the replacement. Security fixes may reject input that an older permissive
parser accepted. Private modules and undocumented CLI output formatting are not
compatibility promises.

When exact resumption matters, use the package version recorded in the run and
keep the full run directory. See [reproducibility](reproducibility.md) for the
submitted/validated/runtime config layers.
