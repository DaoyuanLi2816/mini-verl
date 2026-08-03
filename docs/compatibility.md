# Compatibility policy

miniVERL is a `0.x` research package: a minor release may add validated fields
or tighten unsafe input, but it must not silently reinterpret a supported
artifact. Release notes identify migrations, and the current reader either
loads an older artifact with its historical semantics or rejects it with a
regeneration hint.

## Stable surfaces through v0.6

- Package name `miniverl`, command name `miniverl`, documented command names,
  and the small Python entry points `RunConfig` and `OPDTrainer`.
- Existing run artifact names. The v0.2.4 config files are additive;
  `config.original.yaml` remains the checkpoint-resume compatibility layer.
- Trajectory schema 1, cache schema 2 plus the explicitly constrained legacy
  schema-1 reader, checkpoint manifest schema 1, and benchmark schemas 1/2.
- Historical protocol-v1 and verifier-v1 semantics. New runs default to the
  stricter v2 semantics rather than rewriting old evidence.
- The v0.6 verl bridge is separately versioned as profile
  `single-gpu-online-distillation-v1`. It targets official verl `v0.8.0` at
  commit `7aed6b230776f963fa09509c10d9c3a767d1102c`; every bundle carries that
  exact contract in `REQUIRED_VERL.txt`.

## Pinned bridge, not generic compatibility

Compatibility Level 1 covers standard Hugging Face, PEFT, safetensors,
tokenizer and Parquet artifacts. Level 2 is a fail-closed 14-field config
whitelist. Level 3 adds the generated bundle, reward scaffold, hashes and an
exact-source smoke. Unknown, algorithm-changing or distributed-only verl
fields are rejected instead of guessed.

These levels do not make miniVERL a verl runtime. Optimizer state, distributed
RNG, FSDP/Megatron native checkpoints, Ray runtime state and teacher-cache to
PPO-reference-cache conversion are unsupported. The release smoke validates
artifacts and configuration; distributed execution is recorded as not tested.
See the [bridge contract](verl-bridge.md).

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
