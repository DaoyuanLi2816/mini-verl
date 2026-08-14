# Compatibility policy

miniVERL is a `0.x` research package: a minor release may add validated fields
or tighten unsafe input, but it must not silently reinterpret a supported
artifact. Release notes identify migrations, and the current reader either
loads an older artifact with its historical semantics or rejects it with a
regeneration hint.

## Stable surfaces through v0.9

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

v0.8 adds the separately versioned `verl-opd-v0.8-single-gpu-v1` executable
profile: one local actor, one teacher, one generation, pure GKD
`forward_kl_topk`, token-mean aggregation, LoRA/QLoRA and Parquet prompts.
`import-verl --config` publishes a canonical profile plus field report;
`export-verl` recognizes a completed compatible run and emits reward-free OPD
overrides. PG OPD, task-reward mixtures, multiple teachers, multimodal and all
distributed semantics fail closed. Local physical phase scheduling is always
labelled as a reinterpretation of upstream resource fields.

The [machine-readable field matrix](generated/verl-opd-v0.8-compatibility.json)
is regenerated directly from the typed compiler and its pinned resolved
fixture; CI compares the committed bytes with the generator output. The
[field-effect record](generated/verl-opd-v0.8-field-effects.json) independently
mutates every executable non-informational field and records its observed plan
or native-runtime effect.

The older `single-gpu-online-distillation-v1` import/export contract remains
available for migration and retains its historical non-launchable PPO/reward
scaffold semantics.

Compatibility Level 1 covers standard Hugging Face, PEFT, safetensors,
tokenizer and Parquet artifacts. Level 2 is a fail-closed 14-field config
whitelist. **miniVERL-defined compatibility Level 3** adds the generated
bundle, reward scaffold, hashes and an exact-source smoke. Unknown,
algorithm-changing or distributed-only verl fields are rejected instead of
guessed.

These miniVERL-defined levels do not make miniVERL a verl runtime. Optimizer state, distributed
RNG, FSDP/Megatron native checkpoints, Ray runtime state and teacher-cache to
PPO-reference-cache conversion are unsupported. The release smoke validates
artifacts and configuration; distributed execution is recorded as not tested.
Current bundles contain a fail-closed reward scaffold and an absent base
snapshot, so they use `launch.template.sh` and report `launchable: false`.
See the [current local runtime](verl-opd-runtime.md), [current scale-out
contract](verl-opd-scaleout.md), and [legacy bridge](legacy-verl-bridge.md).

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
