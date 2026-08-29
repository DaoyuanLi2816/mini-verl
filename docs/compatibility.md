# Compatibility policy

miniVERL versions its executable profiles and artifact schemas independently.
A minor release may add validated fields or tighten input validation while
preserving the historical meaning of existing artifacts. Release notes identify
migrations, and readers retain the schema-specific behavior needed to load them.

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

## Pinned OPD profiles

v0.8 introduced `verl-opd-v0.8-single-gpu-v1`: one local actor, one teacher,
pure GKD `forward_kl_topk`, token-mean aggregation, LoRA/QLoRA and Parquet
prompts. The registry now also contains sampled-k1, grouped-independent and a
separate rewarded sampled-k1 contract.
`import-verl --config` publishes a canonical profile plus field report;
`export-verl` recognizes a completed compatible run and emits its profile-bound
OPD overrides. The rewarded profile accepts only a closed deterministic
exact-answer provider and remains critic/value/GRPO-free. Multiple teachers,
arbitrary reward code, multimodal and all distributed semantics fail closed.
Local physical phase scheduling is always
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

These miniVERL-defined levels describe artifact and configuration exchange.
Optimizer state, distributed RNG, FSDP/Megatron native checkpoints, Ray runtime
state and teacher-cache conversion sit beyond that exchange contract. The
release smoke records artifact/config validation and distributed execution as
separate evidence fields. Legacy reward-scaffold bundles use
`launch.template.sh`; current profile exports remain explicit about their
individual launch blockers and semantic-parity status.
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
