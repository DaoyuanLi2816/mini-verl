# verl bridge: portable artifacts, bounded semantics

miniVERL is an independent project; no endorsement by the verl project is
implied. The bridge targets the documented
`single-gpu-online-distillation-v1` profile subset of
[`verl v0.8.0`](https://github.com/verl-project/verl/tree/v0.8.0), pinned to
commit `7aed6b230776f963fa09509c10d9c3a767d1102c` (`7aed6b23`). It is
**miniVERL-defined compatibility Level 3**, not full verl compatibility.

<picture class="bridge-architecture">
  <source media="(max-width: 600px)" srcset="../verl-bridge-architecture-mobile.svg">
  <img src="../verl-bridge-architecture.svg" alt="Three verified bridge layers—miniVERL local runtime, a portable artifact bundle, and a pinned upstream parse/load smoke—followed by a dashed arrow to distributed execution marked NOT TESTED.">
</picture>

The solid arrows cover local artifact production, the portable bundle, and the
pinned parse/load smoke. The dashed arrow is deliberate: no Ray, FSDP, vLLM or
distributed verl job ran, and no miniVERL-OPD-to-verl-PPO semantic parity is
claimed.

## Compatibility state

| State | Current value | Meaning |
| --- | --- | --- |
| `artifact_bundle_complete` | `true` | PEFT, safetensors, Parquet, config and provenance are present and hashed. |
| `upstream_config_parse_passed` | `false` in a new bundle | Set only by a separate pinned upstream smoke record, never inferred at export time. |
| `model_data_load_smoke_passed` | `false` in a new bundle | The export itself does not load the base snapshot or execute a model. |
| `reward_implementation_complete` | `false` | The generated reward function deliberately fails closed. |
| `launchable` | `false` | Base weights, reward logic and confirmed mappings are incomplete. |
| `distributed_execution_tested` | `false` | No distributed job ran. |
| `algorithm_semantic_parity` | `false` | The target is a PPO/reward scaffold, not a continuation of miniVERL OPD. |

The committed [pinned smoke record](generated/verl-bridge-smoke.json) verifies a
specific artifact-only upstream parse/load exercise. It remains separate from
the readiness state of a newly exported bundle and from any execution claim.

## Import a resolved profile subset

`import-verl` accepts the documented, resolved field subset—not arbitrary
Hydra/OmegaConf or verl YAML. With only a source profile, it writes
`import-report.json` and a non-executable `imported.template.yaml`:

```bash
miniverl import-verl resolved-verl.yaml \
  --profile single-gpu-online-distillation-v1 \
  --target-verl v0.8.0 \
  --out recipes/imported.yaml
```

The status is `needs_user_input` until the source or command determines the
training environment, qualified teacher, objective and schedule interpretation.
Parquet paths never silently select the calculator environment, and a same-base
standard teacher without a distinct model or adapter is never invented.

To deliberately produce a runnable recipe, supply the missing contract:

```bash
miniverl import-verl resolved-verl.yaml \
  --profile single-gpu-online-distillation-v1 \
  --target-verl v0.8.0 \
  --environment jsonnav \
  --teacher-model Qwen/Qwen3-1.7B \
  --loss-profile topk-tail-reverse-kl \
  --schedule-mapping epochs-as-cycles \
  --out recipes/imported.yaml
```

The explicit schedule option acknowledges that verl epochs/save/test frequency
units are not proven equivalent to miniVERL cycles. Every source field is
classified as `exact`, `derived`, `informational_only`,
`requires_user_confirmation` or `unsupported`. In particular:

| Source field | Classification | Treatment |
| --- | --- | --- |
| `data.train_files`, `data.val_files`, `data.prompt_key` | `informational_only` | Recorded in the report; never substituted for a `ToolEnvironment`. |
| `data.max_response_length` | `exact` | Copied to the per-turn response bound. |
| `data.max_prompt_length` | `derived` | Combined with response length for miniVERL's total trajectory bound. |
| optimizer learning rate and seed | `exact` | Copied after finite numeric validation. |
| `trainer.total_epochs`, `save_freq`, `test_freq` | `requires_user_confirmation` | Copied only after the explicit schedule mapping. |
| algorithm, distributed or unknown fields | `unsupported` | Rejected with a report. |

Finite scientific-notation strings such as `1e-5` are accepted. NaN, infinity
and unresolved `${...}` interpolations are rejected with an actionable error.
Every runnable output passes `RunConfig` validation before atomic publication.

## Export a portable bundle

```bash
miniverl export-verl --run runs/<run-id> \
  --target-verl v0.8.0 \
  --out exports/<bundle>

miniverl bridge doctor exports/<bundle> --require-verl
```

The bundle contains:

```text
model/       adapter_config.json, adapter_model.safetensors, tokenizer metadata,
             base-model.json (identity only; base snapshot is not bundled)
data/        train.parquet, val.parquet
recipe/      verl-overrides.yaml, launch.template.sh, REQUIRED_VERL.txt
reward/      reward_or_verifier_scaffold.py (fails closed)
provenance/  source manifest/result, compatibility-report.json, SHA256SUMS
README.md
```

Available source-run response length and learning rate are preserved in the
override file. The miniVERL total-token bound, cycle schedule and environment
identity are preserved in `source_run_values`; they are not relabelled as
equivalent verl intent. Any prompt limit or schedule value inserted for the PPO
scaffold appears in `placeholder_defaults` with `source_run_intent: false`.

`bridge doctor` verifies pins, standard adapter structure, tokenizer metadata,
Parquet schema, override structure, reward importability, privacy and hashes.
An `ok` verdict means the artifact checks passed; it still returns
`launchable: false` while the fail-closed reward scaffold remains. The template
script also refuses to proceed without the immutable base snapshot and a
completed reward implementation.

## Unsupported boundary

The bridge does not translate optimizer state, distributed RNG, FSDP or
Megatron checkpoints, Ray state, PPO advantage/clipping semantics, GRPO group
semantics, or a miniVERL teacher cache into PPO reference log-probabilities.
See [compatibility](compatibility.md), [launch requirements](verl-bridge-launch.md)
and the [demo recording script](verl-bridge-demo.md).
