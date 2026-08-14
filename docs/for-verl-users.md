# For verl users

miniVERL is a local runtime for one documented subset of verl v0.8 OPD. It
keeps familiar field names and Parquet data, then compiles distributed resource
intent into sequential phases on one CUDA GPU. It is an independent project;
the mapping is explicit and does not imply endorsement or full compatibility.

Inspect the closed, versioned registry with `miniverl profiles list`; use
`miniverl compat explain` before assuming that an accepted field is effective.
The [profile registry guide](profiles/index.md) documents the identity carried
by plans, caches, checkpoints and exports.

<picture>
  <source media="(max-width: 640px)" srcset="../verl-local-runtime-mobile.svg">
  <img src="../verl-local-runtime.svg" alt="verl-shaped YAML, overrides and Parquet prompts pass through a typed compiler; one CUDA GPU runs actor rollout, teacher scoring and actor update; portable artifacts can be handed to pinned verl while distributed execution remains outside miniVERL.">
</picture>

## What you can reuse

- A resolved YAML using the `verl-opd-v0.8-single-gpu-v1` field subset.
- Reward-free verl-style Parquet prompts with structured chat messages.
- One actor, one teacher, `n=1`, forward top-k GKD and token-mean aggregation.
- Immutable Hugging Face revisions, PEFT adapters and tokenizer snapshots.
- Familiar fields such as `actor_rollout_ref.model.path`,
  `distillation.teacher_models.teacher_model.model_path`, response bounds,
  learning rate and LoRA configuration.

What is not reusable: arbitrary Hydra composition inside miniVERL, shell launch scripts,
resource pools, Ray actors, FSDP/Megatron checkpoints, PPO/GRPO, critics,
policy-gradient OPD, task-reward mixtures, multiple teachers and multimodal
workers. Unsupported semantics fail closed instead of falling back silently.

## Command mapping

| verl action | miniVERL action |
| --- | --- |
| compose or capture a resolved config | provide resolved YAML to `--config` |
| add a Hydra-style override | repeat `--set`, use `--overrides-file`, or place tokens after `--` |
| inspect resolved intent | `miniverl plan --json` |
| launch OPD | `miniverl run` |
| read prompt Parquet | use the file directly |
| allocate resource pools | compile to sequential local phases |
| save a distributed checkpoint | unsupported |
| hand artifacts back | `miniverl export-verl` |

```bash
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 \
  --config verl-opd.yaml \
  --set 'data.train_files=["data/train.parquet"]' \
  --set actor_rollout_ref.actor.optim.lr=1e-5
```

Planning is weight-free and offline. Use `--json` to retain the complete field
matrix. The stable v0.9 CLI records repeated `--set`, override files and
trailing tokens with deterministic precedence; it does not execute `${...}`
interpolations or shell text. See [Config overrides](config-overrides.md).

For a serious run, add `--accept-local-reinterpretations --out plan.json`,
inspect the immutable artifact, then execute `miniverl run --plan plan.json`.
This binds the source YAML, overrides and scanned Parquet bytes to the exact
native config; see [Immutable execution plans](immutable-plans.md).

## Same fields, different placement

An upstream-shaped fragment can stay recognizable:

```yaml
actor_rollout_ref:
  model: {path: Qwen/Qwen3-0.6B}
  rollout: {name: vllm, n: 1}
distillation:
  teacher_models:
    teacher_model:
      model_path: Qwen/Qwen3-1.7B
      inference: {name: vllm}
```

miniVERL does not launch vLLM here. The compiler classifies both engine names
as local reinterpretations and runs local Hugging Face generation and teacher
scoring sequentially. The source value, local meaning and risk are preserved in
the compatibility report.

## Data mapping

`data.train_files`, `data.val_files`, `data.prompt_key`, prompt and response
bounds, shuffle and seed feed the native Parquet source directly. Each prompt
row preserves its structured messages and source metadata. Run
`miniverl data sample --out prompts.parquet` for a valid small file, or use
`miniverl convert-dataset` when crossing the native trajectory boundary.

The compiler never substitutes a calculator environment for missing Parquet
data. A missing file is an actionable error when execution begins. Dataset
bytes and schema are copied into export provenance.

## Actor, teacher and reference roles

The actor is the trainable PEFT policy. The teacher produces top-k token IDs
and log-probabilities on actor-generated tokens; top-k mass and overlap are
diagnostics, not an explicit tail bucket. Pure GKD in this profile has no reference
policy, critic, task reward or policy-gradient term. Local runtime strategies
may keep quantized roles resident, swap movable unquantized roles, or share a
compatible backbone, but role identities never collapse in provenance.

## Export boundary

```bash
miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout
miniverl bridge materialize scaleout --download --offline
miniverl bridge doctor scaleout --json
```

The stable v0.9 bundle carries PEFT, Parquet, config and provenance artifacts,
but is reported as `launchable: false` until exact base snapshots are materialized and
the pinned upstream checks pass. Read the [materialization workflow](scaleout-materialization.md).
Upstream parse/load smoke, artifact completeness, launchability and distributed
execution are separate statuses. miniVERL never reports a distributed job as
tested.

## Common errors

- **Unknown field:** capture a resolved config and remove fields outside the
  documented profile; inspect-only compilation can still explain known
  unsupported values.
- **Algorithm field rejected:** PG OPD, rewards, KL penalties, `n>1`, multiple
  teachers and distributed counts are intentionally unsupported.
- **Interpolation rejected:** resolve Hydra/OmegaConf in your trusted verl
  environment first. miniVERL will not execute `${...}`.
- **High-risk reinterpretation not accepted:** inspect `miniverl plan`, then
  pass `--accept-local-reinterpretations` for an external config. Packaged
  profiles carry reviewed acceptance metadata.
- **Prompt schema mismatch:** validate the Parquet `prompt` column as structured
  role/content messages, or convert it explicitly.
- **Tokenizer mismatch:** actor and teacher scoring require structural identity
  for the shared token space; a legacy behavioral fingerprint is not proof.
- **CUDA out of memory:** reduce context, response length or physical batches;
  keep logical update semantics unchanged. See [single-GPU planning](single-gpu-guide.md).
- **Bundle not launchable:** materialize exact snapshots and install the pinned
  verl commit; inspect the reported blocker. Read [scale-out materialization](scaleout-materialization.md).

Next: follow the [OPD quickstart](opd-quickstart.md), inspect the
[compatibility policy](compatibility.md), or review [all limitations](limitations.md).
