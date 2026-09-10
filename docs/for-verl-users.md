# For verl users

miniVERL turns supported experiment semantics from a resolved verl-shaped
config into a validated one-GPU plan. Field names, datasets, logical batches,
algorithms and artifacts remain recognizable; cluster placement is replaced by
sequential local phases and recorded as a lowering decision.

<picture>
  <source media="(max-width: 640px)" srcset="../verl-local-runtime-mobile.svg">
  <img src="../verl-local-runtime.svg" alt="A resolved verl config compiles into a one-GPU plan; actor, critic, reference, teacher and reward roles run in phases and publish portable artifacts with a readiness report.">
</picture>

## Start with the installed PPO/GRPO workflow

```bash
miniverl data sample --reward-profile target-length --rows 8 --out data/rl-prompts.parquet
miniverl run --example grpo --bind reward.provider=target_length --dry-run
miniverl run --example grpo --bind reward.provider=target_length --run-id local-grpo
```

This profile pins official verl `v0.9.0` at
`483b8a009ba3a97563edee3a19887e4862b8094a`. It accepts a resolved documented
input for PPO/GAE, GRPO, Dr.GRPO, RLOO or REINFORCE++, automatically selecting v4
and retaining the native IR as an inspection artifact. Replace `--example grpo`
with your own resolved YAML. Scientific-notation strings such as `1e-5` are
accepted when finite; `${...}`, NaN, infinity and unknown fields fail before a
runnable recipe is published.

Follow the [direct-config workflow](direct-verl-config.md) to run, inspect,
resume and export both examples. The [real upstream corpus](verl-compatibility-corpus.md)
records complete configuration outcomes; the [RL reference](verl-rl-runtime.md)
explains the algorithms and local execution model.

## Command mapping

| verl action | miniVERL action |
| --- | --- |
| capture a resolved Hydra config | provide it to `miniverl run resolved-verl.yaml` |
| inspect field semantics | `miniverl run resolved-verl.yaml --dry-run --json` |
| inspect the resolved local plan | read `verl-direct-report.json` and `config.resolved.yaml` |
| run actor/reward/reference phases | `miniverl run resolved-verl.yaml --bind ...` |
| read prompt Parquet | retain `data.train_files` and `data.prompt_key` directly |
| lower resource pools | record one process/device plus original source intent |
| inspect training and versions | `miniverl inspect runs/<id> --json` |
| recover an interrupted experiment | `miniverl run resolved-verl.yaml --resume runs/<id>` with the original bindings |
| export actor/critic and data | `miniverl export-verl --run runs/<id> --target-verl v0.9.0 --out handoff` |

## What maps into the RL runtime

- `data.train_files`, `val_files`, `prompt_key`, prompt/response limits,
  shuffle and seed drive the local Parquet source.
- `data.train_batch_size × rollout.n` is the logical trajectory count per
  rollout iteration.
- v4 retains v3's prompt-based `actor.ppo_mini_batch_size`, multiplying by
  `rollout.n`. Physical microbatch one and temporal offload are automatic.
- Actor model, revision, LoRA, optimizer, sampling, clipping and schedule
  fields feed the native recipe with their source units recorded.
- `trainer.total_training_steps` caps rollout iterations; `total_epochs` derives
  the schedule from the retained, filtered dataset and drop-last batches.
- Reward code uses an explicit local file/function binding plus checksum approval.
  Built-in exercise rewards and the upstream classifier-RM path are also available.
- Reference KL uses an independently frozen initial base. This role supplies
  the KL baseline, not a distillation teacher.

Distributed resource counts can be larger than one in the source config. They
are classified `distributed_only`, preserved in the report, and lowered to one
local process/device. Unknown algorithm or objective fields remain rejected.

## Existing v0.8 OPD profiles

The OPD compiler targets official verl `v0.8.0` at
`7aed6b230776f963fa09509c10d9c3a767d1102c`. It supports direct forward-top-k
GKD and sampled-k1 policy-gradient distillation, including measured profiles
and grouped/rewarded conformance variants.

```bash
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 \
  --config verl-opd.yaml \
  --set 'data.train_files=["data/train.parquet"]' \
  --set actor_rollout_ref.actor.optim.lr=1e-5 \
  --accept-local-reinterpretations --out plan.json
miniverl run --profile verl-opd-v0.8-single-gpu-v1 --plan plan.json --dry-run
```

`plan.json` binds the source YAML, ordered overrides, scanned Parquet bytes,
profile identity and native config. Use [immutable plans](immutable-plans.md)
and the [OPD quickstart](opd-quickstart.md) for that workflow.

## Artifact handoff

Completed RL runs first export a standard adapter, then a pinned handoff bundle:

```bash
miniverl export-adapter --run runs/my-rl --out runs/my-rl/model
miniverl export-verl --run runs/my-rl --target-verl v0.9.0 --out scaleout
miniverl bridge doctor scaleout --json
```

The report separates artifact completeness, upstream config parse, model/data
load smoke, reward implementation, launchability, distributed execution and
algorithm parity. Read the [RL workflow](local-rl-workflow.md) for the remaining
cluster steps, or the [OPD scale-out contract](verl-opd-scaleout.md) for the
separately supported v0.8 distillation path.

## Practical diagnostics

- Use `miniverl doctor` to inspect the CUDA stack.
- Resolve Hydra interpolation in the trusted upstream environment before
  importing.
- Keep immutable model and tokenizer revisions in portable configs.
- Reduce context or physical trajectory batch size on OOM; logical group and
  objective semantics remain unchanged.
- Read [compatibility](compatibility.md) for field status and
  [limitations](limitations.md) for the consolidated boundaries.
