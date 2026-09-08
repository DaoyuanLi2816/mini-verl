# For verl users

miniVERL turns supported experiment semantics from a resolved verl-shaped
config into a validated one-GPU plan. Field names, datasets, logical batches,
algorithms and artifacts remain recognizable; cluster placement is replaced by
sequential local phases and recorded as a lowering decision.

<picture>
  <source media="(max-width: 640px)" srcset="../verl-local-runtime-mobile.svg">
  <img src="../verl-local-runtime.svg" alt="A resolved verl config compiles into a one-GPU plan; actor, critic, reference, teacher and reward roles run in phases and publish portable artifacts with a readiness report.">
</picture>

## Start with the v0.9 RL profile

```bash
miniverl data sample --task-rewards --rows 8 --out data/rl-prompts.parquet
miniverl import-verl --profile verl-rl-v0.9-single-gpu-v1 \
  --config examples/verl-rl-v0.9-single-gpu.yaml --out local-grpo.yaml
miniverl validate local-grpo.yaml --json
miniverl train local-grpo.yaml --dry-run
```

This profile pins official verl `v0.9.0` at
`483b8a009ba3a97563edee3a19887e4862b8094a`. It accepts a resolved documented
subset for GRPO, Dr.GRPO, RLOO or REINFORCE++, then writes both a native recipe
and `*.import-report.json`. Scientific-notation strings such as `1e-5` are
accepted when finite; `${...}`, NaN, infinity and unknown fields fail before a
runnable recipe is published.

The [single-GPU RL guide](verl-rl-runtime.md) documents every algorithm and
lowering. The committed [example report](generated/verl-rl-v0.9-compatibility.json)
shows the exact machine-readable output.

## Command mapping

| verl action | miniVERL action |
| --- | --- |
| capture a resolved Hydra config | provide it to `import-verl --config` |
| inspect field semantics | read `*.import-report.json` |
| validate the local plan | `miniverl validate local.yaml --json` |
| run actor/reward/reference phases | `miniverl train local.yaml` |
| read prompt Parquet | retain `data.train_files` and `data.prompt_key` directly |
| lower resource pools | record one process/device plus original source intent |
| inspect trajectories | `miniverl inspect runs/<id>/trajectories.jsonl` |

## What maps into the RL runtime

- `data.train_files`, `val_files`, `prompt_key`, prompt/response limits,
  shuffle and seed drive the local Parquet source.
- `data.train_batch_size × rollout.n` is the logical trajectory count per
  rollout iteration.
- `actor.ppo_mini_batch_size` controls how that logical batch is divided into
  actor updates; physical trajectory batching remains a separate `miniverl`
  execution control.
- Actor model, revision, LoRA, optimizer, sampling, clipping and schedule
  fields feed the native recipe with their source units recorded.
- `trainer.total_training_steps` is the rollout-iteration cap. Epoch-only
  scheduling is not guessed because it depends on dataset traversal semantics.
- Task rewards are explicit. The portable compiler accepts the built-in
  exact-answer and target-length providers; trusted Python and environment
  providers are injected through the local API.
- Fixed reference KL requires an explicit frozen reference adapter. The
  compiler never creates an unqualified same-base reference policy.

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

Completed OPD runs can produce a pinned scale-out bundle:

```bash
miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout
miniverl bridge materialize scaleout --download --offline
miniverl bridge doctor scaleout --json
```

The report separates artifact completeness, upstream config parse, model/data
load smoke, reward implementation, launchability, distributed execution and
algorithm parity. Read the [scale-out contract](verl-opd-scaleout.md).

## Practical diagnostics

- Use `miniverl doctor` to inspect the CUDA stack.
- Resolve Hydra interpolation in the trusted upstream environment before
  importing.
- Keep immutable model and tokenizer revisions in portable configs.
- Reduce context or physical trajectory batch size on OOM; logical group and
  objective semantics remain unchanged.
- Read [compatibility](compatibility.md) for field status and
  [limitations](limitations.md) for the consolidated boundaries.
