# From verl to one GPU and back

Run an inspectable PPO or GRPO exercise with the installed package. Both use
Qwen3-0.6B, four samples per iteration and a deterministic length reward.
PPO additionally trains a separate critic, computes GAE and records value loss.

## Install and inspect compatibility

Choose the CUDA PyTorch build for your driver from the
[official installer](https://pytorch.org/get-started/locally/), then install
`miniverl[train]`. The examples use unquantized LoRA and require no repository
checkout. Run all commands from the same new working directory.

```bash
python -m pip install "miniverl[train]"
miniverl doctor
miniverl data sample --reward-profile target-length --rows 8 --out data/rl-prompts.parquet
miniverl import-verl --profile verl-rl-v0.9-single-gpu-v3 --example ppo --out local-ppo.yaml
miniverl validate local-ppo.yaml --json
miniverl train local-ppo.yaml --dry-run --json
```

Review `local-ppo.import-report.json` before downloading weights. It records
exact fields, conformant semantics, local lowerings and unresolved requirements.
`local-ppo.verl.yaml` is the original input; `local-ppo.yaml` is the validated
native recipe. `local-ppo.example-provenance.json` records the upstream file,
commit and every adaptation. In particular, this replaces the upstream
GSM8K/MATH task with a bounded educational length-reward exercise.

For your own experiment, replace `--example ppo` with `--config resolved.yaml`.
Use the [corpus report](verl-compatibility-corpus.md) to see which real upstream
examples need further adaptation. Each compiler profile has a fixed identity;
selecting a profile does not select a rollout backend.

## Run, inspect, replay and hand off

```bash
miniverl train local-ppo.yaml --run-id local-ppo
miniverl inspect runs/local-ppo
miniverl inspect runs/local-ppo --json
miniverl report runs/local-ppo
miniverl train local-ppo.yaml --resume-from runs/local-ppo/checkpoints/step-000002
miniverl export-adapter --run runs/local-ppo --out runs/local-ppo/model
miniverl export-verl --run runs/local-ppo --target-verl v0.9.0 --out ppo-handoff
miniverl bridge doctor ppo-handoff --json
```

The resume command deliberately replays the second iteration from the saved
first-iteration checkpoint. It restores actor, critic, both optimizers and RNG
without changing the learning-rate schedule. New checkpoints also bind the
training-log prefixes. Replayed tails are archived under `recovery-tails/`,
so current metrics count each committed update once. For an interrupted run,
`--resume runs/local-ppo` selects the highest valid checkpoint instead.

The planner predicts eight trajectories, four actor updates and four critic
updates. Inspection reports actual counts, rewards, advantages, returns,
policy/value losses, KL/entropy/clipping, memory, checkpoint and resume identity.
Missing GPU measurements are `null`, not zero. Corrupt records and checkpoint
disagreements produce errors rather than a plausible-looking summary.

## The matching GRPO workflow

Reuse the Parquet file above:

```bash
miniverl import-verl --profile verl-rl-v0.9-single-gpu-v3 --example grpo --out local-grpo.yaml
miniverl train local-grpo.yaml --dry-run --json
miniverl train local-grpo.yaml --run-id local-grpo
miniverl inspect runs/local-grpo --json
miniverl train local-grpo.yaml --resume-from runs/local-grpo/checkpoints/step-000001
miniverl export-adapter --run runs/local-grpo --out runs/local-grpo/model
miniverl export-verl --run runs/local-grpo --target-verl v0.9.0 --out grpo-handoff
miniverl bridge doctor grpo-handoff --json
```

GRPO has eight trajectories, two actor updates and no critic. Its advantages
compare the two sampled responses for each prompt. Equal rewards correctly
produce a zero-variance group; the inspection output makes that visible.

## Expected artifacts

```text
local-ppo.yaml / .verl.yaml / .import-report.json / .example-provenance.json
runs/local-ppo/
  manifest.json                  status, identities, final checkpoint
  config.{submitted,validated,resolved}.yaml
  trajectories.jsonl             tokens, masks, group/sample/policy identity
  rewards.jsonl / advantages.jsonl / metrics.jsonl / events.jsonl
  checkpoints/step-000002/        first complete PPO iteration
  checkpoints/final/             actor, critic, optimizers, RNG, log bindings
  recovery-tails/                records replaced by checkpoint replay
  report.html / summary.md
  model/                        exported standard PEFT adapter
ppo-handoff/
  model/ / critic/ / data/ / reward/ / recipe/ / provenance/
```

The handoff preserves the actor adapter, critic tensors, Parquet, source
semantics and hashes. The readiness report identifies remaining cluster work:
materialize the base model, integrate the critic/reward roles, choose cluster
placement and validate upstream execution. Local optimizer/RNG state resumes
in miniVERL; it is not an upstream sharded checkpoint.

## What the exercise measures

These commands demonstrate the training and recovery workflow, not improvement
on an external task benchmark. The length reward is an understandable example
objective, not a quality judge. Compare each run's actual metrics with its
exact-wheel [RTX 4080 release qualification](release-qualification.md). The
[v0.14.0 release assets](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.14.0)
publish the measured per-case rewards, losses, runtime and peak VRAM in
`qualification-evidence.tar.gz` → `v014/product-workflows.json` after the
exact-candidate gate passes. Other VRAM classes retain an
[evidence-scoped planning status](hardware-planning.md).
