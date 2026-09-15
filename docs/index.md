# miniVERL

**verl for a single consumer GPU.**

Train with PPO, GRPO or on-policy distillation on one NVIDIA GPU.
Bring your verl configs, follow the training, resume from a checkpoint
and take the resulting models into your next workflow.

[Start training](for-verl-users.md){ .md-button .md-button--primary }
[Choose a workflow](comparisons.md){ .md-button }

## Start with PPO

Install the matching CUDA-enabled build from the
[PyTorch installer](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install "miniverl[train,hydra]"
miniverl data sample --reward-profile target-length --rows 8 --out data/rl-prompts.parquet
miniverl run --example hydra-ppo --bind reward.provider=target_length --dry-run
miniverl run --example hydra-ppo --bind reward.provider=target_length --run-id local-ppo
```

This Qwen3-0.6B example uses a simple length reward you can inspect in
`rewards.jsonl`. The dry run checks the configuration before downloading models.
Continue with [inspect → resume → export](for-verl-users.md).

## Three ways to use miniVERL

<div class="path-grid" markdown>

<div class="path-card" markdown>

### Train with rewards

Run PPO or GRPO using your verl config and local data.

```bash
miniverl run --example hydra-ppo \
  --bind reward.provider=target_length --dry-run
```

**Artifact:** training recipe and configuration report.

**Next:** [Bring your own config](direct-verl-config.md)

</div>

<div class="path-card" markdown>

### Distill locally

Train a student on its own rollouts with feedback from a teacher.

```bash
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 \
  --config builtin:qwen3-0.6b-1.7b-opd
```

**Artifact:** a memory and execution plan for the student/teacher pair.

**Next:** [Run your first OPD update](opd-quickstart.md)

</div>

<div class="path-card" markdown>

### Prepare a larger run

Export your model, data and config for handoff to verl.

```bash
miniverl export-verl --run runs/my-run \
  --target-verl v0.9.0 --out scaleout
```

**Artifact:** PEFT, Parquet and config files with a setup checklist.

**Next:** [Review the bundle](verl-opd-scaleout.md)

</div>

</div>

## Make the most of your GPU

miniVERL schedules training roles in phases so actor, critic, reference,
reward and teacher can share one GPU. Choose a model and placement strategy
with the [single-GPU guide](single-gpu-guide.md) and
[hardware planner](hardware-planning.md).
Native recipes also offer SFT, DPO and offline KD.

## Explore the results

- [OPD on an RTX 4080](verl-opd-reference-workload.md): update timing, memory and exact recovery.
- [Rollout Runtime v2](benchmarks/rollout-runtime-v2.md): 24 measurements across rollout settings.
- [Alignment Lab](alignment-lab/alignment-lab-v1.md): a saturated tool-policy case study.
- [RecoveryBench](recoverybench/recoverybench-v1.md): tool recovery outcomes.
- [External Alignment Gate](alignment-external/alignment-external-v1.md): frozen external evaluation results.

## About and support

miniVERL is an independent project for single-GPU NVIDIA CUDA training.
The [compatibility matrix](compatibility.md) lists supported verl versions,
profiles and handoff stages. [Limitations](limitations.md) collects hardware,
scientific and security scope. See [comparisons](comparisons.md) for help choosing
a training workflow.
