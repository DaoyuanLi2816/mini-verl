# miniVERL

Run a documented subset of verl v0.8 on-policy distillation on one consumer
GPU. Bring a typed OPD profile and Parquet prompts, inspect rollout → teacher
scoring → update locally, then export standard artifacts. miniVERL is
independent; distributed execution and full verl compatibility are not claimed.

[Install and run locally](single-gpu-guide.md){ .md-button .md-button--primary }
[Read the compatibility boundary](verl-opd-runtime.md){ .md-button }

## Pip-only OPD quickstart

```bash
python -m pip install "miniverl[train,cuda]"
miniverl data sample --format verl-parquet --out prompts.parquet
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 \
  --config builtin:qwen3-0.6b-1.7b-opd \
  --set 'data.train_files=["prompts.parquet"]'
miniverl run --profile verl-opd-v0.8-single-gpu-v1 \
  --config builtin:qwen3-0.6b-1.7b-opd \
  --set 'data.train_files=["prompts.parquet"]' --dry-run
```

Planning loads no weights. Remove `--dry-run` on one CUDA GPU to execute the
pinned Qwen3 recipe and export a standard PEFT adapter. Install the matching
CUDA-enabled PyTorch wheel first; the `[cuda]` extra does not select one.

## Runtime and compatibility boundary

miniVERL runs one local CPU process or one NVIDIA CUDA GPU. Fit depends on the
model pair, context, kernels and VRAM; there is no GPU-name allowlist. Ray,
FSDP, Megatron, PPO, GRPO and distributed launch are outside the runtime.

The executable profile pins verl `v0.8.0` at `7aed6b23`: one actor, one teacher,
`n=1`, GKD `forward_kl_topk`, token-mean and no reward/KL penalty. Unsupported
algorithm or distributed semantics fail closed. Exports remain unlaunchable
until exact base snapshots are materialized and checked under the pinned verl
commit. See [scale-out materialization](scaleout-materialization.md).

## Measured runtime evidence

The v0.9 RTX 4080 developer workload consumed 32 distinct prompts and completed
8 current-policy updates at 3.1914 GiB peak reserved VRAM. Median steady-state
rollout, teacher-scoring and update times were 9.7200, 0.4864 and 2.3260 seconds;
a matched interruption/resume reproduced byte-identical trajectories, adapter
and optimizer tensors. No quality endpoint or method comparison ran.

[Measured workload evidence](verl-opd-reference-workload.md){ .md-button }

## Choose a path

<div class="path-grid" markdown>

<div class="path-card" markdown>

## Run OPD locally

Plan, execute, inspect and resume one local pure-OPD run.

```bash
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml
```

**Artifact:** a checksummed local execution plan, trajectories and PEFT adapter.

**Next:** [Plan and run](opd-quickstart.md)

</div>

<div class="path-card" markdown>

## Bring a verl config

Import the resolved, documented OPD subset with field-by-field classifications.

```bash
miniverl import-verl --profile verl-opd-v0.8-single-gpu-v1 \
  --config verl-opd.yaml --out local-opd.yaml
```

**Artifact:** a round-trippable profile and `local-opd.import-report.json`.

**Next:** [Supported field boundary](compatibility.md)

</div>

<div class="path-card" markdown>

## Move data and artifacts

Convert Parquet, export standard artifacts and inspect the unsupported boundary.

```bash
miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout
```

**Artifact:** PEFT + Parquet + OPD overrides and separate readiness flags.

**Next:** [current scale-out contract](verl-opd-scaleout.md)

</div>

</div>

## Research Notes

The v0.7 external study stopped at its first preregistered gate: **0 selected
checkpoints, 0 qualified teachers, 0 continuation arms and 0 final-test tasks
accessed**. All eight candidates scored 0/64 retained JSONNav utility against
the unchanged 20% floor.

```bash
miniverl pilot --builtin-study alignment-external-v1 --json
```

[Read the early-stop study](alignment-external/alignment-external-v1.md){ .md-button .md-button--primary }

Alignment Lab starts from a saturated SFT checkpoint. No continuation method
improves it; measured regressions and unexecuted external safety endpoints stay
visible. RecoveryBench and the calculator study likewise preserve their
negative and mixed results rather than turning them into product claims.

- [Alignment Lab](alignment-lab/alignment-lab-v1.md)
- [RecoveryBench](recoverybench/recoverybench-v1.md)
- [Calculator protocol study](benchmarking.md)
- [External Alignment Gate](alignment-external/alignment-external-v1.md)
