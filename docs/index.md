# miniVERL

Run verl-style on-policy distillation on one NVIDIA GPU, inspect every local
mapping and teacher target, and carry standard artifacts into a scale-out
workflow.

[Install and run locally](single-gpu-guide.md){ .md-button .md-button--primary }
[Start with a verl profile](for-verl-users.md){ .md-button }

## From profile to adapter

miniVERL compiles typed YAML and structured Parquet prompts into three local
phases: actor rollout, teacher scoring and actor update. The same execution plan
binds the profile version, overrides and input bytes to trajectories,
checkpoints, teacher caches and the final PEFT adapter.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install "miniverl[train,cuda]"
miniverl data sample --format verl-parquet --out prompts.parquet
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 \
  --config builtin:qwen3-0.6b-1.7b-opd \
  --set 'data.train_files=["prompts.parquet"]' --out plan.json
miniverl run --profile verl-opd-v0.8-single-gpu-v1 \
  --plan plan.json --dry-run
```

Planning is weight-free. Review `plan.json`, remove `--dry-run` on a CUDA GPU,
then inspect and export the result:

```bash
miniverl run --profile verl-opd-v0.8-single-gpu-v1 --plan plan.json \
  --output runs --run-id my-opd
miniverl inspect runs/my-opd/trajectories.jsonl
miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout
miniverl bridge doctor scaleout --json
```

The `[train,cuda]` extra supplies the training and quantization stack. Choose
the matching CUDA PyTorch wheel separately; the
[single-GPU installation guide](single-gpu-guide.md) includes flexible and
maintainer-measured setups.

## Three ways to use miniVERL

<div class="path-grid" markdown>

<div class="path-card" markdown>

### Run local OPD

Compile a pinned direct-GKD or sampled-k1 profile, execute it in local phases,
and inspect strict current-policy trajectories.

```bash
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml
```

**Artifact:** immutable plan, trajectories, teacher cache and PEFT adapter.

**Next:** [OPD quickstart](opd-quickstart.md)

</div>

<div class="path-card" markdown>

### Bring verl-shaped inputs

Keep familiar field names and structured Parquet while receiving a
field-by-field account of each local effect.

```bash
miniverl compat check --profile verl-opd-v0.8-single-gpu-v1 \
  --config verl-opd.yaml
```

**Artifact:** resolved compatibility matrix and local execution plan.

**Next:** [For verl users](for-verl-users.md)

</div>

<div class="path-card" markdown>

### Prepare scale-out artifacts

Package the local adapter, source Parquet, resolved config and provenance, then
materialize exact model snapshots against the pinned upstream source.

```bash
miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout
```

**Artifact:** checksummed PEFT + Parquet + config bundle with readiness states.

**Next:** [Scale-out contract](verl-opd-scaleout.md)

</div>

</div>

## Runtime design

The local scheduler keeps actor, teacher and optional reference roles explicit
while choosing resident, swap or shared-backbone placement. Structured token
provenance, pickle-free caches and transactional publication make a run
inspectable and resumable across phase boundaries.

Two measured profiles pin official verl `v0.8.0` at `7aed6b23`:

| Profile | Objective | Teacher signal |
| --- | --- | --- |
| direct GKD | `forward_kl_topk` | top-k token IDs and log-probabilities |
| sampled-k1 PG | sampled `k1` + vanilla policy loss | sampled-token teacher log-probability |

[Compatibility profiles](profiles/index.md) describes the exact field and
semantic contract. [Hardware planning](hardware-planning.md) explains how the
same device-name-agnostic CUDA path adapts to different VRAM budgets.

## Measured runtime evidence

The RTX 4080 Qwen3 developer workload consumed 32 distinct prompts and
completed eight current-policy updates at 3.1914 GiB peak reserved VRAM. A
matched interruption/resume reproduced byte-identical trajectories, adapter
and optimizer tensors. The companion SmolLM2 workload completed the same run
shape at 1.4961 GiB peak reserved VRAM.

[Qwen3 workload](verl-opd-reference-workload.md){ .md-button }
[SmolLM2 workload](smollm2-opd-workload.md){ .md-button }

## Research record

The studies section preserves the actual outcome of each scoped experiment.
It includes a protocol-qualified teacher that tied supervised continuation, a
negative RecoveryBench result, regressions from a saturated Alignment Lab
starting point, and a preregistered external-alignment early stop.

- [Calculator protocol study](benchmarking.md)
- [RecoveryBench](recoverybench/recoverybench-v1.md)
- [Alignment Lab](alignment-lab/alignment-lab-v1.md)
- [External Alignment Gate](alignment-external/alignment-external-v1.md)

## Scope and boundaries

miniVERL's execution scope is one local process, one NVIDIA CUDA GPU and its
versioned profiles. Its scale-out path ends in a validated artifact handoff.
Read [compatibility](compatibility.md) for field and handoff semantics, and
[limitations](limitations.md) for the consolidated architecture, measurement,
security and generalization boundaries.
