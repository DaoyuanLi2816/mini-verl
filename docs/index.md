# miniVERL

Run common resolved verl PPO/GRPO configs directly on one NVIDIA GPU.
The local runtime produces policy-bound trajectories,
transactional checkpoints and portable PEFT/Parquet artifacts.

[Run PPO or GRPO locally](local-rl-workflow.md){ .md-button .md-button--primary }
[Choose a workflow](comparisons.md){ .md-button }

## From a verl config to one GPU

```text
resolved verl-shaped config
        ↓  versioned compatibility compiler
field report + validated native recipe
        ↓  temporal role scheduler
actor rollout → reward / reference / teacher → actor update
                 ↘ PPO critic/value update
        ↓
PEFT + Parquet + config + typed provenance
```

The current RL profile targets official verl `v0.9.0` at commit
`483b8a009ba3a97563edee3a19887e4862b8094a`. It runs PPO/GAE, GRPO, Dr.GRPO,
RLOO and REINFORCE++ with grouped samples, task rewards, trained reward-model
inference and optional fixed reference-policy KL. The existing verl `v0.8.0` profiles run direct GKD and
sampled-k1 OPD with explicit teacher targets.

## Start with PPO

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install "miniverl[train]"
miniverl data sample --reward-profile target-length --rows 8 --out data/rl-prompts.parquet
miniverl run --example ppo --bind reward.provider=target_length --dry-run
miniverl run --example ppo --bind reward.provider=target_length --run-id local-ppo
```

Replace `--example ppo` with your resolved verl YAML to use an existing experiment.
`--dry-run` checks semantic compatibility without model downloads; execution
resolves snapshots, filters prompts, derives the epoch schedule and checks GPU
capacity. Continue with [direct run → inspect → resume → handoff](direct-verl-config.md).

## Three ways to use miniVERL

<div class="path-grid" markdown>

<div class="path-card" markdown>

### Prototype verl RL

Bring a resolved v0.9-shaped config and run PPO or a grouped critic-free
algorithm on one CUDA device.

```bash
miniverl run verl-rl.yaml --dry-run
```

**Artifact:** native recipe plus field-by-field compatibility report.

**Next:** [Direct verl configs](direct-verl-config.md)

</div>

<div class="path-card" markdown>

### Distill locally

Compile a direct-GKD or sampled-k1 OPD profile, then inspect teacher targets,
trajectory provenance and the final adapter.

```bash
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 \
  --config verl-opd.yaml --out plan.json
```

**Artifact:** immutable plan, teacher cache, trajectories and PEFT adapter.

**Next:** [OPD quickstart](opd-quickstart.md)

</div>

<div class="path-card" markdown>

### Prepare scale-out artifacts

Package a local run into standard PEFT, safetensors, Parquet and config
artifacts with a separate readiness report.

```bash
miniverl export-verl --run runs/my-run --target-verl v0.9.0 --out scaleout
```

**Artifact:** checksummed portable bundle and compatibility states.

**Next:** [Inspect and hand off](local-rl-workflow.md)

</div>

</div>

## Capability map

| Surface | Local status |
| --- | --- |
| PPO/GAE | independent actor/critic updates with exact dual-role resume |
| GRPO, Dr.GRPO, RLOO, REINFORCE++ | verl v0.9-conformant estimator and policy-loss semantics |
| Grouped rollout, task rewards, trained RM, actor KL/entropy | supported and provenance-bound |
| Direct GKD and sampled-k1 OPD | supported through pinned verl v0.8 profiles |
| SFT, DPO, offline KD, tool environments | native recipe workflows |
| Ray, FSDP/FSDP2, Megatron, TP/PP/DP > 1 | distributed-only |

Read [compatibility](compatibility.md) for the complete field and semantic
contract. [Limitations](limitations.md) collects measurement, architecture,
security and generalization boundaries.

## Evidence and research

The RTX 4080 systems record covers current-policy OPD updates, exact resume and
rollout backends. The v0.9 RL path adds upstream numerical/gradient conformance,
CPU integration and exact-wheel GPU qualification. Runtime qualification is
kept separate from task-quality claims.

- [RTX 4080 workload](verl-opd-reference-workload.md)
- [Rollout Runtime v2](benchmarks/rollout-runtime-v2.md)
- [Scientific studies](alignment-lab/alignment-lab-v1.md)
- [For verl users](for-verl-users.md)
