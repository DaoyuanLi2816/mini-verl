# miniVERL

Compile supported verl experiment semantics into a validated execution plan for
one NVIDIA GPU. The same local runtime produces policy-bound trajectories,
transactional checkpoints and portable PEFT/Parquet artifacts.

[Run verl-shaped RL locally](verl-rl-runtime.md){ .md-button .md-button--primary }
[Choose a workflow](comparisons.md){ .md-button }

## From a verl config to one GPU

```text
resolved verl-shaped config
        ↓  versioned compatibility compiler
field report + validated native recipe
        ↓  temporal role scheduler
actor rollout → reward / reference / teacher → actor update
        ↓
PEFT + Parquet + config + typed provenance
```

The current RL profile targets official verl `v0.9.0` at commit
`483b8a009ba3a97563edee3a19887e4862b8094a`. It runs GRPO, Dr.GRPO,
RLOO and REINFORCE++ with grouped samples, task rewards and optional fixed
reference-policy KL. The existing verl `v0.8.0` profiles run direct GKD and
sampled-k1 OPD with explicit teacher targets.

## Start with GRPO

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install "miniverl[train,cuda]"
miniverl data sample --task-rewards --rows 8 --out data/rl-prompts.parquet
miniverl import-verl --profile verl-rl-v0.9-single-gpu-v1 \
  --config examples/verl-rl-v0.9-single-gpu.yaml --out local-grpo.yaml
miniverl train local-grpo.yaml --dry-run
```

The import report shows the value and disposition of every source field. The
generated recipe has already passed `RunConfig` validation and can be inspected
with `miniverl validate local-grpo.yaml --json` before model weights are loaded.

## Three ways to use miniVERL

<div class="path-grid" markdown>

<div class="path-card" markdown>

### Prototype verl RL

Bring a resolved v0.9-shaped config and run a critic-free algorithm with true
grouped rollouts on one CUDA device.

```bash
miniverl import-verl --profile verl-rl-v0.9-single-gpu-v1 \
  --config verl-rl.yaml --out local.yaml
```

**Artifact:** native recipe plus field-by-field compatibility report.

**Next:** [Single-GPU verl RL](verl-rl-runtime.md)

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
miniverl export-verl --run runs/my-run --target-verl v0.8.0 --out scaleout
```

**Artifact:** checksummed portable bundle and compatibility states.

**Next:** [Scale-out contract](verl-opd-scaleout.md)

</div>

</div>

## Capability map

| Surface | Local status |
| --- | --- |
| GRPO, Dr.GRPO, RLOO, REINFORCE++ | verl v0.9-conformant estimator and policy-loss semantics |
| Grouped rollout, task rewards, fixed reference KL | supported and provenance-bound |
| Direct GKD and sampled-k1 OPD | supported through pinned verl v0.8 profiles |
| SFT, DPO, offline KD, tool environments | native recipe workflows |
| PPO/GAE execution | value math tested; trainable critic lifecycle not yet implemented |
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
