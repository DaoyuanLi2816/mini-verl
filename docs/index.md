# miniVERL

Auditable single-GPU alignment and distillation runtime with native SFT, DPO,
KD and strict OPD recipes, inspectable artifacts and a bounded bridge to one
pinned verl profile. miniVERL is independent; distributed execution and full
algorithm compatibility are not claimed.

[Install and run locally](single-gpu-guide.md){ .md-button .md-button--primary }
[Read the compatibility boundary](verl-bridge.md){ .md-button }

## Install and verify in about a minute

```bash
python -m pip install "miniverl[train]"
miniverl demo --fast --output runs/quickstart
miniverl inspect runs/quickstart/trajectories.jsonl
miniverl evidence validate alignment-external-v1
```

The deterministic demo downloads no model and produces typed trajectories, a
checksummed teacher cache, manifest and report. Packaged evidence commands need
no repository checkout. For CUDA, install the matching CUDA-enabled PyTorch
wheel first; the `[cuda]` extra does not select one.

## Runtime and compatibility boundary

miniVERL runs one local CPU process or one NVIDIA CUDA GPU. Fit depends on the
model pair, context, kernels and VRAM; there is no GPU-name allowlist. Ray,
FSDP, Megatron, PPO, GRPO and distributed launch are outside the runtime.

The artifact bridge pins verl `v0.8.0` at `7aed6b23`. It verifies standard
artifact interchange and pinned config/model/data parse-load smoke. Current
exports are not launchable and do not establish algorithmic parity.

## Measured systems evidence

On one RTX 4080 with Qwen3-0.6B and eight fixed SQLite trajectories, padded
updates increased dual-model update throughput from 2.369 to 3.866
trajectories/s. Shared-backbone batch 4 used 2.227 GiB peak reserved memory
versus 3.035 GiB for dual model while running 10.1% slower. All 12
preregistered equivalence comparisons passed. This is one measured workload,
not a hardware-wide promise.

[Consumer Runtime methods and caveats](consumer-runtime/index.md){ .md-button }

## Choose a path

<div class="path-grid" markdown>

<div class="path-card" markdown>

## Align

Choose SFT, DPO, offline KD or OPD from explicit pilot evidence.

```bash
miniverl pilot recipes/alignment_tool_policy_toy.yaml --json
```

**Artifact:** an [Alignment Card](alignment-lab/alignment-lab-v1.md#reproducibility-and-artifacts)
with starting checkpoint, metrics, cost and limitations.

**Next:** [When should OPD follow SFT?](alignment-lab/when-opd-should-follow-sft.md)

</div>

<div class="path-card" markdown>

## Distill locally

Use strict OPD, shared-backbone role switching and padded trajectory updates on
one CUDA GPU.

```bash
miniverl train recipes/qwen_consumer_gpu_shared.yaml --dry-run --json
```

**Artifact:** a resolved recipe and typed provenance plan before model loading.

**Next:** [Consumer-GPU shared runtime](consumer-runtime/index.md)

</div>

<div class="path-card" markdown>

## Scale out

Convert Parquet, export standard artifacts and inspect the unsupported boundary.

```bash
miniverl bridge doctor exports/my-bundle --json
```

**Artifact:** `provenance/compatibility-report.json` with separate readiness
flags; current bundles are not launchable.

**Next:** [verl bridge contract](verl-bridge.md)

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
