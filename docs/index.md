# miniVERL

Auditable single-GPU LLM post-training for choosing, running and inspecting
SFT, DPO, knowledge distillation and strict OPD—plus a bounded artifact bridge
to one pinned verl profile. miniVERL is independent; no upstream endorsement is
implied, and distributed execution is not tested.

[Install and run locally](single-gpu-guide.md){ .md-button .md-button--primary }
[Read the compatibility boundary](verl-bridge.md){ .md-button }

## Install and verify in about a minute

Install the PyTorch build that matches your CPU or CUDA system first, then the
training extra. This CPU example is deterministic and downloads no model:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "miniverl[train]"
miniverl demo --fast --output runs/quickstart
miniverl inspect runs/quickstart/trajectories.jsonl
```

The result is a typed trajectory log, checksummed teacher cache, manifest and
self-contained report. For CUDA wheels and memory-aware recipes, use the
[single-GPU guide](single-gpu-guide.md).

## v0.7.0 evidence release: the external study stopped at its first gate

**0 selected checkpoints · 0 qualified teachers · 0 continuation arms · 0
final-test tasks accessed.** Every candidate in both declared lineages scored
0/64 retained JSONNav utility against an unchanged 20% floor. The result is a
preregistered checkpoint-selection failure, not a method comparison.

```bash
miniverl pilot --study-result benchmarks/results/alignment-external-v1.json --json
```

[Read the early-stop study](alignment-external/alignment-external-v1.md){ .md-button .md-button--primary }
[Inspect the evidence contract](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.7.0/benchmarks/results/alignment-external-v1.json){ .md-button }

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

Import only the documented profile, convert Parquet, export standard artifacts
and inspect the unsupported boundary.

```bash
miniverl bridge doctor exports/my-bundle --json
```

**Artifact:** `provenance/compatibility-report.json` with separate readiness
flags; current bundles are not launchable.

**Next:** [verl bridge contract](verl-bridge.md)

</div>

</div>

## Measured evidence, kept scoped

The Alignment Lab case study starts from an SFT checkpoint already at 100%
alignment and 100% retained tool utility on its deterministic sandbox suite.
No continuation method improves it; completed regressions remain visible.
External IFEval, XSTest, HarmBench and RewardBench endpoints were not executed.

<picture class="alignment-figure">
  <source media="(max-width: 900px)" srcset="alignment-lab/delta-from-sft-mobile.svg">
  <img src="alignment-lab/delta-from-sft.svg" alt="Forest chart of continuation-method alignment and retained-tool-utility percentage-point deltas from the saturated SFT checkpoint, with every seed and mean printed as text.">
</picture>

The consumer runtime result is a systems result, not a new quality claim:
shared-backbone role switching and padded trajectory updates reduce measured
memory/runtime overhead while preserving the tested local objective. See the
[Consumer Runtime report](consumer-runtime/index.md) and
[RecoveryBench](recoverybench/recoverybench-v1.md) for full evidence and limits.
