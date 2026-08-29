<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — run verl-style OPD on one consumer GPU" width="880">
</p>

<div align="center">

[![CI](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml)
[![Build](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/miniverl.svg)](https://pypi.org/project/miniverl/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/DaoyuanLi2816/mini-verl/blob/main/LICENSE)

</div>

<p align="center">
  <a href="https://pypi.org/project/miniverl/"><strong>PyPI</strong></a> ·
  <a href="https://daoyuanli2816.github.io/mini-verl/"><strong>Stable docs</strong></a> ·
  <a href="https://daoyuanli2816.github.io/mini-verl/dev/">Development docs</a> ·
  <a href="https://github.com/DaoyuanLi2816/mini-verl/blob/main/README.zh-CN.md">中文</a>
</p>

**Run verl-style on-policy distillation on one NVIDIA GPU, with every config
mapping, teacher target and training artifact available for inspection.**
miniVERL turns typed YAML and structured Parquet prompts into a local actor
rollout → teacher scoring → actor update loop, then exports standard PEFT,
Parquet and config artifacts for scale-out work.

PyPI `v0.10.1` is stable; `main` is development.

## Start in 60 seconds

Install the CUDA-enabled PyTorch build that matches your machine, then:

```bash
python -m pip install "miniverl[train,cuda]"
miniverl data sample --format verl-parquet --out prompts.parquet
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 \
  --config builtin:qwen3-0.6b-1.7b-opd \
  --set 'data.train_files=["prompts.parquet"]' --out plan.json
miniverl run --profile verl-opd-v0.8-single-gpu-v1 \
  --plan plan.json --dry-run
```

This path works from the published wheel and loads no model weights while
planning. The generated `plan.json` binds the source config, ordered overrides,
profile version and input Parquet bytes. On a CUDA GPU, remove `--dry-run` to
run the pinned Qwen3-0.6B actor and Qwen3-1.7B teacher recipe.

The `[train,cuda]` extra installs the ML and quantization dependencies. Select
the matching CUDA PyTorch wheel separately with the
[PyTorch installer](https://pytorch.org/get-started/locally/). The
[single-GPU guide](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/single-gpu-guide.md) covers memory planning from 8 GiB
cards upward and includes the maintainer-measured RTX 4080 stack.

## What a run gives you

- **A reviewable plan.** Every accepted verl field has a local effect,
  classification and risk level before weights are loaded.
- **Strict current-policy trajectories.** The actor policy version, token
  spans and teacher-supervised positions travel together.
- **Compact teacher targets.** Top-k targets and sampled-k1 signals use
  checksummed, pickle-free caches.
- **Recoverable training.** Transactional manifests, checkpoints and cache
  indexes support inspection and exact resume.
- **Standard outputs.** PEFT adapters, safetensors, structured Parquet,
  resolved config and typed provenance remain portable.

```bash
miniverl run --profile verl-opd-v0.8-single-gpu-v1 --plan plan.json \
  --output runs --run-id my-opd
miniverl inspect runs/my-opd/trajectories.jsonl
miniverl cache stats runs/my-opd/teacher-cache
miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout
miniverl bridge doctor scaleout --json
```

## How it works

<picture>
  <source media="(max-width: 640px)" srcset="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/verl-local-runtime-mobile.svg">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/verl-local-runtime.svg" alt="Typed verl-style configuration and Parquet prompts compile into sequential actor rollout, teacher scoring and actor update phases on one CUDA GPU, producing inspectable local artifacts and a pinned scale-out bundle.">
</picture>

miniVERL schedules actor, teacher and optional reference roles in phases inside
one ordinary process. The actor generates with its current adapter, the teacher
scores the visited token positions, and the actor receives a padded token-mean
update. Resident, swap and shared-backbone placement strategies keep those role
identities explicit while adapting to available memory.

Each phase publishes evidence before the next boundary: structured
trajectories carry per-token provenance, teacher targets are checksummed,
checkpoints publish transactionally, and the final adapter is reloaded through
standard PEFT. The result is a training run you can inspect without
reconstructing intent from console logs.

## Choose your path

| Goal | First command | Primary artifact | Next step |
| --- | --- | --- | --- |
| **Run local OPD** | `miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml` | immutable execution plan | [OPD quickstart](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/opd-quickstart.md) |
| **Bring a verl profile** | `miniverl compat check --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml` | field-by-field compatibility report | [For verl users](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/for-verl-users.md) |
| **Fit your GPU** | `miniverl plan --config verl-opd.yaml --probe` | measured placement plan | [Hardware planning](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/hardware-planning.md) |
| **Hand off for scale-out** | `miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout` | PEFT + Parquet + config bundle | [Scale-out contract](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/verl-opd-scaleout.md) |

The native recipe system also supports SFT, DPO, offline KD and tool-aware OPD
over calculator, JSON navigation, read-only SQLite and custom environments.

## Familiar verl inputs, local execution

The current profiles pin official verl `v0.8.0` at
`7aed6b230776f963fa09509c10d9c3a767d1102c` and preserve recognizable fields:

```yaml
actor_rollout_ref:
  model: {path: Qwen/Qwen3-0.6B}
  rollout: {name: vllm, n: 1}
distillation:
  teacher_models:
    teacher_model:
      model_path: Qwen/Qwen3-1.7B
      inference: {name: vllm}
```

The compiler translates distributed resource intent into sequential local
Hugging Face phases and records that translation in the plan. Two measured
profiles are available:

| Profile | Objective | Teacher target |
| --- | --- | --- |
| `verl-opd-v0.8-single-gpu-v1` | direct GKD `forward_kl_topk` | top-k token IDs and log-probabilities |
| `verl-opd-v0.8-single-gpu-pg-k1-v1` | sampled `k1` + vanilla policy loss | sampled-token teacher log-probability |

Use `miniverl profiles show`, `compat explain` and `compat check` to inspect
the complete mapping. [Compatibility profiles](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/profiles/index.md) explains
how profile identity follows plans, caches, checkpoints and exports. Separate
conformance-only grouped profiles add transactional Parquet `n>1` samples
without changing either measured `n=1` profile or introducing GRPO semantics.
A separate conformance-only rewarded profile adds deterministic exact-answer
rewards and explicit group advantage composition; it has no task-quality claim.

## Measured systems evidence

The Qwen3-0.6B/1.7B developer workload consumed **32 distinct prompts**, used a
64-token response bound, and completed **8 current-policy updates** at
**3.1914 GiB peak reserved VRAM** on an RTX 4080. Median steady-state rollout,
teacher-scoring and update times were 9.7200, 0.4864 and 2.3260 seconds. A
matched interruption after update four resumed to byte-identical trajectories,
adapter and optimizer tensors.

A second SmolLM2-360M/1.7B workload completed the same 32-prompt, 8-update
shape at 1.4961 GiB peak reserved VRAM, including exact resume, PEFT reload and
scale-out materialization. Read the
[Qwen3](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/verl-opd-reference-workload.md) and
[SmolLM2](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/smollm2-opd-workload.md) systems records for configs, hashes and
phase-level measurements.

Other NVIDIA GPUs use the same device-name-agnostic CUDA path. Model fit and
speed vary with VRAM, context, quantization, kernels and software versions;
`miniverl doctor` and `plan --probe` expose those machine-specific choices.

## Research record

The repository publishes positive, mixed and negative results with the same
resolved configs and source hashes. The calculator study found that a
protocol-qualified teacher prevented collapse but tied supervised
continuation. RecoveryBench found no fresh-state advantage in its scoped
SQLite setting. Alignment Lab began at a saturated SFT checkpoint and exposed
utility regressions that two sandbox safety checks missed. The preregistered
External Alignment Gate stopped before continuation because every candidate
failed the retained-utility threshold.

- [Calculator protocol study](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/benchmarking.md)
- [RecoveryBench](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/recoverybench/recoverybench-v1.md)
- [Alignment Lab](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/alignment-lab/alignment-lab-v1.md)
- [External Alignment Gate](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/alignment-external/alignment-external-v1.md)

These reports answer scoped experimental questions; the
[limitations](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/limitations.md) page collects the measurement, architecture,
security and generalization boundaries in one place.

## Scope

miniVERL is designed for one local process, one NVIDIA CUDA GPU and the
documented OPD profiles above. Scale-out support produces and validates a
portable bundle against the pinned upstream source. The
[compatibility policy](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/compatibility.md) lists supported fields, profile
semantics and handoff readiness states; [limitations](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/limitations.md)
covers the broader execution and scientific boundaries. miniVERL is an
independent Apache-2.0 project.

## Development

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev]"
pytest -q -m "not gpu and not network"
```

See [CONTRIBUTING.md](https://github.com/DaoyuanLi2816/mini-verl/blob/main/CONTRIBUTING.md), [CHANGELOG.md](https://github.com/DaoyuanLi2816/mini-verl/blob/main/CHANGELOG.md),
[CITATION.cff](https://github.com/DaoyuanLi2816/mini-verl/blob/main/CITATION.cff), the
[reproducibility guide](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/reproducibility.md), [SECURITY.md](https://github.com/DaoyuanLi2816/mini-verl/blob/main/SECURITY.md)
and the [Apache-2.0 license](https://github.com/DaoyuanLi2816/mini-verl/blob/main/LICENSE).
