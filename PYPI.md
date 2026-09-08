<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/v0.12.0/docs/banner.svg" alt="miniVERL — lower verl experiment semantics onto one CUDA GPU" width="880">
</p>

<div align="center">

[![CI](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml)
[![Build](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/miniverl.svg)](https://pypi.org/project/miniverl/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/LICENSE)

</div>

<p align="center">
  <a href="https://pypi.org/project/miniverl/"><strong>PyPI</strong></a> ·
  <a href="https://daoyuanli2816.github.io/mini-verl/"><strong>Stable docs</strong></a> ·
  <a href="https://daoyuanli2816.github.io/mini-verl/dev/">Development docs</a> ·
  <a href="https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/README.zh-CN.md">中文</a>
</p>

**miniVERL runs a validated subset of verl experiment semantics on one NVIDIA
GPU.** Give it a resolved verl-shaped config and Parquet prompts; its versioned
compiler produces a reviewable local plan, executes actor/reference/teacher/
reward roles in phases, and publishes portable PEFT and data artifacts.

The current development line adds critic-free RL against official verl
`v0.9.0` (`483b8a00`): GRPO, Dr.GRPO, RLOO and REINFORCE++, grouped rollouts,
task rewards and fixed reference-policy KL. The established verl `v0.8.0` OPD
profiles remain available for direct GKD and sampled-k1 distillation.

PyPI `v0.12.0` is stable; `main` is development.

## Start in 60 seconds

Install the CUDA-enabled PyTorch build that matches your machine, then:

```bash
python -m pip install "miniverl[train,cuda]"
miniverl data sample --task-rewards --rows 8 --out data/rl-prompts.parquet
miniverl import-verl --profile verl-rl-v0.9-single-gpu-v1 \
  --config examples/verl-rl-v0.9-single-gpu.yaml --out local-grpo.yaml
miniverl validate local-grpo.yaml
miniverl train local-grpo.yaml --dry-run
```

The importer writes `local-grpo.import-report.json` beside the native recipe.
It accounts for every accepted source field and records how distributed
resource settings lower to one process and one device. Remove `--dry-run` to
load the pinned Qwen3-0.6B actor and execute the two-iteration example.

The `[train,cuda]` extra installs the ML and quantization stack; select PyTorch's
CUDA build separately with the [PyTorch installer](https://pytorch.org/get-started/locally/).
The [single-GPU guide](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/single-gpu-guide.md) covers memory planning and the
maintainer-measured RTX 4080 environment.

## What a run gives you

- **A field-by-field compiler report.** Experiment fields retain their meaning;
  physical distribution fields receive an explicit one-GPU lowering.
- **Policy-bound trajectories.** Group/sample identity, generated-token spans,
  behavior log-probabilities and policy version travel together.
- **Auditable rewards and objectives.** Reward components, group advantages,
  reference KL, clipping, entropy and update metrics are structured data.
- **Exact recovery.** Transactional manifests and checkpoints restore policy,
  optimizer, cursor, RNG and reward identities before another rollout.
- **Portable outputs.** PEFT adapters, safetensors, Parquet, resolved config and
  typed provenance can move into a larger workflow.

## How it works

<picture>
  <source media="(max-width: 640px)" srcset="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/v0.12.0/docs/verl-local-runtime-mobile.svg">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/v0.12.0/docs/verl-local-runtime.svg" alt="A resolved verl config compiles into a validated single-GPU execution plan; actor, reference, teacher and reward roles execute in phases and produce portable artifacts plus a readiness report.">
</picture>

One GPU is treated as a temporal scheduler for logical roles. Critic-free RL
generates complete prompt groups, scores outcomes, computes the pinned
advantage estimator, optionally evaluates a frozen reference adapter, and then
updates the actor. OPD uses the same trajectory and checkpoint foundations,
with a teacher-scoring phase in place of outcome-only rewards.

## Choose your path

| Goal | First command | Primary artifact | Next step |
| --- | --- | --- | --- |
| **Run local RL** | `miniverl import-verl --profile verl-rl-v0.9-single-gpu-v1 --config verl-rl.yaml --out local.yaml` | native recipe + compatibility report | [RL quickstart](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/verl-rl-runtime.md) |
| **Run local OPD** | `miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml --out plan.json` | immutable execution plan | [OPD quickstart](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/opd-quickstart.md) |
| **Fit your GPU** | `miniverl plan --config verl-opd.yaml --probe` | measured placement plan | [Hardware planning](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/hardware-planning.md) |
| **Hand off artifacts** | `miniverl export-verl --run runs/my-run --target-verl v0.8.0 --out scaleout` | PEFT + Parquet + config bundle | [Scale-out contract](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/verl-opd-scaleout.md) |

Native recipes also support SFT, DPO and offline KD, plus calculator, JSON
navigation, read-only SQLite and custom tool environments.

## Current capability matrix

| Experiment surface | Local status | Contract |
| --- | --- | --- |
| GRPO / Dr.GRPO | semantically conformant | verl v0.9 group statistics and vanilla clipped policy loss |
| RLOO / REINFORCE++ | semantically conformant | verl v0.9 advantage and masking rules |
| Grouped `n > 1` rollouts | supported | complete groups, stable sample seeds and behavior-policy identity |
| Task rewards | supported | exact-answer/target-length built-ins, environment verifier, or trusted Python API |
| Fixed reference KL | locally lowered | frozen adapter role on the shared actor backbone |
| Direct GKD / sampled-k1 OPD | supported | pinned verl v0.8 profiles with teacher targets |
| PPO / GAE execution | not implemented | math is conformance-tested; a trainable critic lifecycle is not yet present |
| Ray, FSDP/FSDP2, Megatron, TP/PP/DP > 1 | distributed-only | these change physical scale, not the local objective |

The generated [v0.9 compatibility record](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/generated/verl-rl-v0.9-compatibility.json)
binds every example field to its source value, local target, classification and
compiler-rule digest. `miniverl import-verl` accepts a resolved documented
subset, rejects unknown algorithm-changing fields, and never substitutes a
dataset or reward implementation.

## Measured systems evidence

The published Qwen3-0.6B/1.7B OPD workload consumed 32 distinct prompts and
completed eight current-policy updates at 3.1914 GiB peak reserved VRAM on an
RTX 4080. A matched interruption after update four reproduced byte-identical
trajectories, adapter and optimizer tensors. The SmolLM2-360M/1.7B workload
completed the same shape at 1.4961 GiB. [System records](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/verl-opd-reference-workload.md)
contain configs, hashes and phase timings.

Rollout Runtime v2 measured `hf_cached` and managed vLLM across 24 RTX 4080
cells spanning response length, sampling mode and `n=1/4`. See the
[runtime report](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/benchmarks/rollout-runtime-v2.md). Evidence for the new
RL family is published through the exact-wheel release qualification rather
than presented as a task-quality comparison.

## Research record

The scientific reports preserve their original outcomes and frozen inputs:
[calculator protocol](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/benchmarking.md),
[RecoveryBench](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/recoverybench/recoverybench-v1.md),
[Alignment Lab](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/alignment-lab/alignment-lab-v1.md), and the
[External Alignment Gate](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/alignment-external/alignment-external-v1.md).
They are scoped studies, separate from runtime qualification.

## Compatibility boundary

miniVERL targets one local process and one NVIDIA CUDA GPU. The compiler
distinguishes exact or conformant semantics, local physical lowering,
distributed-only settings, feasible work not yet implemented, and technically
unsupported inputs. The [compatibility policy](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/compatibility.md) is the
complete matrix; [limitations](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/limitations.md) collects architecture,
measurement, security and generalization boundaries in one place. miniVERL is
an independent Apache-2.0 project.

## Development

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev,train]"
pytest -q -m "not gpu and not network"
```

See [CONTRIBUTING.md](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/CONTRIBUTING.md), [CHANGELOG.md](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/CHANGELOG.md),
[CITATION.cff](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/CITATION.cff), the [guide for verl users](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/docs/for-verl-users.md),
[SECURITY.md](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/SECURITY.md) and the
[Apache-2.0 license](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.12.0/LICENSE).
