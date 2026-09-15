<p align="center">
  <picture>
  <source media="(max-width: 760px)" srcset="docs/banner-mobile.svg">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — verl for a single consumer GPU" width="880">
  </picture>
</p>

<div align="center">

[![CI](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml)
[![Build](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/miniverl.svg)](https://pypi.org/project/miniverl/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

</div>

<p align="center">
  <a href="https://pypi.org/project/miniverl/"><strong>PyPI</strong></a> ·
  <a href="https://daoyuanli2816.github.io/mini-verl/"><strong>Stable docs</strong></a> ·
  <a href="https://daoyuanli2816.github.io/mini-verl/dev/">Development docs</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

**verl for a single consumer GPU.**

Run PPO, GRPO and on-policy distillation on one NVIDIA GPU. Bring your verl
configs, train locally, pick up interrupted runs and export your models.

PyPI `v0.16.0` is stable; `main` is development.

## Your first local experiment

Install the matching CUDA-enabled build using the [PyTorch installer](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install "miniverl[train,hydra]"
miniverl data sample --reward-profile target-length --rows 8 --out data/rl-prompts.parquet
miniverl run --example hydra-ppo --bind reward.provider=target_length --dry-run
miniverl run --example hydra-ppo --bind reward.provider=target_length --run-id local-ppo
```

This Qwen3-0.6B example trains with a simple length reward. Inspect it in
`rewards.jsonl`, or follow the [five-minute workflow](docs/for-verl-users.md)
to use GRPO, resume training and export an adapter.

Already have a verl experiment? Bring its config tree and Hydra overrides:

```bash
miniverl run --verl-config-path /path/to/verl/trainer/config \
  --verl-config-name ppo_trainer --dry-run \
  algorithm.adv_estimator=grpo actor_rollout_ref.rollout.n=8 trainer.n_gpus_per_node=8
```

The [direct-config guide](docs/direct-verl-config.md) walks through binding your
data and reward function. The [single-GPU guide](docs/single-gpu-guide.md)
helps you choose a model and memory settings. Install extras add training,
Hydra or quantization dependencies; choose the CUDA PyTorch build separately.

## Choose your path

| You want to… | Start here | Take away |
| --- | --- | --- |
| **Train with rewards** | [PPO / GRPO walkthrough](docs/for-verl-users.md) | A trained adapter, reward logs and resumable checkpoints |
| **Distill a teacher** | [OPD quickstart](docs/opd-quickstart.md) | A student trained on its own rollouts with teacher feedback |
| **Move to a larger setup** | [Export and hand off](docs/verl-opd-scaleout.md) | PEFT, Parquet and config files, plus a report of remaining setup |

SFT, DPO and offline KD are also available through native recipes.
[Hardware planning](docs/hardware-planning.md) helps fit a workload to your GPU.

## What a run gives you

- **Familiar configs.** Reuse verl fields and overrides; inspect how each maps to local execution.
- **Visible training.** Follow rollouts, rewards, losses and teacher targets.
- **Exact recovery.** Resume model, optimizer, data position and random state from a checkpoint.
- **Reusable outputs.** Export adapters, datasets and the configuration and source identities behind the run.

## How it works

<picture>
  <source media="(max-width: 760px)" srcset="docs/verl-local-runtime-mobile.svg">
  <img src="docs/verl-local-runtime.svg" alt="One GPU runs rollout, reward, reference, teacher and update phases in sequence, then exports models and data with a readiness report.">
</picture>

Actor, critic, reference, reward and teacher roles share the GPU in phases.
PPO trains an actor and a separate critic; GRPO uses grouped rewards; OPD learns
from teacher targets on the student's current rollouts.

## Measured results

On an RTX 4080, the Qwen3-0.6B/1.7B OPD workload completed eight updates over
32 distinct prompts at **3.1914 GiB** peak reserved VRAM. An interruption after
update four reproduced byte-identical trajectories, adapter and optimizer
tensors on resume. The SmolLM2-360M/1.7B workload used **1.4961 GiB**.
[Workload and measurements](docs/verl-opd-reference-workload.md) ·
[24-cell rollout-backend study](docs/benchmarks/rollout-runtime-v2.md).

For task outcomes, read the [calculator study](docs/benchmarking.md),
[RecoveryBench](docs/recoverybench/recoverybench-v1.md),
[Alignment Lab](docs/alignment-lab/alignment-lab-v1.md) and
[External Alignment Gate](docs/alignment-external/alignment-external-v1.md).
These include negative and mixed results, with their original data and scope.

## About and compatibility

miniVERL is an independent Apache-2.0 project for single-GPU NVIDIA CUDA
training. It supports documented verl configuration profiles; model fit
depends on GPU memory and workload. See the [compatibility matrix](docs/compatibility.md)
for supported upstream versions and semantics, and [limitations](docs/limitations.md)
for hardware coverage, scientific scope and the distributed-execution boundary.

## Development

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev,train]"
pytest -q -m "not gpu and not network"
```

[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) ·
[Citation](CITATION.cff) · [Security](SECURITY.md) · [License](LICENSE)
