<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — lower verl experiment semantics onto one CUDA GPU" width="880">
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

**Run common verl PPO/GRPO configs directly on one NVIDIA GPU.** Hand miniVERL
your resolved upstream YAML and bind local data or reward code. Its versioned
compiler preserves the experiment while scheduling actor, critic, reference
and reward roles in phases. No second configuration language to maintain.

The current development line covers PPO/GAE, GRPO, Dr.GRPO, RLOO and
REINFORCE++ against official verl `v0.9.0` (`483b8a00`). PPO uses an independent
trainable critic with its own optimizer and checkpoint state; actor KL, entropy
regularization, grouped rollouts, task rewards and a pinned sequence-classifier
reward role share the same provenance model. The established verl `v0.8.0` OPD
profiles remain available for direct GKD and sampled-k1 distillation.

PyPI `v0.15.0` is stable; `main` is development.

## Your first local experiment

Install the CUDA-enabled PyTorch build that matches your machine, then:

```bash
python -m pip install "miniverl[train]"
miniverl data sample --reward-profile target-length --rows 8 --out data/rl-prompts.parquet
miniverl run --example ppo --bind reward.provider=target_length --dry-run
miniverl run --example ppo --bind reward.provider=target_length --run-id local-ppo
```

PPO and GRPO inputs ship in upstream format in the wheel. These bounded
Qwen3-0.6B examples run two iterations; the explicit binding selects a small
length-reward exercise. To use your own resolved config, replace `--example ppo`
with `resolved-verl.yaml`. Each run keeps the original bytes, field decisions,
bindings, resolved snapshots, hardware plan and native IR. Inspect, resume and export:

```bash
miniverl inspect runs/local-ppo
miniverl run --example ppo --bind reward.provider=target_length \
  --resume-from runs/local-ppo/checkpoints/step-000002
miniverl export-adapter --run runs/local-ppo --out runs/local-ppo/model
miniverl export-verl --run runs/local-ppo --target-verl v0.9.0 --out ppo-handoff
miniverl bridge doctor ppo-handoff --json
```

The [five-minute workflow](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/for-verl-users.md) explains each artifact and
offers the matching GRPO commands. These are small length-reward exercises;
their reward is directly inspectable in `rewards.jsonl`.

`[train]` supplies the ML stack; `[cuda]` additionally supplies quantization.
Select PyTorch's CUDA build with the [PyTorch installer](https://pytorch.org/get-started/locally/).
The [single-GPU guide](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/single-gpu-guide.md) covers memory planning and the
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
  <source media="(max-width: 640px)" srcset="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/verl-local-runtime-mobile.svg">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/verl-local-runtime.svg" alt="A resolved verl config compiles into a validated single-GPU plan; actor, critic, reference, teacher and reward roles run in phases and produce portable artifacts plus a readiness report.">
</picture>

One GPU is treated as a temporal scheduler for logical roles. RL generates
complete prompt groups, scores outcomes, evaluates optional reference and
reward-model roles, computes the pinned advantage estimator, and updates the
actor. PPO adds a separate critic phase with clipped value updates. OPD uses
the same trajectory and checkpoint foundations, with teacher targets supplying
the learning signal.

## Choose your path

| Goal | First command | Primary artifact | Next step |
| --- | --- | --- | --- |
| **Run local RL** | `miniverl run resolved-verl.yaml --dry-run` | semantic report + local plan | [Direct configs](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/direct-verl-config.md) |
| **Run local OPD** | `miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml --out plan.json` | immutable execution plan | [OPD quickstart](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/opd-quickstart.md) |
| **Fit your GPU** | `miniverl plan --config verl-opd.yaml --probe` | measured placement plan | [Hardware planning](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/hardware-planning.md) |
| **Hand off artifacts** | `miniverl export-verl --run runs/my-run --target-verl v0.9.0 --out scaleout` | actor, critic, Parquet + config bundle | [Compatibility contract](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/compatibility.md) |

Native recipes also support SFT, DPO and offline KD, plus calculator, JSON
navigation, read-only SQLite and custom tool environments.

## Current capability matrix

| Experiment surface | Local status | Contract |
| --- | --- | --- |
| PPO / GAE | semantically conformant | independent critic, clipped value loss, actor/critic exact resume |
| GRPO / Dr.GRPO | semantically conformant | verl v0.9 group statistics and vanilla clipped policy loss |
| RLOO / REINFORCE++ | semantically conformant | verl v0.9 advantage and masking rules |
| Grouped `n > 1` rollouts | supported | complete groups, stable sample seeds and behavior-policy identity |
| Task rewards and trained RM | supported | built-ins, environment verifier, trusted Python API, or pinned HF sequence classifier |
| Actor KL and entropy | semantically conformant | sampled-token reference KL and entropy regularization |
| Direct GKD / sampled-k1 OPD | supported | pinned verl v0.8 profiles with teacher targets |
| Ray, FSDP/FSDP2, Megatron, TP/PP/DP > 1 | distributed-only | these change physical scale, not the local objective |

The [upstream compatibility corpus](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/verl-compatibility-corpus.md) resolves
real verl examples and records every field's outcome. The v4 direct compiler
adds prompt filtering, epoch scheduling and common loss reductions, separating
semantic acceptance from missing inputs and hardware capacity. v1/v2/v3 remain
available for existing recipes.

## Measured systems evidence

The published Qwen3-0.6B/1.7B OPD workload consumed 32 distinct prompts and
completed eight current-policy updates at 3.1914 GiB peak reserved VRAM on an
RTX 4080. A matched interruption after update four reproduced byte-identical
trajectories, adapter and optimizer tensors. The SmolLM2-360M/1.7B workload
completed the same shape at 1.4961 GiB. [System records](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/verl-opd-reference-workload.md)
contain configs, hashes and phase timings.

Rollout Runtime v2 measured `hf_cached` and managed vLLM across 24 RTX 4080
cells spanning response length, sampling mode and `n=1/4`. See the
[runtime report](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/benchmarks/rollout-runtime-v2.md). Evidence for the new
RL family is published through the exact-wheel release qualification rather
than presented as a task-quality comparison.

## Research record

The scientific reports preserve their original outcomes and frozen inputs:
[calculator protocol](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/benchmarking.md),
[RecoveryBench](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/recoverybench/recoverybench-v1.md),
[Alignment Lab](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/alignment-lab/alignment-lab-v1.md), and the
[External Alignment Gate](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/alignment-external/alignment-external-v1.md).
They are scoped studies, separate from runtime qualification.

## Compatibility boundary

miniVERL targets one local process and one NVIDIA CUDA GPU. The compiler
distinguishes exact or conformant semantics, local physical lowering,
distributed-only settings, feasible work not yet implemented, and technically
unsupported inputs. The [compatibility policy](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/compatibility.md) is the
complete matrix; [limitations](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/limitations.md) collects architecture,
measurement, security and generalization boundaries in one place. miniVERL is
an independent Apache-2.0 project.

## Development

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev,train]"
pytest -q -m "not gpu and not network"
```

See [CONTRIBUTING.md](https://github.com/DaoyuanLi2816/mini-verl/blob/main/CONTRIBUTING.md), [CHANGELOG.md](https://github.com/DaoyuanLi2816/mini-verl/blob/main/CHANGELOG.md),
[CITATION.cff](https://github.com/DaoyuanLi2816/mini-verl/blob/main/CITATION.cff), the [guide for verl users](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/for-verl-users.md),
[SECURITY.md](https://github.com/DaoyuanLi2816/mini-verl/blob/main/SECURITY.md) and the
[Apache-2.0 license](https://github.com/DaoyuanLi2816/mini-verl/blob/main/LICENSE).
