<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — single-GPU LLM post-training" width="880">
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

**miniVERL is a local, inspectable single-GPU alignment and distillation
runtime.** It runs native SFT, DPO, KD and strict OPD recipes, preserves
assistant-only loss masks and policy-version provenance, and exchanges standard
HF/PEFT/Parquet artifacts through a fail-closed bridge to one pinned verl
profile.

PyPI `v0.7.1` is stable; `main` is development. miniVERL is independent from
verl. It does not claim arbitrary verl YAML execution, distributed execution,
or full algorithmic compatibility.

## Install and verify in about a minute

```bash
python -m pip install "miniverl[train]"
miniverl doctor
miniverl demo --fast --output runs/quickstart
miniverl inspect runs/quickstart/trajectories.jsonl
miniverl evidence validate alignment-external-v1
```

The deterministic demo downloads no model and produces typed trajectories, a
checksummed teacher cache, manifest and report. The evidence command reads
self-contained package data; it works from a wheel without a Git checkout.
For schemas and inspection without the ML stack, install `miniverl` alone.

## Supported hardware and runtime boundary

miniVERL runs one local process on CPU or one NVIDIA CUDA GPU. The CUDA path is
device-name agnostic, but fit depends on model pair, context, kernels and VRAM.
Install the matching CUDA-enabled PyTorch build first, then
`miniverl[train,cuda]`; that extra does not select a CUDA PyTorch wheel.
Ray, FSDP, Megatron, PPO, GRPO and distributed launch are outside the runtime.
See the [single-GPU guide](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/single-gpu-guide.md).

## verl compatibility summary

The bridge targets official verl `v0.8.0` at commit `7aed6b23`. Its verified
boundary is checksummed standard artifacts plus pinned config-parse and
model/data-load smoke—not native checkpoint parity or a completed verl job.
Imports fail closed when dataset, environment, teacher, objective or schedule
semantics are unresolved; they never substitute calculator tasks or invent an
unqualified teacher.

Current exports remain `launchable: false`: the base snapshot is absent, the
reward scaffold fails closed and required mappings remain placeholders. The
entry point is `launch.template.sh`; readiness, parse/load evidence,
launchability, distributed execution and semantic parity are separate facts.
[Read the bridge contract](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/verl-bridge.md).

## One measured systems result

On one RTX 4080 with Qwen3-0.6B and eight fixed SQLite trajectories, physical
batch 4 increased dual-model update throughput from 2.369 to 3.866
trajectories/s. Shared-backbone batch 4 used 2.227 GiB peak reserved memory
versus 3.035 GiB for dual model while running 10.1% slower. All 12
preregistered equivalence comparisons passed. This is one workload on one
machine, not a promise for other GPUs.

![Measured throughput and reserved VRAM for dual-model and shared-backbone runtime cells](https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/consumer-runtime-v1-pareto.svg)

[Consumer Runtime v1 methods and caveats](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/consumer-runtime-v1.md)

## Three paths

| Path | Start with | Concrete artifact | Next |
| --- | --- | --- | --- |
| **Align** — use SFT, DPO, KD or OPD only when pilot evidence supports the cost | `miniverl pilot recipes/alignment_policy_conditioned_qwen.yaml` | `alignment-card.json` | [Alignment Lab](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/alignment-lab/alignment-lab-v1.md) |
| **Distill locally** — strict OPD, shared backbones and padded updates on one CUDA GPU | `miniverl train recipes/qwen_consumer_gpu_shared.yaml --dry-run` | resolved config and revision-pinned PEFT adapter | [Bring your own GPU](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/single-gpu-guide.md) |
| **Scale out** — convert Parquet, export standard artifacts and inspect the unsupported boundary | `miniverl bridge doctor scaleout-bundle` | `provenance/compatibility-report.json` | [Verified artifact bridge](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/verl-bridge.md) |

## Research notes and preserved negative evidence

### v0.7 External Alignment Gate

The preregistered external study stopped before teacher or method training.
Both declared starting-policy lineages scored **0/64** retained JSONNav utility
for every candidate against the unchanged 20% floor.

| selected checkpoints | qualified teachers | continuation arms | final-test tasks accessed |
| ---: | ---: | ---: | ---: |
| **0** | **0** | **0** | **0** |

```bash
miniverl pilot --builtin-study alignment-external-v1 --json
```

The result is `do_not_continue_this_study` and `insufficient_evidence`, not a
recommendation among SFT/DPO/KD/OPD. Granite Guardian values are unqualified
selection diagnostics; Granite, PairRM and teacher qualification and the
reserved final test did not run. [Study and limitations](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/alignment-external/alignment-external-v1.md).

### Earlier measured alignment case study

Alignment Lab v1 began from an SFT checkpoint already at 100% policy compliance
and 100% retained tool utility in all three seeds. No continuation improved the
ceiling; continued SFT and both OPD variants retained measured regressions.
The two sandbox safety checks tied at zero while utility still regressed.
IFEval, XSTest, HarmBench and RewardBench were not executed, and “preference
win rate” is a deterministic Minipolicy paired outcome, not human preference.
[Seed-level evidence](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/alignment-lab/alignment-lab-v1.md).

- [RecoveryBench v1](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/recoverybench/recoverybench-v1.md): frozen-student KD
  beat slower fresh-state OPD on the preregistered primary view; the verifier
  gate remained `insufficient_evidence`.
- [Calculator benchmark](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/benchmarking.md): both negative controls completed
  normally at 0%; the ambiguous historical protocol-v1 prompt prevents
  attributing failure solely to intrinsic teacher behavior.
- [Limitations](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/limitations.md), [math](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/math.md),
  [reproducibility](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/reproducibility.md) and
  [compatibility policy](https://github.com/DaoyuanLi2816/mini-verl/blob/main/docs/compatibility.md).

New runs establish tokenizer compatibility through structural identity. The
legacy behavioral fingerprint is only a migration fallback, not identity proof.

## Develop

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev]"
pytest -q -m "not gpu and not network"
```

Apache-2.0 licensed. See [CONTRIBUTING.md](https://github.com/DaoyuanLi2816/mini-verl/blob/main/CONTRIBUTING.md),
[SECURITY.md](https://github.com/DaoyuanLi2816/mini-verl/blob/main/SECURITY.md), the [changelog](https://github.com/DaoyuanLi2816/mini-verl/blob/main/CHANGELOG.md) and
[citation](https://github.com/DaoyuanLi2816/mini-verl/blob/main/CITATION.cff). Project records: [default GPU recipe](https://github.com/DaoyuanLi2816/mini-verl/blob/main/recipes/qwen_consumer_gpu_calc.yaml),
[frozen calculator result](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/gpu-calc-hard-equal-update-v2.json)
and [license](https://github.com/DaoyuanLi2816/mini-verl/blob/main/LICENSE).
