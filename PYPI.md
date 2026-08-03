<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/v0.6.1/docs/banner.svg" alt="miniVERL — single-GPU LLM post-training" width="880">
</p>

<div align="center">

[![CI](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml)
[![Build](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/miniverl.svg)](https://pypi.org/project/miniverl/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/LICENSE)

</div>

<p align="center">
  <a href="https://pypi.org/project/miniverl/"><strong>PyPI</strong></a> ·
  <a href="https://daoyuanli2816.github.io/mini-verl/"><strong>Stable docs</strong></a> ·
  <a href="https://daoyuanli2816.github.io/mini-verl/dev/">Development docs</a> ·
  <a href="https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/README.zh-CN.md">中文</a>
</p>

**miniVERL is a local, inspectable runtime for a documented subset of
single-GPU LLM alignment and distillation.** It keeps rollout provenance,
assistant-only loss masks, teacher targets, update budgets and run artifacts
explicit, then exports portable artifacts through a fail-closed bridge to one
pinned upstream verl profile.

PyPI `v0.6.1` is stable; `main` is development. The CUDA path has no GPU-name
allowlist, but fit depends on the model pair, context budget, kernels and VRAM.
miniVERL is independent from verl and does not claim distributed execution or
full algorithmic compatibility.

## Install and run the 60-second demo

```bash
python -m pip install "miniverl[train]"
miniverl doctor
miniverl demo --output runs/demo
miniverl inspect runs/demo
```

The demo is deterministic, needs no network or GPU, and performs a real toy
optimization in about 50 seconds on the measured laptop CPU. For inspection,
schemas and reports without the ML stack, use `pip install miniverl`. For CUDA
training, install the matching CUDA-enabled PyTorch wheel first, then install
`miniverl[train,cuda]`; the extra does not select a CUDA PyTorch build. See the
[single-GPU guide](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/single-gpu-guide.md).

## Three paths

| Path | Start with | Concrete artifact | Next |
| --- | --- | --- | --- |
| **Align** — compare SFT, DPO, KD and OPD only when the pilot evidence supports the cost | `miniverl pilot recipes/alignment_policy_conditioned_qwen.yaml` | `alignment-card.json` | [Alignment Lab](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/alignment-lab/alignment-lab-v1.md) |
| **Distill locally** — strict OPD, shared backbones and padded trajectory updates on one CUDA GPU | `miniverl train recipes/qwen_consumer_gpu_shared.yaml --dry-run` | `config.resolved.yaml` plus a revision-pinned PEFT adapter | [Bring your own GPU](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/single-gpu-guide.md) |
| **Scale out** — import a documented profile, convert Parquet, export a bundle and run bridge checks | `miniverl bridge doctor scaleout-bundle` | `provenance/compatibility-report.json` | [Verified verl bridge](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/verl-bridge.md) |

The bridge import is deliberately not generic YAML conversion. If dataset or
environment, teacher identity, objective, or schedule semantics are missing,
`import-verl` writes `import-report.json` and a non-executable
`imported.template.yaml` with `status: needs_user_input`. It never silently
substitutes calculator tasks or an unqualified same-base teacher.

## One measured alignment result

Alignment Lab v1 is a **saturated tool-policy case study**, not a broad safety
benchmark. The shared SFT checkpoint already achieved 100% policy compliance
and 100% retained tool utility in all three seeds. No continuation method
improved it; continued SFT and both OPD variants retained measured regressions.

| continuation | alignment | tool utility | teacher queries | GPU time |
| --- | ---: | ---: | ---: | ---: |
| continued SFT | 94.4% | 88.9% | — | 3.9 s |
| DPO | 100.0% | 100.0% | — | 8.6 s |
| offline soft distillation | 100.0% | 100.0% | 100.0% | 26.6 s |
| standard OPD | 98.6% | 97.2% | 100.0% | 76.7 s |
| verifier-gated OPD | 97.9% | 95.8% | 46.8% | 66.0 s |

![Alignment and utility deltas from the saturated SFT checkpoint; small marks are all three seeds and large marks are means](https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/v0.6.1/docs/alignment-lab/delta-from-sft.svg)

The two sandbox safety checks tied at zero while utility still regressed.
IFEval, XSTest, HarmBench and RewardBench were **not executed**. “Preference
win rate” is a deterministic Minipolicy paired outcome, not human preference.
Read the [study, seed-level values and limitations](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/alignment-lab/alignment-lab-v1.md).

## One measured systems result

On one RTX 4080 with Qwen3-0.6B and eight fixed SQLite trajectories, physical
batch 4 improved update throughput from 2.369 to 3.866 trajectories/s in the
dual-model runtime. The shared-backbone batch-4 cell used 2.227 GiB peak
reserved memory versus 3.035 GiB for dual model, while running 10.1% slower.
All 12 preregistered equivalence comparisons passed. These are one-workload,
one-machine measurements, not promises for other GPUs.

![Measured throughput and reserved VRAM for dual-model and shared-backbone runtime cells](https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/v0.6.1/docs/consumer-runtime-v1-pareto.svg)

[Consumer Runtime v1 methods and caveats](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/consumer-runtime-v1.md)

## Compatibility boundary

![Verified local runtime, portable artifact bundle and pinned upstream smoke; distributed verl execution remains untested](https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/v0.6.1/docs/verl-bridge-architecture.svg)

The bridge targets official verl `v0.8.0` at commit `7aed6b23` and uses the
term **miniVERL-defined compatibility Level 3**. That means a checksummed
standard-artifact bundle plus pinned upstream config-parse/model-data-load
smoke—not arbitrary verl YAML or a completed distributed job.

Current exported bundles are intentionally `launchable: false`: the base
snapshot is absent, the reward implementation fails closed, and required user
mappings remain placeholders. The generated entry point is therefore
`launch.template.sh`. Readiness is reported as separate facts for artifact
completeness, parse/load smoke, reward completeness, launchability,
distributed execution and algorithm-semantic parity. The target is a
PPO/reward scaffold, not an executable continuation of miniVERL OPD semantics.

## Detailed studies and preserved negative evidence

- [RecoveryBench v1](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/recoverybench/recoverybench-v1.md): frozen-student KD
  outperformed much slower fresh-state OPD on the preregistered primary view;
  the verifier gate remained `insufficient_evidence`.
- [Alignment Lab v1](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/alignment-lab/alignment-lab-v1.md): the starting SFT
  checkpoint was at the ceiling, so no positive OPD result is claimed.
- [Calculator benchmark](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/benchmarking.md): both negative controls completed
  normally and measured 0% strict success. They were not configuration
  failures. Because they used the historical ambiguous protocol-v1 prompt,
  their failure cannot be attributed solely to intrinsic teacher behavior.
- [Consumer Runtime v1](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/consumer-runtime-v1.md): padded update batches and
  shared adapters preserve the measured one-update objective within declared
  tolerances; rollout generation remains sequential.
- [Limitations](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/limitations.md), [math](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/math.md),
  [reproducibility](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/reproducibility.md) and
  [compatibility policy](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/docs/compatibility.md).

New runs establish tokenizer compatibility through structural identity. The
legacy behavioral fingerprint—token IDs for one fixed probe plus metadata—is
only a migration fallback for older artifacts and is not an identity proof.

## Scope

miniVERL supports one local CUDA process. It does not implement or wrap Ray,
FSDP, Megatron, PPO, GRPO or a distributed launcher. The public studies cover
small Qwen3 models, deterministic tool environments and one RTX 4080; they do
not establish cross-model, cross-task, cross-GPU or broad safety generality.

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev]"
pytest -q -m "not gpu and not network"
```

Apache-2.0 licensed. See [CONTRIBUTING.md](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/CONTRIBUTING.md) and
[SECURITY.md](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/SECURITY.md). Project records: [default GPU recipe](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/recipes/qwen_consumer_gpu_calc.yaml),
[frozen calculator JSON](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/benchmarks/results/gpu-calc-hard-equal-update-v2.json),
[changelog](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/CHANGELOG.md), [citation](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/CITATION.cff) and [license](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.6.1/LICENSE).
