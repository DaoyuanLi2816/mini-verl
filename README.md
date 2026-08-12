<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — single-GPU LLM post-training" width="880">
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

**Run a documented subset of verl-style on-policy distillation on one consumer
GPU.** miniVERL accepts a typed verl v0.8 OPD profile and Parquet prompts,
executes actor rollout → teacher scoring → actor update locally, and exports
standard PEFT/Parquet/config artifacts for scale-out. Native SFT, DPO, KD and
tool-agent recipes remain available.

PyPI `v0.8.0` is stable; `main` is development. miniVERL is independent from
verl. It does not claim arbitrary verl YAML execution, distributed execution,
or full algorithmic compatibility.

## Pip-only OPD quickstart

```bash
python -m pip install "miniverl[train]"
miniverl data sample --format verl-parquet --out prompts.parquet
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config builtin:qwen3-0.6b-1.7b-opd \
  --set 'data.train_files=["prompts.parquet"]'
miniverl run --profile verl-opd-v0.8-single-gpu-v1 --config builtin:qwen3-0.6b-1.7b-opd \
  --set 'data.train_files=["prompts.parquet"]' --dry-run
```

The sample, plan and dry run need no Git checkout; planning loads no weights.
Remove `--dry-run` on one CUDA GPU to execute the pinned Qwen3-0.6B/1.7B NF4
recipe and produce a loadable PEFT adapter. [Follow the OPD quickstart](docs/opd-quickstart.md).

## Supported hardware and runtime boundary

miniVERL runs one local process on CPU or one NVIDIA CUDA GPU. The CUDA path is
device-name agnostic, but fit depends on model pair, context, kernels and VRAM.
Install the matching CUDA-enabled PyTorch build first, then
`miniverl[train,cuda]`; that extra does not select a CUDA PyTorch wheel.
Ray, FSDP, Megatron, PPO, GRPO and distributed launch are outside the runtime.
See the [single-GPU guide](docs/single-gpu-guide.md).

## verl compatibility summary

The executable profile targets official verl `v0.8.0` at commit `7aed6b23` and
supports one actor, one teacher, `n=1`, pure GKD `forward_kl_topk`, token-mean
aggregation, LoRA/QLoRA and no reward/KL penalty. PG OPD, task-reward mixtures,
multi-teacher, multimodal and distributed fields fail closed.

Compatible OPD exports contain no reward scaffold. They preserve student and
teacher identities, Parquet bytes and OPD overrides, but remain
`launchable: false` until exact base snapshots are materialized. Parse status,
artifact loadability, launchability and distributed execution are separate.
[Read the bridge contract](docs/verl-bridge.md).

## Measured RTX 4080 runtime

The packaged Qwen3-0.6B/1.7B recipe completed two 16-token rollouts and one OPD
update with **3.1758 GiB peak reserved VRAM**; the first update completed in
**12.0224 s** and the standard PEFT adapter reloaded successfully. This proves
one runtime/artifact path only—no alignment-quality endpoint or method
comparison ran. [Exact recipe, timings and hashes](docs/opd-quickstart.md).

## Three paths

| Path | Start with | Concrete artifact | Next |
| --- | --- | --- | --- |
| **Run OPD locally** | `miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml` | compiled plan, trajectories, targets and PEFT adapter | [Plan and run](docs/opd-quickstart.md) |
| **Bring a verl config** | `miniverl import-verl --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml --out local-opd.yaml` | field report plus round-trippable profile | [Compatibility](docs/compatibility.md) |
| **Move data and artifacts** | `miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout` | Parquet + PEFT + OPD override bundle | [Bridge contract](docs/verl-bridge.md) |

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
reserved final test did not run. [Study and limitations](docs/alignment-external/alignment-external-v1.md).

### Earlier measured alignment case study

Alignment Lab v1 began from an SFT checkpoint already at 100% policy compliance
and 100% retained tool utility in all three seeds. No continuation improved the
ceiling; continued SFT and both OPD variants retained measured regressions.
The two sandbox safety checks tied at zero while utility still regressed.
IFEval, XSTest, HarmBench and RewardBench were not executed, and “preference
win rate” is a deterministic Minipolicy paired outcome, not human preference.
[Seed-level evidence](docs/alignment-lab/alignment-lab-v1.md).

- [RecoveryBench v1](docs/recoverybench/recoverybench-v1.md): frozen-student KD
  beat slower fresh-state OPD on the preregistered primary view; the verifier
  gate remained `insufficient_evidence`.
- [Calculator benchmark](docs/benchmarking.md): both negative controls completed
  normally at 0%; the ambiguous historical protocol-v1 prompt prevents
  attributing failure solely to intrinsic teacher behavior.
- [Limitations](docs/limitations.md), [math](docs/math.md),
  [reproducibility](docs/reproducibility.md) and
  [compatibility policy](docs/compatibility.md).

New runs establish tokenizer compatibility through structural identity. The
legacy behavioral fingerprint is only a migration fallback, not identity proof.

## Develop

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev]"
pytest -q -m "not gpu and not network"
```

Apache-2.0 licensed. See [CONTRIBUTING.md](CONTRIBUTING.md),
[SECURITY.md](SECURITY.md), the [changelog](CHANGELOG.md) and
[citation](CITATION.cff). Project records: [default GPU recipe](recipes/qwen_consumer_gpu_calc.yaml),
[frozen calculator result](benchmarks/results/gpu-calc-hard-equal-update-v2.json)
and [license](LICENSE).
