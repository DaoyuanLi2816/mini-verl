<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — 单卡 LLM 后训练" width="880">
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
  <a href="https://daoyuanli2816.github.io/mini-verl/"><strong>稳定版文档</strong></a> ·
  <a href="https://daoyuanli2816.github.io/mini-verl/dev/">开发版文档</a> ·
  <a href="README.md">English</a>
</p>

**miniVERL 是一个本地、可检查的单卡对齐与蒸馏运行时。** 它运行原生
SFT、DPO、KD 与严格 OPD recipe，保留仅 assistant token 的 loss mask 和
policy-version 来源，并通过 fail-closed 桥接与一个锁定的 verl profile 交换
标准 HF/PEFT/Parquet 产物。

PyPI `v0.7.1` 是稳定版；`main` 是开发版。miniVERL 独立于 verl，不声称可
执行任意 verl YAML、分布式任务或具备完整算法兼容性。

## 大约一分钟完成安装与验证

```bash
python -m pip install "miniverl[train]"
miniverl doctor
miniverl demo --fast --output runs/quickstart
miniverl inspect runs/quickstart/trajectories.jsonl
miniverl evidence validate alignment-external-v1
```

确定性 demo 不下载模型，会生成类型化 trajectory、带校验和的 teacher cache、
manifest 与报告。证据命令读取 wheel 自带数据，无需 Git checkout。若只需 schema
与检查功能，可仅安装 `miniverl`。

## 支持的硬件与运行边界

miniVERL 在 CPU 或一张 NVIDIA CUDA GPU 上运行单个本地进程。CUDA 路径不按
显卡名称设限，但能否装下取决于模型组合、上下文、kernel 与显存。请先安装匹配
本机 CUDA 的 PyTorch wheel，再安装 `miniverl[train,cuda]`；该 extra 本身不会
选择 CUDA PyTorch。Ray、FSDP、Megatron、PPO、GRPO 与分布式启动不属于当前
运行时。参见[单卡指南](docs/single-gpu-guide.md)。

## verl 兼容性摘要

桥接锁定官方 verl `v0.8.0`、commit `7aed6b23`。已验证的边界是带校验和的标准
产物以及锁定版本的配置解析、模型/数据加载冒烟测试，不包括原生 checkpoint
等价或已完成的 verl 作业。若数据集、环境、教师、目标或 schedule 语义未解析，
导入会 fail closed；它不会替换成 calculator task 或虚构教师。

当前导出仍为 `launchable: false`：缺少 base snapshot，reward scaffold 会失败关闭，
且必要映射仍是占位符。入口名为 `launch.template.sh`；readiness、parse/load 证据、
launchability、分布式执行与语义等价分别报告。参见[桥接契约](docs/verl-bridge.md)。

## 一项系统实测结果

在一张 RTX 4080、Qwen3-0.6B 与 8 条固定 SQLite trajectory 上，physical batch 4
将 dual-model update throughput 从 2.369 提高到 3.866 trajectories/s。
shared-backbone batch 4 的 peak reserved memory 为 2.227 GiB，dual model 为
3.035 GiB，但前者慢 10.1%。12 项预注册等价比较全部通过。这只是单机单 workload
数据，不是对其他 GPU 的承诺。

![dual-model 与 shared-backbone runtime 的实测吞吐和 reserved VRAM](docs/consumer-runtime-v1-pareto.svg)

[Consumer Runtime v1 方法与限制](docs/consumer-runtime-v1.md)

## 三条使用路径

| 路径 | 起点 | 真实产物 | 下一步 |
| --- | --- | --- | --- |
| **Align** — 仅在 pilot 证据支持成本时使用 SFT、DPO、KD 或 OPD | `miniverl pilot recipes/alignment_policy_conditioned_qwen.yaml` | `alignment-card.json` | [Alignment Lab](docs/alignment-lab/alignment-lab-v1.md) |
| **本地蒸馏** — 在一张 CUDA GPU 上运行严格 OPD、共享 backbone 与 padded update | `miniverl train recipes/qwen_consumer_gpu_shared.yaml --dry-run` | resolved config 与锁定 revision 的 PEFT adapter | [使用自己的 GPU](docs/single-gpu-guide.md) |
| **Scale out** — 转换 Parquet、导出标准产物并检查不支持边界 | `miniverl bridge doctor scaleout-bundle` | `provenance/compatibility-report.json` | [产物桥接](docs/verl-bridge.md) |

## 研究记录与保留的负结果

### v0.7 External Alignment Gate

这项预注册外部研究在教师或方法训练前停止。两个已声明的起始策略 lineage 中，
所有候选的 retained JSONNav utility 都是 **0/64**，未改动下限为 20%。

| 选中 checkpoint | 合格教师 | continuation arm | 已访问 final-test task |
| ---: | ---: | ---: | ---: |
| **0** | **0** | **0** | **0** |

```bash
miniverl pilot --builtin-study alignment-external-v1 --json
```

结果是 `do_not_continue_this_study` 与 `insufficient_evidence`，不是 SFT/DPO/KD/OPD
之间的推荐。Granite Guardian 数值仅为未资格认证的 selection diagnostic；Granite、
PairRM、教师资格认证和保留 final test 均未运行。参见[研究与限制](docs/alignment-external/alignment-external-v1.md)。

### 更早的对齐案例研究

Alignment Lab v1 的起始 SFT checkpoint 在三个 seed 中已经达到 100% policy
compliance 与 100% retained tool utility；没有 continuation 超过这个天花板，
continued SFT 和两个 OPD arm 保留了实测退化。两个 sandbox safety check 同为零，
同时 utility 仍然退化。IFEval、XSTest、HarmBench、RewardBench 未执行；
“preference win rate”是确定性的 Minipolicy 配对结果，不是人类偏好。
[逐 seed 证据](docs/alignment-lab/alignment-lab-v1.md)。

- [RecoveryBench v1](docs/recoverybench/recoverybench-v1.md)：在预注册主视图中，
  frozen-student KD 优于更慢的 fresh-state OPD；verifier gate 仍为
  `insufficient_evidence`。
- [Calculator benchmark](docs/benchmarking.md)：两个负对照都正常完成并得到 0%；
  历史 protocol-v1 prompt 存在歧义，因此不能把失败仅归因于教师内在行为。
- [限制](docs/limitations.md)、[数学](docs/math.md)、[复现](docs/reproducibility.md)
  与[兼容性政策](docs/compatibility.md)。

新运行以结构身份确认 tokenizer 兼容性；旧 behavioral fingerprint 只用于迁移，
不是身份凭证。

## 开发

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev]"
pytest -q -m "not gpu and not network"
```

Apache-2.0 许可。参见 [CONTRIBUTING.md](CONTRIBUTING.md)、
[SECURITY.md](SECURITY.md)、[changelog](CHANGELOG.md) 与 [citation](CITATION.cff)。
