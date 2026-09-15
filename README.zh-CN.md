<p align="center">
  <picture>
  <source media="(max-width: 760px)" srcset="docs/banner-mobile.svg">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — 单张消费级 GPU 上的 verl" width="880">
  </picture>
</p>

<div align="center">

[![CI](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml)
[![Build](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/miniverl.svg)](https://pypi.org/project/miniverl/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

</div>

<p align="center">
  <a href="https://pypi.org/project/miniverl/"><strong>PyPI</strong></a> ·
  <a href="https://daoyuanli2816.github.io/mini-verl/"><strong>稳定版文档</strong></a> ·
  <a href="https://daoyuanli2816.github.io/mini-verl/dev/">开发版文档</a> ·
  <a href="README.md">English</a>
</p>

**单张消费级 GPU 上的 verl。**

在一张 NVIDIA 显卡上运行 PPO、GRPO 和在线策略蒸馏（OPD）。
沿用 verl 配置，本地训练、中断续跑，再把模型导出到下一步工作流。

PyPI `v0.16.0` 是稳定版；`main` 为开发版。

## 第一次本地实验

先通过 [PyTorch 安装向导](https://pytorch.org/get-started/locally/) 安装匹配的 CUDA 版本，然后运行：

```bash
python -m pip install "miniverl[train,hydra]"
miniverl data sample --reward-profile target-length --rows 8 --out data/rl-prompts.parquet
miniverl run --example hydra-ppo --bind reward.provider=target_length --dry-run
miniverl run --example hydra-ppo --bind reward.provider=target_length --run-id local-ppo
```

这个 Qwen3-0.6B 示例使用简单的长度奖励，可以在 `rewards.jsonl` 中直接查看。
[五分钟工作流](docs/for-verl-users.md) 还介绍了如何换成 GRPO、恢复训练和导出 adapter。

已经有 verl 实验？直接传入配置树和 Hydra overrides：

```bash
miniverl run --verl-config-path /path/to/verl/trainer/config \
  --verl-config-name ppo_trainer --dry-run \
  algorithm.adv_estimator=grpo actor_rollout_ref.rollout.n=8 trainer.n_gpus_per_node=8
```

[直接运行配置](docs/direct-verl-config.md) 介绍数据和奖励函数的绑定方式；
[单 GPU 指南](docs/single-gpu-guide.md) 帮你选择模型与显存设置。
安装 extras 分别补充训练、Hydra 或量化依赖，CUDA PyTorch 版本需要单独选择。

## 选择你的路径

| 你想做什么 | 从这里开始 | 得到什么 |
| --- | --- | --- |
| **用奖励训练** | [PPO / GRPO 工作流](docs/for-verl-users.md) | 训练后的 adapter、奖励日志和可恢复检查点 |
| **蒸馏教师模型** | [OPD 快速上手](docs/opd-quickstart.md) | 学生在自己生成的轨迹上接受教师反馈 |
| **迁移到更大的训练环境** | [导出与交接](docs/verl-opd-scaleout.md) | PEFT、Parquet、配置文件，以及待完成配置的报告 |

原生 recipe 还支持 SFT、DPO 和离线 KD。
[硬件规划](docs/hardware-planning.md) 帮你根据显存安排工作负载。

## 一次运行会得到什么

- **熟悉的配置。** 沿用 verl 字段和 overrides，查看各项配置如何映射到本地执行。
- **看得见的训练过程。** 检查生成轨迹、奖励、loss 和教师目标。
- **精确续训。** 从检查点恢复模型、optimizer、数据位置和随机状态。
- **可复用的产物。** 导出 adapter、数据集，以及对应的配置和来源信息。

## 工作方式

<picture>
  <source media="(max-width: 760px)" srcset="docs/verl-local-runtime-mobile.svg">
  <img src="docs/verl-local-runtime.svg" alt="单张 GPU 按阶段运行 rollout、奖励、reference、teacher 和更新，最后导出模型、数据和交接报告。">
</picture>

actor、critic、reference、reward 和 teacher 按阶段共用 GPU。
PPO 分别训练 actor 和 critic；GRPO 使用分组奖励；OPD 在学生当前策略生成的轨迹上学习教师目标。

## 实测结果

在 RTX 4080 上，Qwen3-0.6B/1.7B OPD 工作负载用 32 条不同 prompt 完成了八次更新，
峰值 reserved 显存为 **3.1914 GiB**。第四次更新后中断并恢复，生成轨迹、adapter
和 optimizer 张量均与连续运行逐字节一致。SmolLM2-360M/1.7B 工作负载使用 **1.4961 GiB**。
[工作负载与测量记录](docs/verl-opd-reference-workload.md) ·
[24 组 rollout 后端测量](docs/benchmarks/rollout-runtime-v2.md)。

任务效果见[计算器实验](docs/benchmarking.md)、
[RecoveryBench](docs/recoverybench/recoverybench-v1.md)、
[Alignment Lab](docs/alignment-lab/alignment-lab-v1.md) 和
[External Alignment Gate](docs/alignment-external/alignment-external-v1.md)。
其中的负面和混合结果、原始数据与研究范围均完整保留。

## 项目与兼容性

miniVERL 是面向单张 NVIDIA CUDA GPU 的独立 Apache-2.0 项目，
支持文档列出的 verl 配置 profile。能容纳多大的模型取决于显存与工作负载。
[兼容性矩阵](docs/compatibility.md) 列出上游版本和配置支持情况；
[限制说明](docs/limitations.md) 集中介绍硬件验证范围、科学结论和分布式执行边界。

## 开发

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev,train]"
pytest -q -m "not gpu and not network"
```

[参与贡献](CONTRIBUTING.md) · [更新记录](CHANGELOG.md) ·
[引用](CITATION.cff) · [安全](SECURITY.md) · [许可证](LICENSE)
