<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — 把 verl 实验语义转换为单张 CUDA GPU 上的执行计划" width="880">
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

**miniVERL 在一张 NVIDIA GPU 上运行经过验证的 verl 实验语义子集。** 输入 resolved
verl 风格配置与 Parquet prompt，版本化编译器会生成可审阅的本地计划；actor、reference、
teacher 与 reward 角色按阶段执行，最终发布可携带的 PEFT 与数据产物。

当前开发线针对官方 verl `v0.9.0`（`483b8a00`）加入 critic-free RL：GRPO、
Dr.GRPO、RLOO、REINFORCE++、grouped rollout、task reward 与固定 reference-policy
KL。已有 verl `v0.8.0` OPD profile 继续提供 direct GKD 与 sampled-k1 蒸馏。

PyPI `v0.12.0` 是稳定版；`main` 是开发版。

## 60 秒开始

先安装与本机匹配的 CUDA PyTorch，再运行：

```bash
python -m pip install "miniverl[train,cuda]"
miniverl data sample --task-rewards --rows 8 --out data/rl-prompts.parquet
miniverl import-verl --profile verl-rl-v0.9-single-gpu-v1 \
  --config examples/verl-rl-v0.9-single-gpu.yaml --out local-grpo.yaml
miniverl validate local-grpo.yaml
miniverl train local-grpo.yaml --dry-run
```

Importer 会在原生 recipe 旁写入 `local-grpo.import-report.json`，逐项说明源字段及其
本地效果，并记录分布式资源设置如何转换为一个进程、一张 GPU。移除 `--dry-run` 后，
示例会加载固定 revision 的 Qwen3-0.6B actor 并执行两次 rollout iteration。

`[train,cuda]` extra 安装训练与量化依赖；CUDA PyTorch build 仍需通过
[PyTorch 安装器](https://pytorch.org/get-started/locally/)单独选择。
[单卡指南](docs/single-gpu-guide.md)包含显存规划与维护者实测的 RTX 4080 环境。

## 一次运行会得到什么

- **逐字段编译报告。** 实验字段保留语义，物理分布字段得到明确的单卡 lowering。
- **与策略绑定的 trajectory。** group/sample 身份、模型生成 token span、behavior
  log-probability 与 policy version 始终一起保存。
- **可审计的 reward 与 objective。** reward component、group advantage、reference
  KL、clip fraction、entropy 与 update metric 都是结构化数据。
- **精确恢复。** 事务化 manifest 与 checkpoint 在下一次 rollout 前恢复 policy、
  optimizer、cursor、RNG 与 reward identity。
- **可携带产物。** PEFT adapter、safetensors、Parquet、resolved config 与类型化
  provenance 可以进入更大的工作流。

## 工作方式

<picture>
  <source media="(max-width: 640px)" srcset="docs/verl-local-runtime-mobile.svg">
  <img src="docs/verl-local-runtime.svg" alt="Resolved verl 配置被编译为经过验证的单卡执行计划；actor、reference、teacher 与 reward 角色分阶段执行，并生成可携带产物与 readiness report。">
</picture>

miniVERL 把一张 GPU 当作逻辑角色的时间调度器。Critic-free RL 先生成完整 prompt
group，计算 outcome reward 和固定版本的 advantage estimator，按需运行冻结的 reference
adapter，最后更新 actor。OPD 复用相同的 trajectory 与 checkpoint 基础设施，并以
teacher-scoring 阶段提供训练信号。

## 选择你的路径

| 目标 | 第一个命令 | 主要产物 | 下一步 |
| --- | --- | --- | --- |
| **本地 RL** | `miniverl import-verl --profile verl-rl-v0.9-single-gpu-v1 --config verl-rl.yaml --out local.yaml` | 原生 recipe + 兼容报告 | [RL quickstart](docs/verl-rl-runtime.md) |
| **本地 OPD** | `miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml --out plan.json` | 不可变执行计划 | [OPD quickstart](docs/opd-quickstart.md) |
| **适配显卡** | `miniverl plan --config verl-opd.yaml --probe` | 实测 placement plan | [硬件规划](docs/hardware-planning.md) |
| **交接产物** | `miniverl export-verl --run runs/my-run --target-verl v0.8.0 --out scaleout` | PEFT + Parquet + config bundle | [Scale-out 契约](docs/verl-opd-scaleout.md) |

原生 recipe 还支持 SFT、DPO、offline KD，以及 calculator、JSON navigation、只读
SQLite 和自定义 tool environment。

## 当前能力矩阵

| 实验能力 | 本地状态 | 契约 |
| --- | --- | --- |
| GRPO / Dr.GRPO | 语义一致 | verl v0.9 group statistic 与 vanilla clipped policy loss |
| RLOO / REINFORCE++ | 语义一致 | verl v0.9 advantage 与 masking 规则 |
| Grouped `n > 1` rollout | 已支持 | 完整 group、稳定 sample seed 与 behavior-policy identity |
| Task reward | 已支持 | exact-answer/target-length 内置奖励、environment verifier 或可信 Python API |
| 固定 reference KL | 本地 lowering | 与 actor 共用 backbone 的冻结 adapter 角色 |
| Direct GKD / sampled-k1 OPD | 已支持 | 固定 verl v0.8 profile 与 teacher target |
| PPO / GAE 执行 | 尚未实现 | 数学原语已做一致性验证；还没有可训练 critic 生命周期 |
| Ray、FSDP/FSDP2、Megatron、TP/PP/DP > 1 | 仅分布式 | 这些能力改变物理规模，不改变本地 objective |

生成的 [v0.9 兼容记录](docs/generated/verl-rl-v0.9-compatibility.json)绑定示例中每个
字段的源值、本地目标、分类与编译规则哈希。`miniverl import-verl` 接收 documented
resolved subset；未知且会改变算法的字段会被拒绝，dataset 与 reward 实现不会被替换。

## 实测系统证据

已发布的 Qwen3-0.6B/1.7B OPD workload 在 RTX 4080 上消费 32 个不同 prompt，完成
8 次 current-policy update，peak reserved VRAM 为 3.1914 GiB。第 4 次 update 后的
匹配中断/续跑复现了字节一致的 trajectory、adapter 与 optimizer tensor。配套
SmolLM2-360M/1.7B workload 以 1.4961 GiB 完成同一形状；完整配置、哈希和阶段耗时见
[系统记录](docs/verl-opd-reference-workload.md)。

Rollout Runtime v2 在 RTX 4080 上测量了 `hf_cached` 与 managed vLLM 的 24 个 cell，
覆盖不同 response length、sampling mode 与 `n=1/4`。详见
[runtime 报告](docs/benchmarks/rollout-runtime-v2.md)。新 RL 家族的证据由 exact-wheel
release qualification 发布，不作为任务质量对比。

## 研究记录

科学报告保留原始结论与 frozen 输入：
[calculator protocol](docs/benchmarking.md)、
[RecoveryBench](docs/recoverybench/recoverybench-v1.md)、
[Alignment Lab](docs/alignment-lab/alignment-lab-v1.md) 与
[External Alignment Gate](docs/alignment-external/alignment-external-v1.md)。这些是范围明确的
研究记录，与 runtime qualification 分开。

## 兼容边界

miniVERL 面向一个本地进程、一张 NVIDIA CUDA GPU。编译器分别标记精确或语义一致、
本地物理 lowering、仅分布式、概念上可行但尚未实现，以及因具体原因不支持的输入。
[兼容性政策](docs/compatibility.md)给出完整矩阵；[限制页面](docs/limitations.md)集中说明
架构、测量、安全与泛化边界。miniVERL 是独立的 Apache-2.0 项目。

## 开发

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev,train]"
pytest -q -m "not gpu and not network"
```

另见 [CONTRIBUTING.md](CONTRIBUTING.md)、[CHANGELOG.md](CHANGELOG.md)、
[CITATION.cff](CITATION.cff)、[SECURITY.md](SECURITY.md) 与
[Apache-2.0 license](LICENSE)。
