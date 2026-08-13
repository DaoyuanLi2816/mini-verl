<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — 在单张个人 GPU 上运行 verl 风格 OPD" width="880">
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

**在一张个人 GPU 上运行边界明确的 verl 风格在线策略蒸馏。** miniVERL 读取类型化的
verl 风格 YAML 与 Parquet prompt，在本地依次执行 actor rollout → teacher scoring →
actor update，记录每个本地语义重解释，并导出标准 PEFT、Parquet 与配置产物供固定版本
的 verl 接手扩展。

PyPI `v0.8.1` 是稳定版；`main` 是开发版。miniVERL 是独立项目，不代表上游背书；它
不执行任意 verl YAML、不启动分布式任务，也不声称完整的算法兼容性。

## 仅用 pip 的快速开始

请先安装与本机 CUDA 匹配的 PyTorch wheel，然后运行：

```bash
python -m pip install "miniverl[train]"
miniverl data sample --format verl-parquet --out prompts.parquet
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 \
  --config builtin:qwen3-0.6b-1.7b-opd \
  --set 'data.train_files=["prompts.parquet"]' --out plan.json
miniverl run --profile verl-opd-v0.8-single-gpu-v1 \
  --plan plan.json --dry-run
```

这些命令不需要 Git checkout。`data sample` 生成真实的结构化 Parquet；`plan` 在不加载
权重时编译完整字段矩阵；`run --dry-run` 验证本地执行契约。在一张 NVIDIA CUDA GPU
上移除 `--dry-run`，即可运行固定版本的 Qwen3-0.6B actor 与 Qwen3-1.7B teacher，并
得到可检查、可重新加载的 PEFT adapter。

`train` extra 安装训练运行时，但不会选择正确的 CUDA PyTorch；`cuda` extra 只额外安装
bitsandbytes。真实运行前请阅读[单卡安装与显存指南](docs/single-gpu-guide.md)。

## 架构

<picture>
  <source media="(max-width: 640px)" srcset="docs/verl-local-runtime-mobile.svg">
  <img src="docs/verl-local-runtime.svg" alt="verl 风格 YAML、override 与 Parquet prompt 经过类型化编译器；一张 CUDA GPU 依次执行 actor rollout、teacher scoring 与 actor update；可检查产物交给固定版本 verl，而分布式执行不属于 miniVERL。">
</picture>

miniVERL 在一个普通进程里按阶段调度模型角色。它不模拟 Ray resource pool，也不把本地
Hugging Face generation 伪装成 vLLM。兼容报告保留源字段、解释本地含义、标注风险，
并在字段改变算法或分布式语义时 fail closed。

原生 SFT、DPO、offline KD 与工具型 OPD recipe 仍然保留。它们共享严格 token provenance、
无 pickle teacher cache、事务化 checkpoint 与 adapter 导出；但对于已有 verl 经验的用户，
类型化 verl profile 是最短入口。

### 单一本地 scheduler，角色始终明确

actor 使用当前 adapter 生成；teacher 只对 actor 实际访问的 token 位置评分；随后 actor 接收
padded、token-mean update。teacher 不会被当作 reward model，tool output 不会成为训练标签，
过期 actor-policy version 也不能进入 on-policy batch。显存规划可选择 resident、swap 或兼容
的 shared-backbone，同时始终区分 actor、teacher 与 reference 身份。

每个阶段都在跨越下一边界前写入证据：trajectory 带逐 token provenance，top-k teacher
target 以无 pickle 格式校验，checkpoint 事务化发布，最终 adapter 用标准 PEFT loader 验证。
因此崩溃后的检查与恢复不需要依靠终端文本猜测原始意图。

## 从 verl 迁移？

| verl 中的动作 | miniVERL 对应方式 |
| --- | --- |
| 传入 Hydra 风格 override | 重复使用 `--set`；v0.9 开发版也接受 `--overrides-file` 与 `--` 后的参数 |
| 检查 resolved config | `miniverl plan --json` |
| 执行纯 OPD | `miniverl run` |
| 复用 prompt Parquet | 直接设置 `data.train_files` |
| 分配 rollout/teacher worker | 编译为单卡顺序阶段 |
| 保存 FSDP/Megatron checkpoint | 不支持 |
| 准备扩展交接 | `miniverl export-verl` |

```bash
actor_rollout_ref.model.path=Qwen/Qwen3-0.6B
distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-1.7B

miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml \
  --set actor_rollout_ref.actor.optim.lr=1e-5
```

外部 YAML 在 `run` 前必须显式接受 `plan` 列出的高风险本地重解释；内置
profile 使用与具体值绑定的审核清单。详见[覆盖优先级与安全输入格式](docs/config-overrides.md)，
miniVERL 不执行 Hydra interpolation 或 shell 文本。

公开内置 profile 刻意使用上游形状的 `name: vllm`。miniVERL 会把 rollout 与 teacher
engine 名分类为本地重解释，再通过顺序本地 HF 阶段执行；这不等于 vLLM 语义。完整
config、data、role 与错误映射见[面向 verl 用户的指南](docs/for-verl-users.md)。

## 已测试的 profile 边界

`verl-opd-v0.8-single-gpu-v1` 固定官方 verl `v0.8.0` 与 commit
`7aed6b230776f963fa09509c10d9c3a767d1102c`。可执行边界有意保持狭窄：

- 一个可训练 actor 与一个 teacher；
- 每个 prompt 只生成一个 response（`n=1`）；
- 无 reward 的 generalized knowledge distillation；
- `forward_kl_topk`、top-k + tail target 与 token-mean 聚合；
- 一张 CUDA GPU 上的 LoRA/QLoRA adapter 更新；
- 不可变模型 revision 与 verl 风格结构化 prompt Parquet。

PG OPD、task reward、KL penalty、多 teacher、多模态、PPO、GRPO、critic、Ray、FSDP、
Megatron、多 GPU 与多节点均不支持。已知不支持值会得到机器可读分类；未知字段和未解析
的 `${...}` 会被拒绝。输入必须是 resolved profile 子集，而不是任意启动脚本。

## RTX 4080 实测路径

Qwen3-0.6B/1.7B developer workload 实际消费 **32 个不同 prompt**，response 上限为
64 token，完成 **8 次 current-policy update**，**peak reserved VRAM 为 3.1914 GiB**。
稳态 rollout、teacher scoring 与 update 的中位耗时分别为 9.7200、0.4864 与 2.3260 秒。
匹配的运行在第 4 次 update 后中断并恢复；最终 trajectory、adapter 与 optimizer tensor
均字节一致。详见[数据绑定图与完整记录](docs/verl-opd-reference-workload.md)；原始一次更新的
[pip smoke](docs/opd-quickstart.md)仍完整保留。
另一个固定版本的 SmolLM2-360M/1.7B 兼容性 smoke 完成了一次完整的
rollout/scoring/update 循环；它不是第二套完整实测 recipe。

这只证明一个运行时与产物路径，不是吞吐 benchmark、对齐质量 endpoint，也不证明 OPD
优于 SFT、DPO 或 KD。其他 NVIDIA GPU 使用相同的 device-name-agnostic CUDA 路径，但
能否装下仍取决于显存、上下文、量化与 kernel。

### 按硬件条件选择路径，而不是按显卡名称

| 情况 | 建议起点 | 保持不变的内容 |
| --- | --- | --- |
| CPU 或笔记本上先检查 | `plan` 与 `run --dry-run` | 完整配置分类 |
| 单张小显存 CUDA GPU | QLoRA 与角色 swap | 逻辑 batch 与 loss 语义 |
| 同 base 的 actor/teacher adapter | shared-backbone | 明确角色 provenance |
| 显存更充足 | 增大各阶段 physical batch | 数据与 optimizer 意图 |

BF16/FP16 自动选择取决于设备能力，而不是 3070、4080、5090 或 Titan 等市场名称。
`miniverl doctor` 报告实际 CUDA/PyTorch 路径；普通 plan 不加载权重，显式
`plan --probe` 才执行零 optimizer update 的有界 CUDA 测量。详见
[硬件规划](docs/hardware-planning.md)。
显存紧张时不会静默更换模型、teacher、context、top-k 或 loss。

## 数据与产物互操作

profile 直接读取结构化 verl 风格 Parquet，不会在缺失数据时静默换成 calculator 环境。
prompt 的 role/content 保持结构化，data source、ability 与 extra metadata 也会保留。跨越
原生 trajectory 边界时可用 `miniverl convert-dataset`；除非明确允许部分转换，任何有损行
都会被拒绝。

本地 run 包含源配置、兼容矩阵、本地执行计划、trajectory、teacher target、checkpoint、
实测记录与 PEFT adapter。导出前可执行：

```bash
miniverl inspect runs/my-opd/trajectories.jsonl
miniverl cache stats runs/my-opd/teacher-cache
miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout
miniverl bridge materialize scaleout --download --offline
miniverl bridge doctor scaleout --json
```

v0.8.1 export 保留 student/teacher 身份、Parquet 原始字节与纯 OPD override；在精确 base
snapshot 被物化并通过已安装的固定 verl commit 验证前，仍报告 `launchable: false`。只有此后
`bridge materialize` 才会发布带校验和的 `launch.sh`；分布式执行仍是未测试。详见
[物化契约](docs/scaleout-materialization.md)、[桥接契约](docs/verl-bridge.md)与
[兼容性政策](docs/compatibility.md)。

建议操作闭环是 **plan → inspect → run → inspect → export**。`plan --out` 把 YAML、按顺序
应用的 override、扫描后的 Parquet 与精确 native config 绑定；`run --plan` 在加载权重前拒绝
任何漂移。plan digest 同时进入 run manifest、teacher cache 和 checkpoint；直接
`run --config` 仍适合快速实验。详见[不可变执行计划](docs/immutable-plans.md)。

## 研究与验证

miniVERL 将所有实测研究——包括负结果、被取代的运行与预注册 early stop——继续公开在
文档中；它们都不会被用来宣称 OPD 普遍优于 SFT、DPO 或 KD。参见
[v0.7 External Alignment Gate](docs/alignment-external/alignment-external-v1.md)、
[Alignment Lab](docs/alignment-lab/alignment-lab-v1.md)、
[RecoveryBench](docs/recoverybench/recoverybench-v1.md)与
[calculator study](docs/benchmarking.md)。

新运行通过结构身份确认 tokenizer 兼容性；旧 behavioral fingerprint 只用于迁移，不是身份
证明。科学限制与不可变源哈希继续保留在详细报告和[限制页面](docs/limitations.md)。

## 开发、安全与许可

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev]"
pytest -q -m "not gpu and not network"
```

贡献应保持单卡边界清晰，并为新失败模式添加测试。安全问题请按 [SECURITY.md](SECURITY.md)
私下报告。另见 [CONTRIBUTING.md](CONTRIBUTING.md)、[changelog](CHANGELOG.md)、
[citation](CITATION.cff)、[复现指南](docs/reproducibility.md)与 [Apache-2.0 license](LICENSE)。
