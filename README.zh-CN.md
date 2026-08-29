<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — 在一张个人 GPU 上运行 verl 风格 OPD" width="880">
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

**在一张 NVIDIA GPU 上运行 verl 风格在线策略蒸馏，并能检查每一项配置映射、
teacher target 与训练产物。** miniVERL 把类型化 YAML 和结构化 Parquet prompt
编译成本地 actor rollout → teacher scoring → actor update 循环，再导出标准 PEFT、
Parquet 与配置产物，便于继续扩展。

PyPI `v0.10.1` 是稳定版；`main` 是开发版。

## 60 秒开始

先安装与本机匹配的 CUDA PyTorch，再运行：

```bash
python -m pip install "miniverl[train,cuda]"
miniverl data sample --format verl-parquet --out prompts.parquet
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 \
  --config builtin:qwen3-0.6b-1.7b-opd \
  --set 'data.train_files=["prompts.parquet"]' --out plan.json
miniverl run --profile verl-opd-v0.8-single-gpu-v1 \
  --plan plan.json --dry-run
```

这条路径直接使用 PyPI wheel。规划阶段无需加载模型权重；生成的 `plan.json` 会绑定
源配置、顺序 override、profile 版本与输入 Parquet 字节。准备好 CUDA GPU 后移除
`--dry-run`，即可运行固定版本的 Qwen3-0.6B actor 与 Qwen3-1.7B teacher recipe。

`[train,cuda]` extra 安装 ML 与量化依赖。CUDA PyTorch wheel 请通过
[PyTorch 安装器](https://pytorch.org/get-started/locally/)单独选择。
[单卡指南](docs/single-gpu-guide.md)介绍从 8 GiB 显卡开始的显存规划，并附有维护者实测的
RTX 4080 环境。

## 一次运行会得到什么

- **可审阅的 plan。** 每个接受的 verl 字段在加载权重前就有本地作用、分类与风险等级。
- **严格的 current-policy trajectory。** actor policy version、token span 与 teacher
  监督位置始终绑定在一起。
- **紧凑的 teacher target。** top-k target 与 sampled-k1 signal 写入带校验和、无 pickle
  的 cache。
- **可恢复训练。** 事务化 manifest、checkpoint 与 cache index 支持检查和精确续跑。
- **标准产物。** PEFT adapter、safetensors、结构化 Parquet、resolved config 与类型化
  provenance 可直接携带。

```bash
miniverl run --profile verl-opd-v0.8-single-gpu-v1 --plan plan.json \
  --output runs --run-id my-opd
miniverl inspect runs/my-opd/trajectories.jsonl
miniverl cache stats runs/my-opd/teacher-cache
miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout
miniverl bridge doctor scaleout --json
```

## 工作方式

<picture>
  <source media="(max-width: 640px)" srcset="docs/verl-local-runtime-mobile.svg">
  <img src="docs/verl-local-runtime.svg" alt="类型化 verl 风格配置和 Parquet prompt 被编译为一张 CUDA GPU 上顺序执行的 actor rollout、teacher scoring 与 actor update，并产生可检查的本地产物和固定版本的 scale-out bundle。">
</picture>

miniVERL 在一个普通进程中分阶段调度 actor、teacher 与可选 reference。actor 使用当前
adapter 生成，teacher 对实际访问的 token 位置评分，随后 actor 接收 padded token-mean
update。resident、swap 与 shared-backbone placement 会适配可用显存，同时保持各角色身份
清晰可追踪。

每个阶段都会先发布证据，再跨越下一边界：trajectory 携带逐 token provenance，teacher
target 带校验和，checkpoint 事务化发布，最终 adapter 通过标准 PEFT 重新加载。检查一次
运行时不必再从终端日志反推原始意图。

## 选择你的路径

| 目标 | 第一个命令 | 主要产物 | 下一步 |
| --- | --- | --- | --- |
| **在本地运行 OPD** | `miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml` | 不可变执行 plan | [OPD quickstart](docs/opd-quickstart.md) |
| **迁移 verl profile** | `miniverl compat check --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml` | 逐字段兼容报告 | [面向 verl 用户](docs/for-verl-users.md) |
| **适配你的显卡** | `miniverl plan --config verl-opd.yaml --probe` | 实测 placement plan | [硬件规划](docs/hardware-planning.md) |
| **准备 scale-out 交接** | `miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout` | PEFT + Parquet + config bundle | [Scale-out 契约](docs/verl-opd-scaleout.md) |

原生 recipe 系统还支持 SFT、DPO、offline KD，以及 calculator、JSON navigation、只读
SQLite 和自定义环境上的 tool-aware OPD。

## 熟悉的 verl 输入，本地执行

当前 profile 固定官方 verl `v0.8.0` commit
`7aed6b230776f963fa09509c10d9c3a767d1102c`，并保留熟悉的字段：

```yaml
actor_rollout_ref:
  model: {path: Qwen/Qwen3-0.6B}
  rollout: {name: vllm, n: 1}
distillation:
  teacher_models:
    teacher_model:
      model_path: Qwen/Qwen3-1.7B
      inference: {name: vllm}
```

编译器把分布式 resource intent 转换为顺序本地 Hugging Face 阶段，并在 plan 中记录这次
转换。目前有两个实测 profile：

| Profile | 目标 | Teacher target |
| --- | --- | --- |
| `verl-opd-v0.8-single-gpu-v1` | direct GKD `forward_kl_topk` | top-k token ID 与 log-probability |
| `verl-opd-v0.8-single-gpu-pg-k1-v1` | sampled `k1` + vanilla policy loss | sampled-token teacher log-probability |

使用 `miniverl profiles show`、`compat explain` 与 `compat check` 查看完整映射。
[兼容 profile](docs/profiles/index.md)说明 profile identity 如何跟随 plan、cache、checkpoint
与 export。另有两个仅完成一致性验证的 grouped profile，为 Parquet prompt 提供事务化
`n>1` 独立样本；它们不改变两个实测 `n=1` profile，也不引入 GRPO 语义。
另有一个仅完成一致性验证的 rewarded profile，加入确定性的 exact-answer reward 与显式
group advantage 组合；目前不对任务质量作结论。

## 实测系统证据

Rollout Runtime v2 已在 RTX 4080 上完成 24-cell 测量，覆盖 64/256/512-token
response、`n=1/4`、greedy 与 seeded sampling。Managed vLLM 0.28.0 达到
108.5–117.7 output tokens/s；在 256/512-token cell 中比 `hf_cached` 快
1.65–4.55×，peak total GPU memory 为 11.54 GiB，并通过 8 次 policy refresh 与
完整 teardown。它是实测的 direct-GKD engine；engine log-probability probe 超出数值
门槛，因此 PG-k1 继续使用 `hf_cached`。

同一份证据也给出了下一步优化目标：`hf_cached` 在多数 seeded 256/512-token cell 中
约为旧路径的 1.8–1.9×，没有达到预注册的 2× gate；greedy 路径通常比旧 oracle 更慢。
因此开发线保持 `0.11.0.dev0`，尚未发布 v0.11.0。每个 cell 与 artifact hash 见
[完整 runtime 结果](docs/benchmarks/rollout-runtime-v2.md)。

Qwen3-0.6B/1.7B developer workload 消费 **32 个不同 prompt**，response 上限为 64 token，
在 RTX 4080 上完成 **8 次 current-policy update**，**peak reserved VRAM 为 3.1914 GiB**。
稳态 rollout、teacher scoring 与 update 的中位耗时分别为 9.7200、0.4864 与 2.3260 秒。
匹配运行在第 4 次 update 后中断并续跑，最终 trajectory、adapter 与 optimizer tensor
字节一致。

第二套 SmolLM2-360M/1.7B workload 以 1.4961 GiB peak reserved VRAM 完成相同的
32-prompt、8-update 形状，并通过精确续跑、PEFT 重载与 scale-out 物化。完整配置、哈希和
分阶段测量见 [Qwen3](docs/verl-opd-reference-workload.md) 与
[SmolLM2](docs/smollm2-opd-workload.md) 系统记录。

其他 NVIDIA GPU 使用同一条 device-name-agnostic CUDA 路径。模型能否装下以及实际速度
取决于显存、上下文、量化、kernel 与软件版本；`miniverl doctor` 和 `plan --probe` 会把
这些机器相关选择显示出来。

## 研究记录

仓库以相同的 resolved config 与源哈希发布正面、混合和负面结果。Calculator study 中，
protocol-qualified teacher 避免了 collapse，但只追平 supervised continuation。
RecoveryBench 在限定的 SQLite 设置中没有观察到 fresh-state 优势。Alignment Lab 从饱和
SFT checkpoint 出发，发现了两个 sandbox safety check 没有捕捉到的 utility regression。
预注册 External Alignment Gate 则因所有候选都未通过 retained-utility 阈值，在 continuation
之前结束。

- [Calculator protocol study](docs/benchmarking.md)
- [RecoveryBench](docs/recoverybench/recoverybench-v1.md)
- [Alignment Lab](docs/alignment-lab/alignment-lab-v1.md)
- [External Alignment Gate](docs/alignment-external/alignment-external-v1.md)

这些报告回答的是各自限定的实验问题；[限制页面](docs/limitations.md)集中列出测量、架构、
安全与泛化边界。

## 适用范围

miniVERL 面向一个本地进程、一张 NVIDIA CUDA GPU 和上面的文档化 OPD profile。
Scale-out 支持会生成 portable bundle，并对固定版本的上游源码执行验证。
[兼容性政策](docs/compatibility.md)列出支持字段、profile 语义与 handoff readiness state；
[限制页面](docs/limitations.md)集中说明更广的执行和科学边界。miniVERL 是独立的
Apache-2.0 项目。

## 开发

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install -e ".[dev]"
pytest -q -m "not gpu and not network"
```

另见 [CONTRIBUTING.md](CONTRIBUTING.md)、[CHANGELOG.md](CHANGELOG.md)、
[CITATION.cff](CITATION.cff)、[复现指南](docs/reproducibility.md)、
[SECURITY.md](SECURITY.md) 与 [Apache-2.0 license](LICENSE)。
