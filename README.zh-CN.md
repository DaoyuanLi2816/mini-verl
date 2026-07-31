<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — 单卡上的工具调用智能体在线策略蒸馏" width="880">
</p>

<div align="center">

[![CI](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml)
[![Build](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/miniverl.svg)](https://pypi.org/project/miniverl/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

</div>

<p align="center">
  <a href="https://pypi.org/project/miniverl/"><strong>PyPI 软件包</strong></a> ·
  <a href="#个人单卡快速上手">安装与训练</a> ·
  <a href="docs/single-gpu-guide.md">适配你的 GPU</a> ·
  <a href="#实测结果协议对齐的-opd-追平-sft">实测结果</a>
</p>

> 本文是 [README.md](README.md) 的中文翻译。英文版为准；若两者不一致，请以英文版为准并提交 issue。

**面向个人单卡、紧凑且可审计的工具调用智能体训练栈。**

PyPI `v0.2.5` 是稳定发布版；`main` 是开发分支，可能领先于稳定版。

miniVERL 是一个紧凑、可审计的训练实验室，让小型语言模型从**它自己生成的
多轮工具轨迹**中学习。它会真实执行工具、显式记录 token 来源，并且只在正确
的位置使用教师分布目标——不需要 Ray 或 GPU 集群。代码没有显卡型号白名单：
使用你现有的 NVIDIA CUDA 显卡，再选择能装进显存的模型组合与 token 预算。

```bash
python -m pip install miniverl            # 轻量核心层
miniverl doctor
python -m pip install "miniverl[train]"   # 添加本地训练依赖
miniverl demo --output runs/demo          # 无需联网、无需 GPU，笔记本 CPU 上约 50 秒
```

基础安装是不含 torch 的核心层（`doctor`、`validate`、`inspect`、`report`、
schema 与 Python API）。`train` extra 会添加 torch、Transformers 与 PEFT，
因为 `demo` 会执行真实优化。
这种拆分是有意的：`pip install miniverl` 可以在不下载数 GB 机器学习依赖的
情况下检查和验证产物；要进行训练或评估，请安装
`pip install "miniverl[train]"`。

**它让三件事可以被检查**

- **策略真实：** 每个 OPD 批次都来自它要更新的策略版本；过期教师目标会被拒绝。
- **Token 真实：** 工具输出只作为上下文，只有带类型的 assistant span 能进入 loss。
- **预算真实：** 精确全词表目标与压缩的 `top-k + tail` 目标分开命名、分开报告。

[运行本地 demo](#本地玩具演示) ·
[在你的 GPU 上训练](#个人单卡快速上手) ·
[查看实测结果](#实测结果协议对齐的-opd-追平-sft) ·
[阅读数学说明](docs/math.md)

## 为什么需要 miniVERL

在线策略蒸馏在概念上很简单，工程上很容易出错：学生采样一条轨迹，教师给**学生真实走过的那些状态**打分，然后只在学生自己生成的 token 上做分布级更新。实践中有四类错误，而且它们都是**静默**的：

1. **把工具输出当成了标签。** 环境返回的内容是上下文，不是监督目标。掩码错一次，模型就会学会凭空编造工具结果。
2. **差一位（off-by-one）。** 预测第 `j` 个 token 的分布位于位置 `j - 1`。搞错了，loss 照样下降。
3. **其实并不是 on-policy。** 跨策略版本复用教师缓存，做的就是离线 KD，却仍叫它 OPD。
4. **显存放不下 logits。** `[batch, seq_len, 152k]` 的张量在消费级显卡上放不下，于是真正有意思的配置恰好都跑不了。

miniVERL 把上面每一条都变成**被代码检查的性质**，而不是注释里的一句承诺，并把整个生命周期放进一个可读的单卡进程。

## 已实现的能力

| 能力 | 状态 |
| --- | --- |
| 学生自采样的多轮 rollout，带真实工具执行 | 支持 |
| 严格的逐 token 来源标注（`system` / `user` / `assistant_*` / `tool_result`） | 支持，读写时都会校验 |
| 精确全词表 forward KL、reverse KL、beta-JSD | 支持，与暴力参考实现逐项比对 |
| 压缩的 `top-k + tail` KL / JSD | 支持；未平滑粗粒化与精确散度的下界关系有严格证明 |
| 特权上下文教师模式，带显式对齐表 | 支持 |
| 标准冻结 PEFT 教师适配器，带来源记录与能力门禁 | 支持 |
| 自动选择 bf16/fp16 的单卡 CUDA 路径 | 支持；CUDA 路径不绑定设备名称，实测参考为 RTX 4080 |
| `resident` / `swap` 显存策略与 `auto` 解析 | 支持，并有等价性测试 |
| 带版本号与校验和、完全不用 pickle 的教师目标缓存 | 支持 |
| SFT / 离线 KD / 严格 OPD / 显式标注 replay 统一在一个 trainer 中 | 支持 |
| 计算器、JSON 导航、SQLite 三个环境 | 支持，确定性生成 + 精确判分 |
| 精确的断点续训 | 支持，逐参数断言 |
| 完全自包含、可离线打开的 HTML 报告 | 支持 |
| Ray、FSDP、DeepSpeed、vLLM、VLM、跨词表、PPO/GRPO | **不支持**，见[局限](docs/limitations.md) |

## 实测结果：协议对齐的 OPD 追平 SFT

> [!IMPORTANT]
> **受支持的协议对齐 OPD 在两个种子上都达到 100%，与继续 SFT 持平。**
> 主要的 schema-v2 对照使用两个预先指定的种子、相同的优化步数，以及已经
> 饱和的 `hard` 计算器划分。两个不懂工具协议的实验臂是诊断性负对照，
> 不是推荐配置：

| 角色 | 实验臂 | seed 1234 | seed 20260727 |
| --- | --- | ---: | ---: |
| 起点 | 冷启动 | 75.0% | 75.0% |
| 基线 | 继续 SFT | **100.0%** | **100.0%** |
| 受支持的 OPD | 协议对齐教师 | **100.0%** | **100.0%** |
| 诊断对照 | 未经工具协议训练的原始教师 | 0.0% | 0.0% |
| 诊断对照 | 获知答案但不懂协议的教师 | 0.0% | 0.0% |

[公开且固定版本的协议教师适配器](https://huggingface.co/DaoyuanLi/mini-verl-qwen3-1.7b-protocol-teacher)
现在是单卡配方的默认教师。它在下游 benchmark 被查看之前，已经通过
预先指定的能力门槛。两个负对照均正常完成且两个种子都是 0%；它们不是配置
失败或崩溃。两者使用含歧义的历史 protocol-v1 prompt，故不能把 0% 只归因
于教师内在行为；它诊断的是该设置缺少资格门禁。

OPD 在这里仍然只是追平 SFT，继续训练耗时是 SFT 的 6.1 倍（523.8 秒对
86.4 秒）。任务已经饱和；两个种子既不支持显著性结论，也不构成 OPD
普遍更优的证据。完整结果和旧实验的逐轨迹诊断见
[`docs/rtx4080-baselines.md`](docs/rtx4080-baselines.md)。

![双种子协议教师对照](docs/gpu-calc-hard-equal-update-v2.svg)

| 产物 | 定位 |
| --- | --- |
| [默认配方](recipes/qwen_consumer_gpu_calc.yaml) | 协议合格 |
| [Schema-v2 结果](benchmarks/results/gpu-calc-hard-equal-update-v2.json) | 冻结五臂对照 |
| [Raw-teacher](recipes/qwen_consumer_gpu_calc_raw_teacher.yaml) | 历史对照；非默认 |

<details>
<summary>481 秒 smoke（v1）</summary>

16 步 481 秒，峰值 **4.25/4.76 GiB 已分配/保留**，12 题
从 **0% 到 100%**。冷启动完成了大部分工作（首批 OPD：83.3%）；这证明
流水线，而非 OPD 优于 SFT。[追溯](docs/rtx4080-baselines.md)。

</details>

## 本地玩具演示

不联网、不用 GPU、不下载任何权重。师生两个模型都是由配置直接构建的小型 transformer，分词器是一个可逆的约 190 词条玩具分词器，计算器环境自己生成并判分。

```bash
python -m pip install ".[train]"       # 在克隆后的仓库中执行；CPU 版 torch 就够
miniverl doctor                        # 这台机器能跑什么？
miniverl demo --output runs/demo
```

它跑的是**真实**流水线——学生 rollout、工具执行、教师对这些状态打分、写入带来源校验的压缩 top-k 缓存、只在 assistant token 上做掩码 reverse-KL 更新——然后打印它到底证明了什么：

```text
demo complete  runs/demo
 mode              opd (genuine on-policy distillation)
 optimizer steps   132
 parameter version 132
 rollout iterations 13
 wall clock        52.9 s
 token provenance  45597 of 226383 tokens trainable (20%); 180786 are context
                   and can never be a target
 teacher cache     735 scored positions, 131.6 KiB on disk, 2.0x smaller than
                   a dense fp16 dump
 task success      0.0% -> 0.0% (greedy, held-out eval split)
```

demo 证明的是**机制**，不是能力：在这个规模下玩具学生只学会了工具调用**格式**，学不会算术复制，所以这里的 0% 是预期结果而非失败。想看真正学起来的 CPU 运行（实测 0.0% → 91.7%，192 秒）：

```bash
miniverl train recipes/toy_cpu.yaml
```

这不是承诺而是实测：`recipes/toy_cpu.yaml` 在 CPU 上耗时 **192 秒**，在 24 个留出任务上把贪心成功率从 **0.0% 提升到 91.7%**（600 步监督冷启动 + 40 个在线策略蒸馏 cycle）。它在这个模型规模下**对随机种子敏感**：同样 600 步预算，`run.seed: 1234` 得到 81.2%，`run.seed: 20260727` 得到 0.0%。这个方差正是"玩具后端只是机制验证台、能力数字必须来自 GPU 配方"的原因。

最值得先跑的是 `miniverl inspect`，它打印的来源表就是这个项目的核心：

```text
tokens by span type (only assistant_* can enter the loss)
+---------------------------------------------+
| span type           | tokens | in loss      |
|---------------------+--------+--------------|
| system              |    776 | no (context) |
| tool_result         |    685 | no (context) |
| user                |    318 | no (context) |
| assistant_tool_call |    153 | yes          |
| assistant_text      |     85 | yes          |
| assistant_final     |     25 | yes          |
+---------------------------------------------+
```

玩具后端是**机制验证台，不是能力展示**。它的模型太小，除了 `easy` 难度之外什么都做不了。能力数字来自 GPU 配方。

## 个人单卡快速上手

默认 `device: auto` / `dtype: auto`：新卡用 bf16，Titan V 等旧卡用 fp16。
RTX 3070、Titan V、RTX 4080、RTX 5090 走同一 CUDA 路径，但仅 4080 有实测。
能否装下取决于显存、模型、驱动和 token 预算。修改前请阅读
[`单卡适配指南`](docs/single-gpu-guide.md)。

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl.git
cd mini-verl
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install ".[train,cuda]"

miniverl doctor                                                   # 确认 CUDA 与 bitsandbytes
miniverl validate recipes/qwen_consumer_gpu_calc.yaml
miniverl train    recipes/qwen_consumer_gpu_calc.yaml --dry-run   # 不下载任何东西
miniverl train    recipes/qwen_consumer_gpu_calc.yaml
miniverl report   runs/<run-id> --out runs/<run-id>/report.html
```

配方中两个模型都锁定了 revision：

| 角色 | 模型 | revision | 许可证 |
| --- | --- | --- | --- |
| 学生 | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 |
| 教师 | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0 |

两者的 `tokenizer.json` **逐字节相同**（`sha256 aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`）。新运行首查结构身份；旧产物回退到固定探针行为指纹。

配方还把[协议教师适配器](https://huggingface.co/DaoyuanLi/mini-verl-qwen3-1.7b-protocol-teacher)
锁定在 revision `23323751318135484c06c043b1f9b9e7016dd89f`，并在分配教师模型
之前要求其已记录的严格策略成功率至少达到 50%。

## 精确 vs. top-k + tail

两类目标函数被明确区分命名，因为把它们混为一谈正是蒸馏结果无法复现的常见原因。

**`exact_full_vocab`** 会构造完整的 `[chunk, V]` 师生分布并计算真实散度。适用于词表很小（玩具后端），或教师常驻显存、分布按 chunk 现算的情况。由 `loss.exact_max_vocab`（默认 8192）兜底，避免静默地尝试持久化 `[positions, 152k]` 张量。

**`bucketed_topk_tail`** 把词表粗粒化为「教师 top-k 个 token + 一个聚合尾桶」，再在两个 `K+1` 类分布之间算散度。这**不是**全词表 KL。数据处理不等式严格适用于未平滑的粗粒化；实际实现会对非空尾桶做 epsilon 下限和重新归一化，因此文档把它称为 epsilon 平滑目标，不宣称每个输入上仍严格满足该定理。当 `k == V` 时空尾桶绕过平滑，float64 测试确认实现与精确目标在 `1e-9` 内一致。函数名就叫 `bucketed_forward_kl` / `bucketed_reverse_kl` / `bucketed_jsd`，任何调用点都无法把它伪装成精确 KL。

压缩真正省下的是**教师侧的存储**，以及把教师从显存中换出的能力。它**不会**按比例减少教师的 FLOPs——教师仍要跑一次完整前向来产生 hidden states。因此报告里写的是 `teacher_queried_position_ratio`，绝不写"节省了教师算力"。

top-k + tail 目标本身并不新颖：TRL 的 `ServerDistillationTrainer` 就有 `loss_top_k` 和可选的尾桶。详见 [`docs/math.md`](docs/math.md)。

## 工具 token 掩码

每条轨迹都是「一维 token 序列 + 类型化 span 划分」。三个掩码既被存储，**也**在每次读取时从 span 重新推导；一旦掩码与 span 不一致，文件会被拒绝而不是拿去训练。

上下文 span 会包含结尾的 `<|im_start|>assistant\n` 头部，因此模型 span 恰好从第一个采样 token 开始，任何被强制写入的脚手架 token 都不会成为监督目标。位置 `0` 永远不能作为目标。这两点都由 `tests/unit/test_token_provenance.py` 强制执行。

## 基准结果

下面所有数字都由 [`docs/benchmarking.md`](docs/benchmarking.md) 中的命令、在对应结果文件记录的硬件上跑出来。没有估算，没有外推。

* **RTX 4080，真实模型** —— [`docs/rtx4080-baselines.md`](docs/rtx4080-baselines.md) 记录了实测显存峰值、解码吞吐、完整配方运行以及旧版等优化器更新对照，同时也记录了尝试过的配置和**未运行**的配置。
* **CPU，玩具模型** —— `recipes/toy_cpu.yaml` 在 192 秒内把成功率从 0.0% 提到 91.7%；`benchmarks/results/` 中还有旧版等优化器更新机制对照。后者的准确率差异**在噪声范围内**，作用是证明所有分支在同一比较轴下都能跑完，而不是给它们排名。原因见 [`benchmarks/README.md`](benchmarks/README.md)。

## 安装分层

| 层次 | 安装命令 | 得到什么 |
| --- | --- | --- |
| 核心 | `python -m pip install .` | `doctor`、`validate`、`inspect`、`report`、`cache`，以及 schema 和 Python API。**不含 torch**。 |
| 训练 | `python -m pip install ".[train]"` | `demo`、`train`、`eval`、`benchmark`。加入 torch、transformers、peft、accelerate。 |
| 4-bit | `python -m pip install ".[cuda]"` | bitsandbytes，用于 NF4 QLoRA 与 8-bit 优化器。 |
| 开发 | `python -m pip install ".[dev]"` | pytest、hypothesis、ruff、mypy、build、twine。 |

缺少可选依赖时不会抛出裸异常：

```text
$ miniverl demo --output runs/demo
error miniverl demo requires the optional dependency 'torch', which is not installed.
hint  pip install "miniverl[train]"
```

### 严格离线执行

所有会加载模型的命令共享同一份“零网络”契约：

```bash
miniverl train <recipe> --offline
miniverl benchmark <benchmark.yaml> --offline
miniverl eval --run <run-dir> --offline
miniverl export-adapter --run <run-dir> --out <adapter-dir> --offline
```

此模式要求基础模型、分词器和每个适配器文件已经位于本地路径或 Hugging
Face 缓存中；miniVERL 不允许 HTTP、metadata、ETag 或 Hub API 请求，也不会
静默退回在线解析。Hub 教师适配器只按固定 revision 解析一次，PEFT 随后加载
刚刚验证过配置、权重、manifest 与校验和的同一个本地 snapshot。缓存缺失时，
错误会给出不可变身份以及精确的 `hf download` 预加载命令。

## Python API

对外暴露的接口刻意保持很小：

```python
from miniverl.config import RunConfig
from miniverl.trainer import OPDTrainer

config = RunConfig.from_yaml("recipes/toy_cpu.yaml")
with OPDTrainer.from_config(config) as trainer:
    result = trainer.train()

print(result.run_dir, result.global_step, result.eval["success_rate"])
```

自定义环境见 `examples/custom_environment/`，自定义教师见 `examples/custom_teacher/`，两个例子都可直接运行。

标准冻结 PEFT 教师适配器、Qwen3 协议 SFT 配方、导出命令、兼容性检查和教师策略能力门禁见
[`docs/teacher-adapters.md`](docs/teacher-adapters.md)。

## 局限

简版如下，完整清单见 [`docs/limitations.md`](docs/limitations.md)。

* 仅支持师生同一分词器；跨词表蒸馏会直接报错。
* 每次前向只处理一条轨迹，因此 `gradient_accumulation_steps` **就是** batch size；当前版本没有 padding 批处理。
* 量化模型不能用 `swap`，因为 bitsandbytes 的参数绑定在量化时所在的设备上。
* 只测试过 Qwen3 与 Qwen2 架构。其他架构可能能通过架构适配器工作，但本项目不作任何声明。
* 主要 GPU 对照使用两个预先指定的随机种子；旧 GPU 产物仍是单种子。不声称统计显著性。
* 在实测机器上，解码是 kernel 启动开销受限而非算力受限，因此吞吐数字与平台强相关。

## 可复现性

每次运行都会写出 `manifest.json`，记录 miniVERL 版本、git commit、Python 与操作系统、torch/CUDA/驱动版本、GPU 型号与显存、模型 id **及解析后的 revision**、分词器指纹、随机种子、精度、量化、显存策略、损失模式、top-k、策略版本，以及一个 `measurement_status` 块，说明每项结果是实测、模拟还是未运行。

可写运行会原子地经过 `ready`、`running`，再进入 `completed`、`failed`、`interrupted` 或 `closed_before_training` 终态。同一把跨进程锁覆盖构造、训练/续训、独立评估的 checkpoint 选择与加载，以及自动报告；训练持有模型时，外部评估和 checkpoint 调用会被拒绝。

每个内置环境在 `reset` 后都会把任意字符串变成有界验证结果，不向外泄漏解析或数值异常；protocol-v2 使用各环境可被验证器接受的 final 格式示例。可分享的 HTML、Markdown、JSON、benchmark 导出和 portable manifest 会遮蔽语义化密钥、URL 凭证及跨平台私有路径，而私有运行目录仍保留精确续训所需的本地状态。

它**不**记录用户名、主机名、家目录，也不记录白名单之外的任何环境变量（白名单只包含会影响数值结果的少数几个）——这一点有测试断言。
来自文件的运行还会分别保存原始提交字节、规范化验证配置、v0.2
续训兼容层和运行时解析后的选择。

详见 [`docs/reproducibility.md`](docs/reproducibility.md) 和
[`兼容性策略`](docs/compatibility.md)。

## 路线图

以下均**未实现**、也不作承诺，仅为明确边界：跨词表蒸馏、padding 多序列批处理、熵感知散度混合（arXiv:2603.07079）、更多模型族、更多环境、多卡。任何集群规模的需求，请直接用 verl。

## 致谢与声明

> miniVERL 是一个独立项目，与 verl 项目、字节跳动（ByteDance）或火山引擎（Volcano Engine）没有隶属关系，也未获得其背书。它**不是** verl 的直接替代品。

这个名字只是对问题领域的致意，不代表任何兼容性声明。verl 是一个优秀得多、规模也大得多的系统，它同样实现了在线策略蒸馏和多轮工具调用——只不过是在集群规模上，依赖 Ray。如果你有集群，请用它。miniVERL 面向的是「只有一块个人显卡、并且希望把每一行发生的事都读懂」的场景：它可以是较老的 12 GiB 显卡，也可以是当前的高端显卡；仓库只对实际跑过的硬件声明实测性能。对比见 [`docs/comparisons.md`](docs/comparisons.md)。

## 引用与许可证

引用格式见 [CITATION.cff](CITATION.cff)，变更记录见 [CHANGELOG.md](CHANGELOG.md)，贡献指南见 [CONTRIBUTING.md](CONTRIBUTING.md)，安全策略见 [SECURITY.md](SECURITY.md)。

Apache-2.0，见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
