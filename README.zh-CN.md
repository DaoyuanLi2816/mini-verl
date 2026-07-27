<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/main/docs/banner.svg" alt="miniVERL — 单卡上的工具调用智能体在线策略蒸馏" width="880">
</p>

<div align="center">

[![CI](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/ci.yml)
[![Build](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml/badge.svg)](https://github.com/DaoyuanLi2816/mini-verl/actions/workflows/build.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

</div>

> 本文是 [README.md](README.md) 的中文翻译。英文版为准；若两者不一致，请以英文版为准并提交 issue。

**单卡上的工具调用智能体在线策略蒸馏（on-policy distillation）。**

miniVERL 让一个小型的、会调用工具的语言模型在**它自己生成的多轮轨迹**上学习，监督信号来自教师模型的稠密 token 级分布。不需要 Ray，不需要 GPU 集群，也不需要 40 GB 显存的加速卡。

```bash
pip install "miniverl[train]"
miniverl demo --output runs/demo        # 无需联网、无需 GPU，笔记本 CPU 上约 50 秒
```

在一块 RTX 4080（16 GB）上，仓库内的正式配方用 **Qwen3-1.7B** 蒸馏 **Qwen3-0.6B**，在一个两轮计算器任务上耗时 **481 秒 / 16 个优化步**，显存峰值 **4.25 GiB 已分配 / 4.76 GiB 已保留**，留出集上贪心解码的任务成功率从 **0.0% 提升到 100.0%**（12 个任务）。

> 请如实理解这个数字：其中大部分提升来自 8 个 cycle 的监督冷启动——**第一批**在线策略 rollout 就已经达到 83.3%。这次运行证明的是整条流水线能在真实硬件上端到端跑通，**并不能**说明在线策略蒸馏优于监督微调，因为该任务已经饱和。真正回答这个问题的等预算对照实验见 [`docs/rtx4080-baselines.md`](docs/rtx4080-baselines.md)。本文中的每个数字都可以从那里记录的产物复现。

---

## 为什么需要 miniVERL

在线策略蒸馏在概念上很简单，工程上很容易出错：学生采样一条轨迹，教师给**学生真实走过的那些状态**打分，然后只在学生自己生成的 token 上做分布级更新。实践中有四类错误，而且它们都是**静默**的：

1. **把工具输出当成了标签。** 环境返回的内容是上下文，不是监督目标。掩码错一次，模型就会学会凭空编造工具结果。
2. **差一位（off-by-one）。** 预测第 `j` 个 token 的分布位于位置 `j - 1`。搞错了，loss 照样下降。
3. **其实并不是 on-policy。** 跨策略版本复用教师缓存，做的就是离线 KD，却仍叫它 OPD。
4. **显存放不下 logits。** `[batch, seq_len, 152k]` 的张量在消费级显卡上放不下，于是真正有意思的配置恰好都跑不了。

miniVERL 把上面每一条都变成**被代码检查的性质**，而不是注释里的一句承诺，并且整套流程只用一块 16 GB 显卡。

## 已实现的能力

| 能力 | 状态 |
| --- | --- |
| 学生自采样的多轮 rollout，带真实工具执行 | 支持 |
| 严格的逐 token 来源标注（`system` / `user` / `assistant_*` / `tool_result`） | 支持，读写时都会校验 |
| 精确全词表 forward KL、reverse KL、beta-JSD | 支持，与暴力参考实现逐项比对 |
| 压缩的 `top-k + tail` KL / JSD | 支持，并证明了它是精确散度的下界 |
| 特权上下文教师模式，带显式对齐表 | 支持 |
| QLoRA（NF4）学生，bf16 或量化教师 | 支持，已在 RTX 4080 上实测 |
| `resident` / `swap` 显存策略与 `auto` 解析 | 支持，并有等价性测试 |
| 带版本号与校验和、完全不用 pickle 的教师目标缓存 | 支持 |
| SFT / 离线 KD / 真正的 OPD 统一在一个 trainer 中 | 支持 |
| 计算器、JSON 导航、SQLite 三个环境 | 支持，确定性生成 + 精确判分 |
| 精确的断点续训 | 支持，逐参数断言 |
| 完全自包含、可离线打开的 HTML 报告 | 支持 |
| Ray、FSDP、DeepSpeed、vLLM、VLM、跨词表、PPO/GRPO | **不支持**，见[局限](docs/limitations.md) |

## 本地玩具演示

不联网、不用 GPU、不下载任何权重。师生两个模型都是由配置直接构建的小型 transformer，分词器是一个可逆的约 190 词条玩具分词器，计算器环境自己生成并判分。

```bash
pip install "miniverl[train]"          # CPU 版 torch 就够
miniverl doctor                        # 这台机器能跑什么？
miniverl demo --output runs/demo
```

它跑的是**真实**流水线——学生 rollout、工具执行、教师对这些状态打分、写入带来源校验的压缩 top-k 缓存、只在 assistant token 上做掩码 reverse-KL 更新——然后打印它到底证明了什么：

```text
demo complete  runs/demo
 mode              opd (genuine on-policy distillation)
 optimizer steps   132
 policy versions   13
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

## 消费级 GPU 快速上手

```bash
pip install "miniverl[train,cuda]"
pip install torch --index-url https://download.pytorch.org/whl/cu130   # 与你的驱动匹配

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

两者的 `tokenizer.json` **逐字节相同**（`sha256 aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`），这正是"同一分词器"契约成立的依据。miniVERL 在加载时用行为指纹核对，不一致就直接报错退出。

## 精确 vs. top-k + tail

两类目标函数被明确区分命名，因为把它们混为一谈正是蒸馏结果无法复现的常见原因。

**`exact_full_vocab`** 会构造完整的 `[chunk, V]` 师生分布并计算真实散度。适用于词表很小（玩具后端），或教师常驻显存、分布按 chunk 现算的情况。由 `loss.exact_max_vocab`（默认 8192）兜底，避免静默地尝试持久化 `[positions, 152k]` 张量。

**`bucketed_topk_tail`** 把词表粗粒化为「教师 top-k 个 token + 一个聚合尾桶」，再在两个 `K+1` 类分布之间算散度。这**不是**全词表 KL。由数据处理不等式，它是精确散度的**下界**，当 `k == V` 时取等；两点都有测试断言，后者在 float64 下精确到 `1e-9`。函数名就叫 `bucketed_forward_kl` / `bucketed_reverse_kl` / `bucketed_jsd`，任何调用点都无法把它伪装成精确 KL。

压缩真正省下的是**教师侧的存储**，以及把教师从显存中换出的能力。它**不会**按比例减少教师的 FLOPs——教师仍要跑一次完整前向来产生 hidden states。因此报告里写的是 `teacher_queried_position_ratio`，绝不写"节省了教师算力"。

top-k + tail 目标本身并不新颖：TRL 的 `ServerDistillationTrainer` 就有 `loss_top_k` 和可选的尾桶。详见 [`docs/math.md`](docs/math.md)。

## 工具 token 掩码

每条轨迹都是「一维 token 序列 + 类型化 span 划分」。三个掩码既被存储，**也**在每次读取时从 span 重新推导；一旦掩码与 span 不一致，文件会被拒绝而不是拿去训练。

上下文 span 会包含结尾的 `<|im_start|>assistant\n` 头部，因此模型 span 恰好从第一个采样 token 开始，任何被强制写入的脚手架 token 都不会成为监督目标。位置 `0` 永远不能作为目标。这两点都由 `tests/unit/test_token_provenance.py` 强制执行。

## 基准结果

下面所有数字都由 [`docs/benchmarking.md`](docs/benchmarking.md) 中的命令、在对应结果文件记录的硬件上跑出来。没有估算，没有外推。

* **RTX 4080，真实模型** —— [`docs/rtx4080-baselines.md`](docs/rtx4080-baselines.md) 记录了实测显存峰值、解码吞吐、完整配方运行以及等预算对照，同时也记录了尝试过的配置和**未运行**的配置。
* **CPU，玩具模型** —— `recipes/toy_cpu.yaml` 在 192 秒内把成功率从 0.0% 提到 91.7%；`benchmarks/results/` 中还有等预算对照运行。后者的准确率差异**在噪声范围内**，作用是证明所有分支在同一预算下都能跑完，而不是给它们排名。原因见 [`benchmarks/README.md`](benchmarks/README.md)。

## 安装分层

| 层次 | 安装命令 | 得到什么 |
| --- | --- | --- |
| 核心 | `pip install miniverl` | `doctor`、`validate`、`inspect`、`report`、`cache`，以及 schema 和 Python API。**不含 torch**。 |
| 训练 | `pip install "miniverl[train]"` | `demo`、`train`、`eval`、`benchmark`。加入 torch、transformers、peft、accelerate。 |
| 4-bit | `pip install "miniverl[cuda]"` | bitsandbytes，用于 NF4 QLoRA 与 8-bit 优化器。 |
| 开发 | `pip install "miniverl[dev]"` | pytest、hypothesis、ruff、mypy、build、twine。 |

缺少可选依赖时不会抛出裸异常：

```text
$ miniverl demo --output runs/demo
error miniverl demo requires the optional dependency 'torch', which is not installed.
hint  pip install "miniverl[train]"
```

## Python API

对外暴露的接口刻意保持很小：

```python
from miniverl.config import RunConfig
from miniverl.trainer import OPDTrainer

config = RunConfig.from_yaml("recipes/toy_cpu.yaml")
trainer = OPDTrainer.from_config(config)
result = trainer.train()

print(result.run_dir, result.global_step, result.eval["success_rate"])
```

自定义环境见 `examples/custom_environment/`，自定义教师见 `examples/custom_teacher/`，两个例子都可直接运行。

## 局限

简版如下，完整清单见 [`docs/limitations.md`](docs/limitations.md)。

* 仅支持师生同一分词器；跨词表蒸馏会直接报错。
* 每次前向只处理一条轨迹，因此 `gradient_accumulation_steps` **就是** batch size；v0.1 没有 padding 批处理。
* 量化模型不能用 `swap`，因为 bitsandbytes 的参数绑定在量化时所在的设备上。
* 只测试过 Qwen3 与 Qwen2 架构。其他架构可能能通过架构适配器工作，但本项目不作任何声明。
* GPU 结果是单随机种子，不声称统计显著性。
* 在实测机器上，解码是 kernel 启动开销受限而非算力受限，因此吞吐数字与平台强相关。

## 可复现性

每次运行都会写出 `manifest.json`，记录 miniVERL 版本、git commit、Python 与操作系统、torch/CUDA/驱动版本、GPU 型号与显存、模型 id **及解析后的 revision**、分词器指纹、随机种子、精度、量化、显存策略、损失模式、top-k、策略版本，以及一个 `measurement_status` 块，说明每项结果是实测、模拟还是未运行。

它**不**记录用户名、主机名、家目录，也不记录白名单之外的任何环境变量（白名单只包含会影响数值结果的少数几个）——这一点有测试断言。

详见 [`docs/reproducibility.md`](docs/reproducibility.md)。

## 路线图

以下均**未实现**、也不作承诺，仅为明确边界：跨词表蒸馏、padding 多序列批处理、熵感知散度混合（arXiv:2603.07079）、更多模型族、更多环境、多卡。任何集群规模的需求，请直接用 verl。

## 致谢与声明

> miniVERL 是一个独立项目，与 verl 项目、字节跳动（ByteDance）或火山引擎（Volcano Engine）没有隶属关系，也未获得其背书。它**不是** verl 的直接替代品。

这个名字只是对问题领域的致意，不代表任何兼容性声明。verl 是一个优秀得多、规模也大得多的系统，它同样实现了在线策略蒸馏和多轮工具调用——只不过是在集群规模上，依赖 Ray。如果你有集群，请用它。miniVERL 面向的是「只有一块消费级显卡、并且希望把每一行发生的事都读懂」的场景。对比见 [`docs/comparisons.md`](docs/comparisons.md)。

## 引用与许可证

引用格式见 [CITATION.cff](CITATION.cff)，变更记录见 [CHANGELOG.md](CHANGELOG.md)，贡献指南见 [CONTRIBUTING.md](CONTRIBUTING.md)，安全策略见 [SECURITY.md](SECURITY.md)。

Apache-2.0，见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
