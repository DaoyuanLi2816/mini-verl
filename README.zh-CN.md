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

**在一张个人 GPU 上运行有明确边界的 verl 风格在线策略蒸馏。** miniVERL
读取类型化的 verl v0.8 OPD profile 与 Parquet prompt，在本地执行 actor rollout
→ teacher scoring → actor update，并导出标准 PEFT/Parquet/config 产物用于扩展。
原生 SFT、DPO、KD 与 tool-agent recipe 仍然保留。

PyPI `v0.8.0` 是稳定版；`main` 是开发版。miniVERL 独立于 verl，不声称可
执行任意 verl YAML、分布式任务或具备完整算法兼容性。

## 仅用 pip 的 OPD quickstart

```bash
python -m pip install "miniverl[train]"
miniverl data sample --format verl-parquet --out prompts.parquet
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config builtin:qwen3-0.6b-1.7b-opd \
  --set 'data.train_files=["prompts.parquet"]'
miniverl run --profile verl-opd-v0.8-single-gpu-v1 --config builtin:qwen3-0.6b-1.7b-opd \
  --set 'data.train_files=["prompts.parquet"]' --dry-run
```

sample、plan 与 dry run 无需 Git checkout，plan 也不加载权重。在一张 CUDA GPU
上去掉 `--dry-run` 即可执行锁定的 Qwen3-0.6B/1.7B NF4 recipe，并生成可加载的
PEFT adapter。参见 [OPD quickstart](docs/opd-quickstart.md)。

## 支持的硬件与运行边界

miniVERL 在 CPU 或一张 NVIDIA CUDA GPU 上运行单个本地进程。CUDA 路径不按
显卡名称设限，但能否装下取决于模型组合、上下文、kernel 与显存。请先安装匹配
本机 CUDA 的 PyTorch wheel，再安装 `miniverl[train,cuda]`；该 extra 本身不会
选择 CUDA PyTorch。Ray、FSDP、Megatron、PPO、GRPO 与分布式启动不属于当前
运行时。参见[单卡指南](docs/single-gpu-guide.md)。

## verl 兼容性摘要

可执行 profile 锁定官方 verl `v0.8.0`、commit `7aed6b23`，支持单 actor、单 teacher、
`n=1`、纯 GKD `forward_kl_topk`、token-mean、LoRA/QLoRA，且不使用 reward 或 KL
penalty。PG OPD、task-reward mixture、多 teacher、多模态与分布式字段会 fail closed。

兼容的 OPD 导出不包含 reward scaffold；它保留 student/teacher 身份、Parquet 原始
字节与 OPD overrides，但在精确 base snapshot 尚未 materialize 时仍为
`launchable: false`。解析、产物可加载性、launchability 与分布式执行分别报告。
参见[桥接契约](docs/verl-bridge.md)。

## RTX 4080 运行时实测

打包的 Qwen3-0.6B/1.7B recipe 完成了两条 16-token rollout 与一次 OPD update；
**peak reserved VRAM 为 3.1758 GiB**，首次 update 在 **12.0224 秒**完成，标准
PEFT adapter 成功重新加载。这只证明一个运行时/产物路径；没有运行对齐质量
endpoint 或方法比较。[精确 recipe、计时与哈希](docs/opd-quickstart.md)。

## 三条使用路径

| 路径 | 起点 | 真实产物 | 下一步 |
| --- | --- | --- | --- |
| **本地运行 OPD** | `miniverl plan --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml` | compiled plan、trajectory、target 与 PEFT adapter | [Plan 与 run](docs/opd-quickstart.md) |
| **带入 verl config** | `miniverl import-verl --profile verl-opd-v0.8-single-gpu-v1 --config verl-opd.yaml --out local-opd.yaml` | 字段报告与可往返 profile | [兼容性](docs/compatibility.md) |
| **移动数据与产物** | `miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout` | Parquet + PEFT + OPD override bundle | [桥接契约](docs/verl-bridge.md) |

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
