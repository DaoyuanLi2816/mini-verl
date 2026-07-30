# References

Primary sources for the ideas miniVERL implements, and for the projects it is
compared against. Each entry is tagged `[primary]` for a peer-reviewed paper,
preprint or source repository, or `[blog]` for an informal write-up.

Every arXiv identifier below was resolved on 2026-07-27 and the title, author
list and version dates were read off the abstract page. Mutable repository and
official-documentation facts were refreshed through the GitHub API and project
documentation on 2026-07-29.

## The framework miniVERL is named after

**HybridFlow: A Flexible and Efficient RLHF Framework** `[primary]`
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang,
Yanghua Peng, Haibin Lin, Chuan Wu. arXiv:2409.19256, submitted 2024-09-28.
<https://arxiv.org/abs/2409.19256>
The paper verl implements. Its hybrid single-controller / multi-controller
dataflow is the scaling design miniVERL deliberately does not attempt; the name
"miniVERL" acknowledges the debt without implying compatibility.

**verl** `[primary]`
`verl-project/verl`, Apache-2.0. <https://github.com/verl-project/verl>
The reference implementation. Relevant to this project because it already has
first-class on-policy distillation (`verl/trainer/distillation/`, with FSDP and
Megatron backends and a `DistillationConfig` in `verl.workers.config`) and an
agent loop with tool support (`verl/experimental/agent_loop/`, including
`tool_agent_loop.py` and `tool_parser.py`). Ray is an unconditional dependency
(`ray[default]` in `requirements.txt`). See
[comparisons.md](comparisons.md) for when to use verl instead of this project.

## On-policy distillation

**On-Policy Distillation of Language Models: Learning from Self-Generated
Mistakes** `[primary]`
Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos,
Matthieu Geist, Olivier Bachem. arXiv:2306.13649, submitted 2023-06-23.
<https://arxiv.org/abs/2306.13649>
The GKD paper. Source of the two ideas miniVERL builds the objective around:
training the student on its own sampled sequences to remove the
train/inference distribution mismatch, and the generalized Jensen-Shannon
divergence as a tunable interpolation between forward and reverse KL.
`miniverl.losses.exact.exact_jsd` and `bucketed_jsd` implement the
beta-weighted form, with `beta` restricted to the open interval `(0, 1)`
because the divergence is identically zero at either endpoint.

**GKD Trainer documentation** `[primary]`
Hugging Face TRL. <https://huggingface.co/docs/trl/main/en/gkd_trainer>
The reference implementation of GKD in the transformers ecosystem, now at
`trl.experimental.gkd.GKDTrainer` / `GKDConfig`. Useful as a calibration point:
`lmbda` defaults to `0.5`, so on-policy sampling is configurable rather than the
default, and the trainer recomputes full-vocabulary teacher logits each step
under `no_grad` with no teacher cache.

**On-Policy Context Distillation for Language Models** `[primary]`
Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, Furu Wei. arXiv:2602.12275,
v1 2026-02-12, v2 2026-03-23. <https://arxiv.org/abs/2602.12275>
The student trains on its own trajectories while minimizing reverse KL against a
teacher that is conditioned on additional context the student never sees. This
is exactly what `models.teacher.mode: privileged_context` does in miniVERL: the
teacher trajectory is re-rendered with an environment-supplied hint and
`miniverl.trajectory.alignment.build_alignment_map` recovers the position
correspondence so the student is still scored on its own tokens.

**Entropy-Aware On-Policy Distillation of Language Models** `[primary]`
Woogyeol Jin, Taywon Min, Yongjin Yang, Dennis Wei, Yi Zhou,
Swanand Ravindra Kadhe, Nathalie Baracaldo, Kimin Lee. arXiv:2603.07079,
v1 2026-03-07, v2 2026-05-22, v3 2026-06-12; the author comment reads
"18 pages, 11 figures, ICML 2026". <https://arxiv.org/abs/2603.07079>
Argues that reverse KL is mode-seeking and destabilizes where the teacher's
entropy is high, and mixes in forward KL at those tokens. This is the reason
miniVERL records per-selected-position teacher entropy
(`exact_teacher_entropy`, `bucketed_teacher_entropy`) and surfaces it in the
reports. The mixing itself is **not implemented** here; see the roadmap section
of [limitations.md](limitations.md).

**Rethinking On-Policy Distillation of Large Language Models: Phenomenology,
Mechanism, and Recipe** `[primary]`
Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian,
Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, Ning Ding. arXiv:2604.13016,
v1 2026-04-14, v2 2026-04-15. <https://arxiv.org/abs/2604.13016>
Official code: `thunlp/OPD`, <https://github.com/thunlp/OPD>. Builds on verl
v0.7.0 with LlamaFactory v0.9.5 for the SFT stage; experiments run on 8xA800
80 GB; the repository has **no LICENSE file**, so it is all-rights-reserved.
Relevant as a statement of the conditions under which on-policy distillation
works at all, including compatible student and teacher behaviour, which is one
motivation for miniVERL enforcing an identical tokenizer.

**KDFlow: A User-Friendly and Efficient Knowledge Distillation Framework for
Large Language Models** `[primary]`
Songming Zhang, Xue Zhang, Tong Zhang, Bojie Hu, Yufeng Chen, Jinan Xu.
arXiv:2603.01875, v1 2026-03-02, v2 2026-03-24, v3 2026-07-17.
<https://arxiv.org/abs/2603.01875>
Code: `songmzhang/KDFlow`, MIT. <https://github.com/songmzhang/KDFlow>
Requires Ray and SGLang unconditionally, and its examples assume 8 GPUs per
node. A code search over the repository for `tool_call` and for `agent` returns
0 results, which is the clearest single contrast with miniVERL's scope.

**OPSD (On-Policy Distillation)** `[primary]`
`HJSang/OPSD_OnPolicyDistillation`.
<https://github.com/HJSang/OPSD_OnPolicyDistillation>
A research harness built on verl. Its README describes multi-turn agent-loop
rollouts with tool and environment tokens excluded from the loss, and chunked
divergence computation rather than materializing full-vocabulary tensors for a
whole batch. The repository has **no LICENSE file**, so despite the overlap in
approach none of it is reusable.

## Distillation mechanics

**Distilling the Knowledge in a Neural Network** `[primary]`
Geoffrey Hinton, Oriol Vinyals, Jeff Dean. arXiv:1503.02531, submitted
2015-03-09. <https://arxiv.org/abs/1503.02531>
Source of the classic `T**2` correction for soft-target cross-entropy / forward
KL in the near-uniform high-temperature regime.
`miniverl.losses.exact.temperature_scale` applies the factor, and
`loss.scale_by_temperature_squared` controls it. Its use for reverse KL and JSD
is explicitly treated as a heuristic and measured by
`scripts/temperature_gradient_sweep.py`, not presented as a general invariance.

## Numerics

**Accurately Computing log(1 - exp(.)) -- Assessed by Rmpfr** `[primary]`
Martin Maechler, 2012. CRAN `Rmpfr` package vignette; the PDF is rebuilt with
each package release, and the copy fetched on 2026-07-27 is dated 2025-10-21.
<https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf>
The two-regime algorithm implemented by `miniverl.losses.numerics.log1mexp`:
`log(-expm1(x))` when `x` is close to zero and `log1p(-exp(x))` when it is not,
switching at `-log 2`. miniVERL needs it to compute the tail log-probability
`log(1 - sum_k p)` of a top-k bucket without catastrophic cancellation, and
evaluates both branches on sanitized inputs so the unused branch cannot inject
`nan` into the backward pass.

## Parameter-efficient and quantized fine-tuning

**LoRA: Low-Rank Adaptation of Large Language Models** `[primary]`
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li,
Shean Wang, Lu Wang, Weizhu Chen. arXiv:2106.09685, v1 2021-06-17,
v2 2021-10-16. <https://arxiv.org/abs/2106.09685>
The adapter method behind `models.student.lora`. In the measured 16 GB run the
student had 10,092,544 trainable LoRA parameters against 385,941,504 base
parameters.

**QLoRA: Efficient Finetuning of Quantized LLMs** `[primary]`
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. arXiv:2305.14314,
submitted 2023-05-23. <https://arxiv.org/abs/2305.14314>
The NF4 plus double-quantization recipe behind `models.student.quantization:
nf4`. It is also the direct cause of one of miniVERL's hard limits: bitsandbytes
4-bit parameters are pinned to the device they were quantized on, so
`memory.strategy: swap` is rejected for any quantized model.

## Storage format

**safetensors** `[primary]`
`safetensors/safetensors` (formerly `huggingface/safetensors`), Apache-2.0.
<https://github.com/safetensors/safetensors>
The on-disk format for the teacher-target cache. It was chosen because loading a
cache shard must not be able to execute code: `torch.save` and `pickle` are
never used anywhere in `miniverl.cache`, and
`miniverl.cache.store.read_safetensors_header` parses the JSON header without
torch, numpy or the safetensors library so that `miniverl cache stats` works on
a base install.

## Pinned models

Both revisions were re-resolved against the Hugging Face API on 2026-07-27,
alongside a negative control (an all-zero SHA returns HTTP 404, so the endpoint
really validates the revision).

**Qwen/Qwen3-0.6B** `[primary]` -- revision
`c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0.
<https://huggingface.co/Qwen/Qwen3-0.6B>
The student in `recipes/qwen_consumer_gpu_calc.yaml`.

**Qwen/Qwen3-1.7B** `[primary]` -- revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Apache-2.0.
<https://huggingface.co/Qwen/Qwen3-1.7B>
The teacher in the same recipe. `tokenizer.json` is byte-identical across the
pair (sha256
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`), which is
what makes the pair legal under miniVERL's same-tokenizer requirement. Both
report `vocab_size` 151936 while `len(tokenizer)` is 151669, because the
embedding matrix is padded; both have 28 layers and `tie_word_embeddings: true`.
Qwen3 requires `transformers >= 4.51`.
