# Limitations

This page is the list of reasons not to trust miniVERL beyond what it has
actually been shown to do. It is deliberately longer than the feature list.

Everything below is traceable to a file in this repository or to a command that
was run against it. Measurements come from one machine: Windows 11 Pro
10.0.22631, RTX 4080 (16376 MiB, driver 596.49), CPython 3.12.13, torch
2.13.0+cu130, transformers 5.14.1, peft 0.19.1, bitsandbytes 0.50.0. The GPU
figures include the published calculator runs, hardware probes and
RecoveryBench v1 plus Alignment Lab v1. Their source artifacts, methods and caveats are linked from
the corresponding sections below.

## Modelling and objective

### Same tokenizer on both sides, with no fallback

The student and the teacher must tokenize identically. This is enforced by the
code rather than documented and hoped for:

- `miniverl.models.factory.build_tokenizer` returns **one** tokenizer object,
  which both backends then share. When the teacher declares a different
  tokenizer id, that tokenizer is loaded and
  `miniverl.models.tokenizers.assert_same_tokenizer` is called before either
  model is constructed. New runs compare structural identity first; any
  difference raises `TokenizerMismatchError`.
- `miniverl.trajectory.alignment.build_alignment_map` and
  `miniverl.teachers.local.LocalTeacherScorer.score` both re-check the
  fingerprint on the teacher render used by `privileged_context` mode.

Structural identity hashes the full vocabulary, added vocabulary, special-token
map, backend tokenizer and behaviour-relevant tokenizer configuration. Local
source paths are canonicalized away. The legacy fingerprint is a SHA-256 over
the tokenizer class, length, special-token metadata and token ids produced for
one fixed string (`miniverl.models.tokenizers.PROBE_TEXT`). It remains a
compatibility fallback when an old artifact has no structural digest; agreement
on that probe is not proof that two tokenizer structures are identical.

The reason is structural, not incidental. A bucketed teacher target is a set of
vocabulary **indices** plus log-probabilities; the student gathers its own
log-probabilities at those same indices. If the two vocabularies differ, the
indices mean different tokens and the divergence is computed between unrelated
coordinates. Making this work requires a token-alignment step that miniVERL does
not have. Pick a teacher from the student's own model family.

### `exact_full_vocab` is guarded, and the guard does not cover every path

`loss.mode: exact_full_vocab` materializes complete `[chunk_size, V]`
distributions. `loss.exact_max_vocab` (default `8192`) plus
`loss.allow_large_exact` (default `false`) exist to stop you doing that
accidentally on a 151936-entry vocabulary.

Read the guard carefully, because it is narrower than it looks. In
`miniverl.teachers.local.LocalTeacherScorer.score`, the check
`_check_exact_is_affordable` runs only when the teacher is **not** resident,
that is, when the targets have to be serialized for `memory.strategy: swap`.
With `memory.strategy: resident` the exact path is taken without the guard:
only `[N, H]` teacher hidden states are kept and the `[chunk, V]` logits are
rebuilt per chunk on demand.

That path is legal, and it is also expensive. A single `[256, 151936]` float32
tensor is 148 MiB, and the chunk loop holds several of them at once (teacher
logits, teacher log-probabilities, student logits, student log-probabilities)
before any gradient is stored. If you enable exact mode on a full-size
vocabulary, lower `loss.chunk_size` first.

Exact-resident teacher targets are also **not cacheable**: the scorer returns
`cacheable=None` for that shape, so no teacher cache is written and
`miniverl cache stats` has nothing to report.

### The unfloored bucketed divergence is a lower bound, not the KL you want

`bucketed_topk_tail` coarse-grains both distributions into `K + 1` categories:
one per teacher top-k token, plus a single bucket holding all remaining mass. By
the data-processing inequality, the divergence between the unsmoothed
coarse-grained distributions is **less than or equal to** the full-vocabulary
divergence. The implementation floors non-empty tails and renormalizes them;
that epsilon-smoothed objective is close but not covered by the theorem.
Property tests exercise the shipped objective over generated inputs, while the
`k == V` identity path bypasses smoothing and reproduces the exact objective.

Consequences you should not paper over:

- A bucketed KL number is not comparable to a full-vocabulary KL number from
  another codebase, and is not comparable across different `top_k`.
- The reported teacher entropy is `bucketed_teacher_entropy`, the entropy of the
  `K + 1` bucket distribution. It lower-bounds the true entropy for the same
  reason, because merging the tail discards its internal spread.
- Selecting fewer positions reduces the LM-head projection work, the cache size
  and the number of loss positions. It does **not** proportionally reduce teacher
  FLOPs, because the teacher still runs a full forward pass over the whole
  sequence to produce hidden states. The reports name this
  `teacher_queried_position_ratio` and never call it compute saved.

### `tail_epsilon` changes the objective

Both tails are floored at `log(tail_epsilon)` (default `1e-9`) and the
`K + 1`-way vectors are then renormalized. Without the floor, a teacher whose
top-k captures all of the mass combined with a student that still leaks
probability outside that top-k gives `+inf` in reverse KL.

The floor is not free. Every bucket probability is perturbed by O(`tail_epsilon`)
by the flooring and renormalization, so the minimized quantity is not exactly
the bucketed divergence either. More importantly, the reverse-KL penalty for
student mass outside the teacher's top-k is capped at `log(1 / tail_epsilon)`
nats, which at the default is about 20.7 nats rather than infinity
(`test_reverse_kl_tail_penalty_is_bounded_by_log_one_over_epsilon`). At the
default that bound is not the binding one: the dtype-aware clamp inside
`log1mexp` floors a float32 teacher tail near
`torch.finfo(torch.float32).eps`, so the effective cap is about 15.94 nats.
[math.md](math.md#54-the-bound-the-floor-buys) derives and measures both. Raising
`tail_epsilon` weakens that penalty further; lowering it makes the gradient
sharper and the loss noisier. Treat it as a hyperparameter of the objective, not
as a numerical detail.

## Training loop

### Historical reserved-memory numbers are order-sensitive

Published v0.1/v0.2 benchmark arms recorded `peak_allocated_bytes` while their
live tensors were present, but their `peak_reserved_bytes` can include CUDA
caching-allocator blocks retained from an earlier arm in the same process.
Those JSON files are preserved because the task-success measurements remain
valid; they are not a clean cross-arm reserved-memory comparison.

The corrected harness gives the cold start and every arm a function-level
trainer lifetime, destructively drops model, scorer, rollout-runner and
optimizer references, runs garbage collection, and empties the CUDA cache
before constructing the next trainer. Use a new lifecycle-isolated run for
memory comparisons rather than silently rewriting the historical artifact.

### Update batching is not rollout batching

The update path supports typed padded trajectory batches with causal attention
masks, selected-position flattening and per-trajectory normalization.
`train.gradient_accumulation_steps` is still the optimizer-group size;
`train.trajectory_batch_size` only controls how many trajectories in that group
share one physical backbone forward. `auto` pads the whole group and can be
slower or use more memory when lengths differ. Sequential physical batching
remains the compatibility default.

Rollout decoding is still one sequence at a time. There is no continuous
batching scheduler, vLLM/SGLang engine or asynchronous actor pool, so update
speedups must not be described as rollout-throughput improvements.

If you set `gradient_accumulation_steps < rollouts_per_cycle` in `opd` mode,
the default `train.opd_freshness: strict` rejects the config: later optimizer
steps would consume trajectories sampled from a policy that has already been
updated. Explicit `opd_freshness: replay` allows the schedule but labels it
`online_distillation_with_replay` and records the rollout policy version. The
shipped 16 GB recipe sets `gradient_accumulation_steps: 6` equal to
`rollouts_per_cycle: 6` for strict freshness.

### `swap` is unavailable whenever anything is quantized

`memory.strategy: swap` moves the student and its optimizer state to host memory
while the teacher scores, then swaps back. bitsandbytes 4-bit and 8-bit
parameters are pinned to the device they were quantized on and cannot make that
round trip, so `OPDTrainer.from_config` raises `ConfigError` for
`swap` plus any non-`none` quantization, and `auto` resolves to `resident` with
the recorded reason:

> auto -> resident: a quantized model cannot be moved off the accelerator, so
> swap is unavailable

`shared_backbone` lowers duplicated-base memory only when actor, teacher and
optional reference all use the same base identity and runtime settings. It is
also resident. It cannot share a 0.6B student base with a 1.7B teacher, and it
does not turn an arbitrary larger teacher into a one-base configuration.

This removes the main lever for fitting a large teacher on a small card exactly
when you need it. On 16 GB the practical envelope is a quantized sub-1B student
with a bf16 teacher of roughly 1.7B parameters, both resident. A bigger teacher
means dropping quantization to regain `swap`, which costs student memory, or
using a different tool.

### OOM handling only halves the chunk

`run_with_oom_retry` retries a CUDA OOM by halving `loss.chunk_size` down to
`memory.min_chunk_size`, at most `memory.oom_retries` times. That is deliberate:
the chunk size is mathematically neutral. Nothing else is adjusted
automatically. Sequence length, batch size, models and objective are never
changed behind your back, so a run that does not fit will fail rather than
quietly train something different.

## Evidence quality

### The toy backend is a machinery harness, not a capability result

`recipes/toy_cpu.yaml` runs the whole pipeline on CPU with a 186-entry
reversible tokenizer and a small RMSNorm/RoPE/SwiGLU transformer (student:
hidden 96, 3 layers). Its purpose is to exercise the loop end to end.
`recipes/toy_exact_full_vocab.yaml` additionally exercises
`loss.mode: exact_full_vocab`, which is affordable only at a vocabulary this
small. Neither is evidence that distillation works.

The toy measurements recorded during development make the point:

| finding | measurement |
| --- | --- |
| Learned absolute position embeddings block copying | toy teacher reached train loss 0.0006 but **0%** eval success; every rollout had valid syntax and the wrong operands |
| RoPE plus task diversity fixes it | 48 oracle traces gives **25%** eval success; 256 traces at 800 steps gives **87.5%** |
| Too little diversity for the step budget | 1024 traces at the same 800 steps gives **18.8%**, only about 3 epochs |
| Toy student SFT convergence (hidden 96, 3 layers, batch 8) | step 100: 0%, 200: 25%, 300: 75%, 400: 100%, 600: 87.5%; whole run 41 s on CPU |

Note the non-monotonicity in the last row: 400 steps scored higher than 600. The
evaluation sets behind these numbers are small, which is visible in the values
themselves -- 25%, 75% and 87.5% are all coarse fractions -- so a single task
moves the figure by more than ten percentage points and none of these
differences are separable from noise. Do not quote toy numbers as accuracy
results.

### RecoveryBench finds no fresh-state advantage in one scoped setting

RecoveryBench v1 is a preregistered mechanism study on one Qwen3 student and
teacher pair, one read-only SQLite recovery environment, three seeds and one
RTX 4080. It is not an alignment benchmark. Under eight equal continuation
updates, frozen-student-state KD reached 23.2% strict success and 22.8%
recovery after error; strict fresh-state OPD reached 10.9% and 9.1%. The paired
fresh-minus-frozen differences were -12.24 and -13.79 percentage points. Fresh
OPD also averaged 686.8 continuation seconds versus 52.1 for frozen KD.

The result does not establish that OPD is universally ineffective, that
offline KD always wins, or that task accuracy is a sufficient alignment
endpoint. Seed variation is large, the qualified teacher is not universally
competent, task-paired bootstrap intervals condition on this fixed test set,
and three seeds alone do not justify a broad significance claim. The
budget-50 arm queried 49.77% of model-generated positions but did not reduce
teacher backbone forwards or wall time.

The nominal 50-second view has an additional limitation. Fresh OPD crossed the
target in one indivisible update, while SFT and frozen KD completed the
configured eight-cycle ceiling before their internal continuation timers
crossed it. It is therefore a cycle-capped wall diagnostic, not exact
equal-time evidence. The completed artifact is preserved as run rather than
post-hoc replaced. See the [data-bound report](recoverybench/recoverybench-v1.md)
and [technical PDF](https://github.com/DaoyuanLi2816/mini-verl/blob/main/paper/recoverybench-v1/recoverybench-v1.pdf).

### Alignment Lab starts from a saturated policy

Alignment Lab v1 uses one Qwen3-0.6B SFT checkpoint, one deterministic
Minipolicy tool-policy suite, three seeds and one RTX 4080. The starting SFT
policy scores 100% alignment and retains 100% tool utility in every seed.
Consequently, the experiment can detect regressions and cost differences but
has no headroom to establish an incremental quality improvement.

DPO and offline soft distillation tie the start. Continued SFT, standard OPD
and verifier-gated OPD each contain completed safe-error-recovery regressions.
All methods record 0% harmful compliance and 0% over-refusal, so those two axes
alone miss the observed benign-utility failure. This is evidence about metric
coverage in this suite, not a broad safety result.

The matched State × Supervision artifact measures teacher signal, not four new
trained outcomes. It finds 100% teacher-argmax/student-token agreement and only
0.0251% mean fresh soft probability mass beyond argmax under matched states,
teacher, budget, starting checkpoint and seeds. No hard/soft quality advantage
is claimed. The verifier gate reduces selected positions from 100% to 46.8%,
but selected positions are not teacher-backbone FLOPs and the quality result
does not improve.

External IFEval-, XSTest-, HarmBench- and RewardBench-style entries are pinned
metadata adapters only in this result. Their datasets or evaluators were not
executed. The reported preference metric is the deterministic suite's fixed
paired policy outcome, not a general human-preference claim. The pilot's
`insufficient_evidence` recommendation is therefore scoped to this recipe and
must not be generalized to other teachers, policies, models or hardware.

### The 16 GB run demonstrates the pipeline, not an OPD-over-SFT win

`recipes/qwen_consumer_gpu_calc_raw_teacher.yaml` (run id
`rtx4080-calc-opd`) completed 16
optimizer steps in 481.1 s, and held-out greedy evaluation on 12 calculator
tasks went from 0.0% to 100.0%.

That headline is misleading unless you read the next sentence. The recipe runs
`sft_warmup_cycles: 8` before `cycles: 8` of OPD, and **the first OPD rollout
batch already scored 83.3%**. The supervised cold start did most of the work.
The run shows that rollouts, teacher scoring, cache round-tripping, the
divergence and the optimizer step all function together on a consumer card. It
does not show that on-policy distillation beat SFT, and no arm of it was
designed to test that.

### Protocol alignment prevents collapse but does not beat SFT

The v0.2 tool prompt is now named protocol `v1` and remains byte-stable for
historical artifacts. Its full and compact prompt examples were ambiguous:
the final-answer placeholder looked like a literal wrapper even though the
calculator verifier expects the numeric payload inside `<final>`. Protocol
`v2` derives tool examples from the active `ToolSpec` and uses an
environment-specific final example whose format is accepted by that
environment's verifier; both examples round-trip through the parser. Existing
adapters and recipes stay on explicit `v1`; miniVERL refuses to use a
competence-gated adapter under a different requested protocol. This versioning
fix prevents silent reinterpretation, but it does not rewrite the frozen v0.2
benchmark or adapter.

The primary schema-v2 comparison uses an explicit equal-optimizer-update axis,
two prespecified seeds and the `hard` split:

| arm | seed 1234 | seed 20260727 |
| --- | ---: | ---: |
| cold-start-only | 75.0% | 75.0% |
| sft-continued | 100.0% | 100.0% |
| opd-raw-teacher | 0.0% | 0.0% |
| opd-privileged-context | 0.0% | 0.0% |
| opd-protocol-sft-teacher | 100.0% | 100.0% |

The protocol-trained teacher passed an independently prespecified competence
gate before the OPD benchmark was inspected. Its arm eliminates the collapse
seen with both protocol-naive teachers, supporting the legacy transcript
diagnosis. It still only ties SFT. This repository therefore has evidence that
teacher protocol competence matters for this setup, but no evidence that OPD
outperforms verified supervised continuation.

There is an additional selection caveat. Candidate A was prespecified, passed
the 50% gate on its first run, and therefore triggered no fallback tuning.
However, the teacher gate used the same 24-task v0.2 `test` set that was later
used for downstream OPD reporting. The downstream OPD outcome was not consulted
when selecting the teacher, but the final task set was not completely untouched.
The historical recipe and grid retain that fact. Future protocol-teacher
selection must evaluate on `eval`; downstream comparisons must report `test`.

### Teacher preparation is part of the cost

The measured continuation-only protocol-OPD cost is **523.8 s** per student run
on the RTX 4080 (mean over the two v0.2 seeds). Preparing Candidate A took
**554.9 s** once. On the same measured wall-time basis:

| accounting view | seconds per student run |
| --- | ---: |
| continuation only, reusing the published teacher | 523.8 |
| teacher preparation + one continuation | 1,078.7 |
| teacher reused across 5 student runs | 634.8 amortized |
| teacher reused across 10 student runs | 579.3 amortized |

These figures add measured training times; they exclude download, export and
human selection time. Teacher preparation is reusable, not free.

### Support claims follow the test matrix

| surface | evidence |
| --- | --- |
| torch-free core | Python 3.10, 3.11, 3.12 and 3.13 in CI |
| full CPU ML suite | Python 3.12 in CI |
| no-network toy training | oldest/newest training-stack Python rows in CI |
| Transformers | 4.51.x and supported 5.x rows on Python 3.12 |
| GPU paths | opt-in workflow plus local RTX 4080 evidence on Python 3.12 |

The core Python matrix is not a claim that every torch/PEFT/bitsandbytes build
supports every one of those interpreters.

The [schema-v2 result](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/gpu-calc-hard-equal-update-v2.json)
contains both seeds and every arm's complete resolved-config diff. The preserved
[legacy result and transcript diagnosis](rtx4080-baselines.md#legacy-equal-update-comparison-schema-v1)
show how the raw teacher corrupted formatting while the privileged teacher
learned to skip the tool.

**What a reader should take from this.** Use miniVERL to run the experiment, not
as evidence of a result. If your teacher is not already competent at your task's
output format, expect on-policy distillation to degrade a cold-started policy,
and measure it against a supervised arm on an explicitly declared comparison
axis before believing otherwise. `miniverl benchmark` exists to make that arm
cheap to add and to record quantities that cannot be matched by construction.

The task family is also close to saturated. At `difficulty: medium` the
calculator environment emits either a three-term arithmetic expression or a
single unit conversion (`CalculatorEnvironment.generate_task`), each solvable
with one tool call. There is no headroom above 100% and very little between the
cold start and the ceiling. Any comparison of objectives on this task will be
measuring the wrong thing; use `difficulty: hard`, which adds four-term
expressions and chained compute-then-convert tasks, or a different environment.

### Two seeds, one machine, one task family

The primary v0.2 GPU comparison repeats all five arms at two prespecified seeds;
older GPU artifacts remain single-seed measurements. Every run used the same RTX
4080 and calculator task family. Two identical outcomes are useful replication,
not a confidence interval or significance test, and no claim in this project
generalizes them to another task, model pair, budget or hardware platform.

### Throughput numbers are platform-specific and mostly measure kernel launches

Single-sequence decode on this machine (64 new tokens from a 36-token prefix):

| configuration | tokens/s | peak allocated |
| --- | --- | --- |
| NF4 + bf16, deterministic algorithms | 11.19 | 0.862 GiB |
| NF4 + bf16, non-deterministic | 11.29 | - |
| bf16 LoRA, deterministic algorithms | 12.84 | 1.170 GiB |
| bf16 LoRA, non-deterministic | 14.12 | - |

The interesting number is not in that table. A 14-token prefill costs 37.0 ms
while a cached single-token step costs 30.9 ms. The work differs by 14x and the
time differs by 20%, so decoding here is bound by kernel launch overhead, not by
compute. For context, the LM head on one position takes 0.48 ms, decoding 64
token ids takes 0.02 ms, and copying a 151936-float vector from device to host
takes 0.11 ms.

That means these throughput figures characterize this Windows/CUDA/torch
combination as much as they characterize miniVERL. On a machine with lower
launch overhead, or with CUDA graphs, or with a batched rollout engine, the
ranking between NF4 and bf16 could change. Do not size a recipe for other
hardware from this table.

Memory, separately measured by `scripts/gpu_smoke.py` (1 SFT warmup cycle, 1 OPD
cycle, 2 rollouts, 2 eval tasks): peak 4.251 GiB allocated and 4.762 GiB
reserved, strategy resolved to `resident`, projection chunk 256, 0 OOM retries.
The student had 10,092,544 trainable LoRA parameters out of 385,941,504
NF4-packed parameters; the teacher had 1,720,574,976. A longer run with larger
budgets will not have this peak.

## Scope

### Only two architectures are tested

`miniverl.models.adapters.TESTED_ARCHITECTURES` is
`("Qwen3ForCausalLM", "Qwen2ForCausalLM")`. The adapter resolves the decoder
backbone and the LM head through documented Transformers APIs
(`get_decoder`, `get_output_embeddings`, `get_base_model`) with a short list of
attribute-path fallbacks, so other decoder-only causal LMs may work. They are
untested, and a model whose head or backbone is not reachable raises with the
class name rather than guessing.

Models with tied embeddings, weight-sharing tricks, or a non-`Linear` output
projection have not been exercised beyond the pinned Qwen3 pair (which does have
`tie_word_embeddings: true`) and a tiny offline `Qwen3ForCausalLM` fixture.

### Things this project does not contain

miniVERL has no distributed training, multi-GPU or multi-node execution,
high-throughput rollout engine, PPO/GRPO objective, trained reward model,
vision-language support, cross-tokenizer distillation, or containerized or
networked tool sandbox. The three built-in environments still generate their
tasks in-process; the v0.6 Parquet converter only exchanges validated prompt
datasets and does not turn arbitrary verl datasets into those environments.
Tools execute in-process. Where that is risky the environment restricts the
input rather than the process: the calculator walks a parsed `ast` with a closed
node whitelist instead of calling `eval`, and the SQLite environment uses a
`sqlite3` authorizer plus a function whitelist and permits one statement per
call.

### The verified verl bridge does not execute verl

The v0.6 **miniVERL-defined compatibility Level 3** bridge targets official verl
`v0.8.0` at one exact commit and one named profile. It validates standard
PEFT/safetensors/tokenizer artifacts, Parquet prompt data, 14 whitelisted
config fields, a safe reward scaffold and bundle hashes. The recorded smoke
installs that exact source and parses or loads each exchange surface.

It does **not** launch Ray, FSDP/Megatron, vLLM/SGLang or a distributed training
job. It does not convert optimizer state, distributed RNG, native sharded
checkpoints, Ray runtime state or teacher caches into PPO reference caches.
Unknown and distributed-only config fields fail by default. The
miniVERL-defined label therefore means a validated scale-out bundle, not
runtime parity or generic verl YAML support. Current bundles are not
launchable: the base snapshot is absent, reward logic fails closed and user
mappings remain placeholders. See the [exact contract and evidence](verl-bridge.md).

`models.teacher.mode: privileged_context` works only with environments that
implement `privileged_context()`. All three built-in environments do;
`miniverl validate` warns if you point it at one that does not.

### Portable redaction is not a secret store

Shareable views structurally redact credential-like keys, URL userinfo and
embedded absolute paths across the supported report and export formats. That is
a best-effort defense against accidental disclosure, not a guarantee over every
possible encoding or future field name. Do not place real credentials in
configs, run directories, exception messages or reports.

### Cache precision

`cache.dtype: float16` (used by the 16 GB recipe) halves the log-probability
payload at a cost of roughly 1e-3 relative precision on the stored teacher
log-probabilities. `float32` round-trips exactly. If you are comparing
divergence magnitudes across runs, keep this fixed.

## A defect this page used to describe, now fixed

An earlier draft of this page documented an open defect that Hypothesis found in
`miniverl.losses.reduction.weighted_mean`: the denominator was clamped at
`MIN_TOTAL_WEIGHT = 1e-12`, so a weight sum that was non-zero but far below that
floor produced an inflated result rather than the true weighted mean. The
counterexample was `values = [0, 0, 0, 0, 1]` with
`weights = [0, 0, 0, 0, 2.22e-16]`, where `2.22e-16 / 1e-12` gives `2.22e-4`
instead of `1.0`.

It is fixed. The floor now applies only when the weights sum to **exactly**
zero, in which case the numerator is exactly zero too and the loss is a clean
`0.0`. For any positive weight sum the true mean is computed and is bounded by
the range of the values, so no rescaling can occur. The same change was applied
to the chunked path in `miniverl.losses.chunked`.
`tests/property/test_property_losses.py::test_weighted_mean_is_a_weighted_mean`
now asserts both branches of that contract.

The record is kept here rather than deleted, because "a property test found a
real numerical defect and it was fixed" is more informative than silence.

## Gate results

The release has 1,000+ tests and 86%+ branch coverage. Exact current counts,
commit identity, date and opt-in GPU/network status live in the single
machine-readable [release-quality record](generated/quality.json), so this page
does not drift when tests are added.

## Roadmap (not implemented)

Nothing in this section exists in the code. It is recorded here so that the
sections above cannot be read as implying it does.

- **Cross-tokenizer distillation.** Referenced as a roadmap item by the error
  hints in `models/factory.py` and `trajectory/alignment.py`. Would require a
  token-alignment layer between the two vocabularies. Not implemented.
- **Entropy-aware forward/reverse KL mixing.** miniVERL records per-position
  teacher entropy and reports it, which is the input such a scheme needs, but
  the divergence is whichever single one `loss.divergence` names. Mixing forward
  KL into high-entropy positions, as in arXiv:2603.07079, is **not implemented**.
- **Swap for quantized models.** Requires dequantize-on-eviction or a second
  copy of the weights. Not implemented, and rejected at config time today.
- **Native multi-GPU or multi-node execution.** Not implemented. The v0.6
  bridge can export one documented profile to pinned verl, where scale-out
  remains the user's separately reviewed and executed operation.
- **A batched or engine-backed rollout path.** Not implemented.
- **Additional tested architectures.** Only Qwen3 and Qwen2 are tested today.
