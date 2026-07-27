# How miniVERL compares

This page exists so that you can decide quickly whether miniVERL is the wrong
tool for your problem. For most serious distillation work it is.

Every cell below is either something read out of the relevant source tree, a
value returned by the GitHub or arXiv API, or the literal string `not verified`.
Nothing here is inferred from a project's marketing copy. Checks were performed
on 2026-07-27; upstream projects move, so re-check before relying on a row.

## Feature table

| dimension | miniVERL | verl | TRL GKD | KDFlow | OPSD | plain SFT |
| --- | --- | --- | --- | --- | --- | --- |
| **Primary scope** | Single-accelerator on-policy distillation of a tool-using agent, plus SFT and offline KD, over three built-in deterministic environments | General RL post-training framework (HybridFlow). PPO/GRPO family, plus a first-class distillation trainer in `verl/trainer/distillation/` | One trainer, `trl.experimental.gkd.GKDTrainer`, doing generalized-JSD distillation on chat datasets | Knowledge-distillation framework: off-policy, on-policy, cross-tokenizer and multimodal, per the repository description | Research harness for on-policy distillation experiments, built on verl | Next-token cross-entropy on a fixed dataset. miniVERL implements this itself as `run.mode: sft`, for use as a baseline arm |
| **Mandatory infrastructure** | None beyond a Python process. `doctor`, `validate`, `inspect`, `report`, `cache`, `schema` and `export-benchmark` run on the base install; torch, transformers and peft arrive only with the `train` extra | Ray. `ray[default]` is an unconditional line in `requirements.txt` | The `transformers` Trainer plus `accelerate`. No cluster runtime | Ray and SGLang, both unconditional in `requirements.txt` | Whatever verl needs, therefore Ray. Setup is a shell script (`scripts/opd/setup_opd.sh`), not a package install | None |
| **Target hardware** | One accelerator, or CPU for the toy backend. Every GPU number in this repository comes from a single RTX 4080 (16 GB). There is no multi-GPU code path | The published device-tuning table starts at 1xH100 and runs through 8xH100, 8xH800 and 32xH800 / 32xH20 | Not stated in the trainer documentation; scales with whatever `accelerate` is configured for | Examples assume 8 GPUs per node | not verified; inherits verl's requirements | Any |
| **On-policy rollouts** | Yes. `run.mode: opd` samples from the current student every cycle, and the config validator forces `cache.strict_policy_version: true`, so a target produced at policy version *v* cannot be consumed at *v+1* | Yes | Configurable, not the default. `lmbda` defaults to `0.5`, so on average half the batches use student-generated sequences | Yes | Yes | No. The trajectory set is fixed before training starts |
| **Multi-turn tool use with environment execution** | Yes. `RolloutRunner` interleaves real tool execution (calculator, JSON navigation, read-only SQLite) with generation, bounded by `max_turns`, `max_new_tokens_per_turn`, `max_total_tokens`, `max_parse_errors` and `max_repeated_calls` | Yes. `verl/experimental/agent_loop/` contains `tool_agent_loop.py` and `tool_parser.py` | No tool environment in the distillation trainers | None found. A code search over the repository for `tool_call` and for `agent` returns 0 results | Yes. The README describes multi-turn agent-loop rollouts with tool and environment tokens excluded from the loss | No |
| **Teacher-target compression / cache** | `bucketed_topk_tail` (teacher top-k log-probs plus one aggregated tail bucket) or `exact_full_vocab`. Bucketed targets persist to a sharded safetensors cache with a SHA-256 per shard and a recorded teacher revision, tokenizer fingerprint, temperature and `top_k` | not verified | `GKDTrainer` recomputes full-vocabulary teacher logits under `no_grad` every step and keeps no cache. Separately, `ServerDistillationTrainer` exposes `loss_top_k` (default `1`) and `loss_add_tail` (default `True`) | not verified | Chunked divergence computation rather than a persistent cache; the README states it processes tokens in chunks instead of materializing full-vocabulary tensors for the whole batch | Not applicable; the targets are the tokens |
| **Packaging and tests** | hatchling wheel and sdist, `miniverl` console script, 933 tests (929 run in CPU CI, 4 marked `gpu`), all passing, at 85% total branch coverage; ruff, ruff format and mypy clean | Published on PyPI as `verl`; 51 workflow files under `.github/workflows` | Published on PyPI as `trl`; tests in-repo, including `tests/experimental/test_server_distillation_trainer.py` | Has a `pyproject.toml`, but no PyPI release under the name `kdflow` (404), no `tests/` directory and no `.github/workflows` | The repository root holds only `README.md`, `scripts/` and `src/`: no `pyproject.toml`, no `tests/`, no `.github/workflows` | Not applicable |
| **License** | Apache-2.0 (`LICENSE` in the repository root) | Apache-2.0 | Apache-2.0 | MIT | No `LICENSE` file; the GitHub API reports no license, so the code is all-rights-reserved and cannot be reused | Not applicable |

One more research codebase is worth naming even though it does not get a column:
**thunlp/OPD**, the official code for arXiv:2604.13016, also builds on verl
(v0.7.0) with LlamaFactory for the SFT stage, runs its experiments on 8xA800
80 GB, and likewise ships **no LICENSE file**.

## What is not novel here

Two claims that a reader might expect this project to make, and which would be
false:

- **Top-k plus a tail bucket is not a new idea.** TRL's
  `ServerDistillationTrainer` already has `loss_top_k` (default `1`) and
  `loss_add_tail` (default `True`), which is the same coarse-graining. miniVERL's
  contribution on this axis is bookkeeping, not the objective: the bucket
  parameters are recorded in the cache index and re-checked on load, and the
  documentation is explicit that the result is a lower bound on the exact
  divergence rather than the divergence itself.
- **verl is not missing on-policy distillation, and it is not missing tool
  use.** verl has `verl/trainer/distillation/` with FSDP and Megatron backends
  and a `DistillationConfig` in `verl.workers.config`, and it has an agent loop
  with a tool parser in `verl/experimental/agent_loop/`. Any comparison that
  presents miniVERL as filling those gaps is wrong.

## What miniVERL is not

miniVERL is not a scaling framework, and it is not a faster or better
implementation of anything listed above. It has no Ray integration, no FSDP or
Megatron backend, no vLLM or SGLang rollout engine, no tensor or pipeline
parallelism, and no multi-node story of any kind. Rollouts come from a custom
sampling loop that decodes one sequence at a time with a KV cache, projecting a
single position through the LM head per step, and the update path processes one
trajectory per forward pass with no padded batching, so
`train.gradient_accumulation_steps` is the effective batch size. There is no
PPO, no GRPO and no reward model. The three environments are synthetic and
generated in-process; there is no dataset loader, no containerized or networked
tool sandbox, and no support for vision-language models. Capability results in
this repository come from one calculator task family on one consumer GPU at one
seed, which is enough to show the pipeline runs and is not enough to establish
that any objective beats any other.

It is a single-GPU teaching and experimentation lab: a readable implementation
of the exact and bucketed divergences with the orientation written into every
function name, a rollout loop that records token provenance as it generates
rather than reconstructing it afterwards, a teacher cache whose policy version
is enforced rather than assumed, and a CLI that refuses contradictory recipes
before anything is downloaded.

## When you should use verl instead

Use verl, not miniVERL, if any of the following is true:

- You have more than one GPU, or more than one node. verl was built for this;
  miniVERL has no code path for it at all.
- Your student is larger than roughly a 1B-parameter model, or your teacher does
  not fit in the memory left over after the student, its LoRA adapters and its
  optimizer state. miniVERL's `swap` strategy exists for that case but is
  unavailable when either model is quantized, which is exactly the configuration
  a 16 GB card pushes you toward.
- You need rollout throughput. verl drives vLLM or SGLang; miniVERL decodes one
  sequence at a time, and on the machine used here that is kernel-launch bound
  rather than compute bound.
- You want on-policy distillation combined with RL objectives, a reward model,
  or an advantage estimator. verl has the PPO and GRPO machinery and a
  distillation trainer that plugs into it.
- You want something with a maintainer community, a release cadence and 51 CI
  workflows behind it.

miniVERL is a reasonable choice when you have exactly one consumer GPU, you want
to read and modify the loss, and you care more about being able to audit which
token was supervised by which teacher distribution than about throughput.
Reading `docs/limitations.md` before starting is strongly recommended.
