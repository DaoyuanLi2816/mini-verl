# PROJECT_STATE

Living build log for **miniVERL** (`mini-verl` / `miniverl`). A checkbox is not
evidence: every completed item below names the command that was run and what it
printed.

Last updated: 2026-07-27.

## Current phase

Phase 4 -- CLI and reports. Phases 1-3 (numerics, environments/toy backend,
Hugging Face + QLoRA backend) are implemented and covered by executed tests.

## Environment (measured)

| item | value | how |
| --- | --- | --- |
| OS | Windows 11 Pro 10.0.22631 | `platform.platform()` |
| GPU | NVIDIA GeForce RTX 4080, 16376 MiB, driver 596.49 | `nvidia-smi --query-gpu=name,memory.total,driver_version` |
| dev interpreter | CPython 3.12.13 (`.venv`, uv-managed) | `uv venv --python 3.12` |
| torch | 2.13.0+cu130, `torch.cuda.is_available() == True` | `python -c "import torch; ..."` |
| transformers | 5.14.1 | `importlib.metadata.version` |
| peft | 0.19.1 | idem |
| bitsandbytes | **not installed yet** | `ModuleNotFoundError` |
| PyPI name `miniverl` | AVAILABLE (HTTP 404 on `/pypi/miniverl/json`) | `Invoke-WebRequest` 2026-07-27 |

## Verified model pair (pinned)

Confirmed twice against `huggingface.co/api/models/...` plus a negative control
(an all-zero SHA returns 404, so the endpoint really validates revisions):

| role | model | revision | notes |
| --- | --- | --- | --- |
| student | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0, 28 layers, hidden 1024 |
| teacher | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0, 28 layers, hidden 2048 |

`tokenizer.json` is **byte-identical** across the pair
(sha256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`),
vocab_size 151936, `tokenizer_class: Qwen2Tokenizer`, `tie_word_embeddings: true`
on both. Qwen3 requires `transformers >= 4.51`.

## Completed, with evidence

### Numerical core

* `pytest tests/unit/test_losses_exact.py -q` -> **21 passed**. Includes
  brute-force Python references for forward KL, reverse KL and beta-JSD, an
  orientation test that catches an accidental KL reversal, `log 2` bound on
  symmetric JSD, finiteness at logit scale 1e4, finite gradients, fp16/bf16
  upcast, and the `T**2` scaling contract.
* `pytest tests/unit/test_losses_bucketed.py -q` -> **30 passed**. Includes
  `log1mexp` against a reference, tail-mass validity, `k == V` converging to the
  exact loss (atol 1e-5), the **data-processing inequality** bound
  (bucketed <= exact for k in {1,2,8,32} and all three divergences),
  monotonicity in k, and a bounded reverse-KL tail penalty when the teacher's
  top-k captures all mass.
* `pytest tests/unit/test_chunked_equivalence.py -q` -> **22 passed**. The
  two-stage detach/backward trick reproduces the unchunked **gradient** for
  chunk sizes 1, 5 and 37 across all three divergences (atol 1e-5); zero-weight
  positions provably contribute nothing to loss or gradient; CE mixing is a
  convex combination; `loss_scale` affects gradients only.

### Environments

Executed script over all three environments at all three difficulties:

* oracle solves **6/6** train tasks for every (environment, difficulty) pair;
* train/eval/test prompt sets are disjoint by construction;
* SQLite lockdown, all **BLOCKED**: `DROP TABLE`, `INSERT`, `UPDATE`,
  `ATTACH DATABASE`, `PRAGMA`, `SELECT * FROM sqlite_master` (authorizer),
  `SELECT randomblob(10)` (function whitelist), `SELECT 1; SELECT 2`
  (one statement per call). `SELECT count(*) FROM orders` returns
  `[{"n": 14}]`.

### Backends

* Toy backend: reversible ~190-entry tokenizer
  (`decode(encode(x)) == x` asserted), RMSNorm + RoPE + SwiGLU transformer.
* Hugging Face backend, **offline** with a tiny `Qwen3ForCausalLM`
  (`vocab_size=186, hidden=32, 2 layers`): architecture adapter resolves
  `Qwen3Model` / `Linear` head through a PEFT wrapper, selected-position
  projection returns `[3, 186]`, backward reaches **8 LoRA tensors / 896
  params** and nothing else.

### End-to-end toy pipeline

`OPDTrainer.from_config(RunConfig.from_yaml("recipes/toy_cpu.yaml")).train()`
completed in **2.9 s** on the first (small) configuration, producing
`config.original.yaml`, `config.resolved.yaml`, `manifest.json`,
`environment.json`, `metrics.jsonl`, `events.jsonl`, `trajectories.jsonl`,
`eval_trajectories.jsonl`, `teacher-cache/`, `checkpoints/final`, `eval.json`.

### Measured toy-scale findings (these drove design decisions)

| finding | measurement |
| --- | --- |
| Learned absolute position embeddings block copying | teacher train loss 0.0006 (memorized) but **0%** eval success; every rollout had valid syntax and wrong operands |
| RoPE + task diversity fixes it | 48 traces -> **25%**; 256 traces, 800 steps -> **87.5%** eval success |
| Too little diversity per step budget | 1024 traces at the same 800 steps -> **18.8%** (only ~3 epochs) |
| Student SFT convergence (h=96, l=3, batch 8) | step 100: 0% / 200: 25% / 300: 75% / 400: 100% / 600: 87.5%, whole run **41 s** on CPU |

## Currently failing / not yet run

* `ruff`, `ruff format --check`, `mypy` have not been run yet.
* Reporting, CLI, benchmark harness, docs, CI and packaging are not written yet.
* No GPU run yet: `bitsandbytes` is not installed, so the QLoRA path is
  **not run**, not "passing".

## Unresolved design risks

1. **Toy capability vs runtime.** The toy models can reach ~87% on the
   calculator environment but need a few hundred optimizer steps. The toy
   pipeline is therefore positioned as a *machinery* harness; capability claims
   must come from the RTX 4080 recipe. Do not let the README imply otherwise.
2. **transformers 5.14 API drift.** `from_pretrained` dtype keyword is probed
   with `inspect.signature`; the quantization path is untested until
   bitsandbytes is installed.
3. **`exact_full_vocab` + `swap`** must materialize `[N, V]` teacher targets.
   Guarded by `loss.exact_max_vocab` (default 8192) with an actionable error.

## Next three actions

1. Write `miniverl/reporting` (self-contained HTML, inline SVG, no matplotlib)
   and `miniverl/cli.py` with every command in the required UX list.
2. Write the benchmark harness with a shared cold-start checkpoint so all arms
   start identically, then run the toy matched comparison.
3. Install bitsandbytes, then run the Qwen3-0.6B/1.7B recipe on the 4080 and
   record real measurements in `docs/rtx4080-baselines.md`.

## External blockers

* None so far. Network and GPU are both available.

## Release-gate checklist

Filled in as gates actually pass; see the final section of this file.
