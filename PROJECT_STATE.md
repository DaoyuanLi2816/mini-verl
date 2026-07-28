# PROJECT_STATE

Living build log for **miniVERL** (`mini-verl` / `miniverl` / CLI `miniverl`).
A checkbox is not evidence: every completed item names the command that was run
and what it printed.

Last updated: 2026-07-27.

## v0.2 takeover checkpoint (in progress)

This section supersedes stale publication and next-action statements lower in
this living log until the v0.2 branch is merged.

| item | current state |
| --- | --- |
| public repository | `https://github.com/DaoyuanLi2816/mini-verl`; fetched `origin/main` is `3383f2b9a3c595e0fa143fecdc27522ab368b27f` |
| working branch | `v0.2-protocol-aligned-opd`, based on the current public `main`; changes below are not committed yet |
| version | `0.2.0` release candidate; no tag or publication has been created |
| current PR | none yet; open a draft after the first complete, fully tested vertical slice |
| GPU runs | none running; expensive work is gated on CPU/schema/adapter checks |
| PyPI/release | not published, no tag, no upload authorized |

Completed v0.2 work in the current worktree:

- Benchmark schema/config v2 separates common and cold-start overrides, performs
  a declared structured diff before model loading, records resolved hashes and
  cumulative accounting, and retains v1 read/JSON-schema compatibility.
- Strict OPD freshness is the default and rejects multiple optimizer updates
  from one rollout batch; explicit replay is labeled
  `online_distillation_with_replay`.
- The ambiguous public `loss.ce_weight` name is replaced by
  `sampled_token_nll_weight` for distillation; a nonzero legacy name is rejected.
- Frozen standard PEFT teacher adapters can be validated, loaded and exported
  with checksums and provenance. The no-network local Qwen-compatible round trip,
  freeze checks, wrong-base/tokenizer checks and missing-file checks pass.
- `recipes/qwen3_1.7b_protocol_teacher_sft.yaml` validates and its exact
  `miniverl train ... --dry-run --json` command succeeds.
- Policy evaluation now records strict success as primary plus lenient
  diagnostic success, valid tool-call rate/count, final-format validity and
  average turns. Protocol-token accuracy is explicitly `null/not_applicable`
  for unaligned free-running trajectories.
- `k == V` bypasses tail smoothing; temperature-squared claims are narrowed and
  a deterministic 48-cell gradient sweep covers three divergences, four
  temperatures and two logit regimes.
- README source-install and CUDA installation order are corrected in English
  and Chinese.

Evidence executed on this branch:

- `ruff check .` and `ruff format --check .` — clean, 146 files formatted.
- `mypy src/miniverl` — clean, 73 source files.
- complete offline/CPU suite — 961 passed, 4 GPU tests deselected, 85.27%
  branch coverage over 6391 statements.
- Transformers 4.51.3 boundary environment — offline Qwen3/PEFT/config bundle,
  117 passed.
- `python -m build` and `python -m twine check dist/*` — 0.2.0 sdist and wheel
  built, both metadata checks passed.
- committed benchmark schema exactly matches `miniverl schema`; every recipe
  validates and the protocol-teacher dry run reports 24 optimizer steps and 192
  oracle traces.
- focused benchmark/config/cache suite — 129 passed before the adapter slice.
- no-network PEFT adapter suite — 14 passed.
- policy metrics, benchmark v2, packaging, toy pipeline and offline HF bundle —
  119 passed; its only failure was a duplicate test tokenizer ID, corrected and
  rerun as 1 passed.
- loss/property/temperature sweep suite — 45 passed.
- protocol-teacher recipe `validate --json` and `train --dry-run --json` —
  valid, 24 planned optimizer steps and 192 oracle traces.

Current unresolved scientific question: the protocol-trained teacher has not
yet been trained or evaluated, so there is still no evidence that it prevents
the raw-teacher collapse. No claim will be made from SFT loss alone.

Exact next actions:

1. Commit and push the complete CPU-validated vertical slice, open a draft PR,
   and let remote CPU/compatibility checks run while local GPU work proceeds.
2. Run one-cycle Qwen smoke tests, then train/export/evaluate the protocol
   teacher and preserve all raw artifacts with checksums.
3. Run the prespecified equal-optimizer-update arm set for the same seeds,
   generate the result visualization from schema-validated JSON, update the
   draft PR, wait for green CI and integrate it.

## Current phase

Phase 7 -- v0.2 release candidate and protocol-teacher experiment. The v0.1
foundation is complete; the v0.2 CPU slice passes locally and the remaining
work is publication of the draft branch plus the prespecified GPU run.

## Environment (measured, not assumed)

| item | value | how |
| --- | --- | --- |
| OS | Windows 11 Pro 10.0.22631 | `platform.platform()` |
| GPU | NVIDIA GeForce RTX 4080, 16376 MiB, driver 596.49 | `nvidia-smi --query-gpu=name,memory.total,driver_version` |
| dev interpreter | CPython 3.12.13 (`.venv`, uv-managed) | `uv venv --python 3.12` |
| torch | 2.13.0+cu130, `torch.cuda.is_available() == True` | `python -c "import torch; ..."` |
| transformers | 5.14.1 | `importlib.metadata.version` |
| peft | 0.19.1 | idem |
| bitsandbytes | 0.50.0, `bnb.nn.Linear4bit` present | idem |
| PyPI name `miniverl` | AVAILABLE (HTTP 404 on `/pypi/miniverl/json`, 2026-07-27) | `Invoke-WebRequest` |

## Verified model pair (pinned in the recipes)

Confirmed twice against `huggingface.co/api/models/...` with a negative control
(an all-zero SHA returns 404, so the endpoint really validates revisions):

| role | model | revision | notes |
| --- | --- | --- | --- |
| student | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0, 28 layers, hidden 1024 |
| teacher | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0, 28 layers, hidden 2048 |

`tokenizer.json` is byte-identical across the pair
(sha256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`).
`config.vocab_size` is 151936 while `len(tokenizer)` is 151669 -- the vocabulary
is padded. miniVERL sizes the cache and `top_k` from the model's output
dimension, never from the tokenizer; a GPU test asserts both numbers.

## Historical v0.1 release gates (executed on 2026-07-27)

This table is retained as the v0.1 evidence baseline. Current v0.2 evidence is
recorded in the takeover checkpoint above and will replace this table after the
GPU experiment and clean-install verification finish.

| gate | command | result |
| --- | --- | --- |
| lint | `ruff check .` | **clean** |
| format | `ruff format --check .` | **clean, 138 files** |
| types | `mypy src/miniverl` | **clean, 71 source files** |
| tests | `pytest -q -m "not gpu and not network" --cov=miniverl` | **934 passed, 4 deselected, 85% branch coverage** over 5911 statements |
| GPU tests | `pytest -q -m gpu` | **4 passed** |
| build | `python -m build` | sdist + wheel built |
| metadata | `python -m twine check dist/*` | **PASSED** for both |
| wheel completeness | enumerate `src/miniverl` against the wheel | all 77 entries present |
| clean install (core) | wheel into a fresh venv | `--help`, `--version`, `doctor --json` work; torch absent |
| clean install (train) | wheel + `[train]` into a second fresh venv | `doctor` resolves a CPU-only torch and reports GPU training / QLoRA unavailable with the right remedy; both `examples/` scripts run to completion |
| determinism | two `miniverl demo` runs in the clean venv, artifact-by-artifact | checkpoints and trajectories **bitwise identical**; `eval.json` differs only in seconds, throughput, timestamp and run id |
| artifact portability | every `*.json` / `*.jsonl` parsed with `parse_constant` set to reject | **strict-valid** (this gate found 16 bare `NaN` tokens, now fixed) |
| published results | `jsonschema` validate against `benchmarks/schema/` + personal-information scan | both result files valid and clean |
| docs links | every relative markdown link and `#anchor` resolved | **37 files, 0 broken** |
| CLI/docs parity | every `miniverl <cmd>` in the docs matched against the Typer app | all 11 commands documented, no phantom commands |
| unfinished markers | `rg "TODO\|FIXME\|XXX\|HACK\|NotImplementedError" src tests examples scripts` | none |

## Completed, with evidence

### Numerical core

* `tests/unit/test_losses_exact.py` -- **21 passed**. Brute-force Python
  references for forward KL, reverse KL and beta-JSD; an orientation test that
  catches an accidental KL reversal; the `log 2` bound on symmetric JSD;
  finiteness at logit scale 1e4; finite gradients; fp16/bf16 upcast; the `T^2`
  contract.
* `tests/unit/test_losses_bucketed.py` -- **30 passed**. `log1mexp` against a
  reference; tail-mass validity; `k == V` converging to the exact loss;
  the data-processing-inequality bound for k in {1,2,8,32} across all three
  divergences; monotonicity in k; a bounded reverse-KL tail penalty.
* `tests/unit/test_chunked_equivalence.py` -- **22 passed**. The two-stage
  detach/backward reproduces the unchunked **gradient** for chunk sizes 1, 5 and
  37 across all three divergences; zero-weight positions provably contribute
  nothing; CE mixing is convex; `loss_scale` affects gradients only.
* `tests/property/` -- **14 passed** under Hypothesis, including the
  data-processing inequality evaluated in float64 to `1e-9`.

### Provenance and alignment

* `tests/unit/test_token_provenance.py` -- **30 passed**. A mask that marks a
  `tool_result` token trainable is rejected; spans must tile the sequence;
  position 0 can never be a target; privileged-context alignment computes
  per-segment offsets and rejects a target-token mismatch, a length-changed
  segment, a missing segment, a duplicate key and a tokenizer mismatch.

### Environments

Oracle solves **6/6** train tasks for every (environment, difficulty) pair;
train/eval/test prompt sets are pairwise disjoint. SQLite lockdown, all
**BLOCKED**: `DROP`, `INSERT`, `UPDATE`, `ATTACH`, `PRAGMA`,
`SELECT * FROM sqlite_master` (authorizer), `SELECT randomblob(10)` (function
whitelist), two statements in one call.

### End-to-end

* `tests/integration/test_toy_pipeline.py` -- SFT, offline KD and OPD each run
  to completion and write every documented artifact; OPD increments its policy
  version once per cycle; offline KD announces that it is reusing fixed targets;
  no stored trajectory ever marks tool output trainable.
* `tests/integration/test_resume_and_swap.py` -- an interrupted-and-resumed run
  matches an uninterrupted one to `1e-6` parameter-for-parameter; `swap` matches
  `resident` to the same tolerance; checkpoints contain no pickle; the OOM retry
  only halves the chunk and gives up with actionable advice.
* `tests/integration/test_hf_backend_offline.py` -- a real `Qwen3ForCausalLM`
  built from a config (no download): adapter resolution through PEFT,
  selected-position projection matching a full forward to `1e-4`, KV-cache
  generation matching a cacheless reference token for token, and gradients
  reaching **only** the LoRA tensors.

### Real hardware

* **Full recipe**, `recipes/qwen_consumer_gpu_calc.yaml`, run id
  `rtx4080-calc-opd`: 16 optimizer steps in **481.1 s**; held-out greedy success
  **0.0% -> 100.0%** on 12 tasks. The 8-cycle SFT cold start did most of that:
  the first OPD rollout batch already scored 83.3%.
* **One-cycle smoke**: peak **4.251 GiB allocated / 4.762 GiB reserved**;
  strategy resolved to `resident` because a quantized model cannot be moved off
  the accelerator; projection chunk 256; 0 OOM retries; student 10,092,544
  trainable LoRA parameters; teacher 1,720,574,976 parameters.
* **Throughput probe**: NF4 11.19 tok/s, bf16 LoRA 12.84 tok/s (14.12 with
  determinism off). A 14-token prefill costs 37.0 ms and a cached one-token step
  costs 30.9 ms, so decoding here is **kernel-launch bound, not compute bound**.
* **Matched-budget GPU benchmark** (`benchmarks/configs/gpu_calc_hard.yaml`,
  chained `hard` calculator split, single seed, 12 matched optimizer steps from
  one shared cold start). Result published in
  `benchmarks/results/rtx4080-calc-hard-matched.json`:

  | arm | steps | success |
  | --- | --- | --- |
  | cold-start-only | 0 | 62.5% |
  | sft-continued | 12 | **100.0%** |
  | opd-bucketed-k64 | 12 | **0.0%** |
  | opd-privileged-context | 12 | **0.0%** |

  **On-policy distillation lost, and the privileged-context prediction was
  wrong.** Decoding the final evaluations showed the two OPD arms fail
  differently: the standard teacher drops the opening `<tool_call>` tag and
  wraps answers in its own `<answer>` tags (83.3% invalid calls), while the
  privileged teacher emits clean output and no tool calls at all (1.00 turns,
  0.0% invalid) because it knows the answer and so never needs one.
  `scripts/attribute_failures.py` re-scores the collected trajectories with a
  lenient parser and shows formatting explains 4 of 24 failures for the standard
  arm and **none** for the privileged arm, so neither number is a verifier
  artifact. Full write-up in `docs/rtx4080-baselines.md`; both executions of the
  benchmark are reported and nothing was discarded.

* **Matched-budget CPU benchmark** (`recipes/benchmark_calc.yaml`, 7 arms x 2
  seeds, all 14 completed): published in
  `benchmarks/results/cpu-toy-calc-matched.json`. Changing the seed moves one
  arm by up to **58 points** (cold-start-only: 66.7% at seed 1234, 8.3% at seed
  20260727) while changing the objective moves it by at most 4 points at a fixed
  seed. This is a parity check -- it shows all seven arms including
  `exact_full_vocab` run to completion under identical budgets -- and the table
  is in `benchmarks/README.md` so a reader can see that no ranking is available.
  The first attempt of this benchmark restarted the cosine schedule and damaged
  every arm; that attempt is recorded but not published, because it measured a
  harness bug rather than any arm.

### Defects found and fixed during the build

Recorded because each one is a class of bug worth remembering.

| defect | how it was found | fix |
| --- | --- | --- |
| `src/miniverl/models/` excluded from git **and** the wheel by an unanchored `models/` rule in `.gitignore` | installing the wheel into a clean venv | anchored every directory rule; CI now enumerates the source tree against the wheel and fails on any git-ignored source file |
| Rich markup ate `[...]`, so the missing-extra hint printed `pip install "miniverl"` | reading actual CLI output | every dynamic string goes through `rich.markup.escape` |
| A bare install raised `ModuleNotFoundError` instead of naming the extra | a CLI test | the scorer import moved inside the constructor; the CLI pre-checks the stack |
| `exact_full_vocab` rejected a recipe that omitted `top_k` -- the remedy its own error suggested -- and then wrote a `config.resolved.yaml` that could not be re-read | an agent-written config test | consult `model_fields_set`, then normalize to 1 |
| `weighted_mean` rescaled the result for a tiny-but-positive weight sum | Hypothesis | floor only an exactly-zero sum |
| `EnvironmentConfig.name` was a hard-coded regex, making `register()` useless | writing the custom-environment example | validate against the registry |
| `render_final` could emit a block that parsed back truncated | an agent-written protocol test | escape closing tags in JSON, refuse them in plain text |
| `configure_logging(None)` silently downgraded `--log-level DEBUG` | an agent-written utils test | only an explicit level changes the level |
| The report dropped stored rollouts and printed `0.000 GiB` for CPU runs on a GPU machine | an agent-written reporting test | fill the display budget; decide on the run's device |
| Sampling drew from a CUDA tensor with a CPU generator | the first GPU smoke test | always draw on the CPU, which also makes a seed reproduce across devices |
| Resume replayed completed cycles with a different LR schedule | the resume-equivalence test | `train()` continues at the checkpointed cycle; the teacher fit is wrapped in `fork_rng` |
| The toy model could not learn to copy | measuring, not assuming | learned absolute positions -> RoPE; 48 traces -> 256 traces |
| `ArmResult.run_dir` wrote an absolute path, carrying a username and home directory into a file meant for pull requests | a new privacy test over `benchmarks/results/` | a validator reduces any path to its final component |
| The committed JSON schema was never checked against the model it is generated from, though `schema.py` claimed the two could not drift | writing the schema-validation test | a test asserts the committed file equals `json_schema()` |
| 16 bare `NaN` tokens in `metrics.jsonl` and `eval.json`, which strict JSON parsers reject | diffing two identical demo runs | `tokens_per_solved_task` is `null` when nothing was solved, via `schema.finite_or_none`; a test parses every artifact with `parse_constant` set to reject |
| `TerminationReason.ENVIRONMENT_ERROR` was declared but never assigned | auditing for dead public API | a raising tool now ends the policy rollout with it; the oracle path still raises, because a truncated oracle trace would silently degrade every SFT target |
| `log1mexp` clamped at a fixed `-1e-7`, silently saturating tail mass below float32 resolution regardless of dtype | the documentation pass, then Hypothesis in float64 | the clamp derives from `finfo(dtype).eps`; the data-processing-inequality property now models the `1 / q_tail` error amplification instead of using a flat tolerance |
| Both example scripts printed a bare `0.0%` with no context, which reads as a broken example | running them in a clean install | each prints the same "expected, not a capability claim" note that `miniverl demo` prints |

## Currently failing commands

None. Every gate in the table above passes.

## Unresolved design risks

1. **The measured teachers were not taught the protocol, and this remains the
   top open question until the v0.2 GPU run finishes.**
   The GPU benchmark shows that distilling a protocol-cold-started student
   toward a raw instruct teacher *reduces* task success from 62.5% to 0.0% while
   the objective improves monotonically. Giving that teacher privileged access
   to the answer does not fix it -- it removes tool use entirely, because a
   teacher that knows the answer never needs the tool. This is a property of the
   setup rather than a bug in the implementation, but it means the repository
   currently has **no evidence that its headline method helps**, and the README,
   CHANGELOG and `docs/limitations.md` all say so in those terms. The experiment
   that would settle it is now implemented, competence-gated and prespecified
   in `docs/protocol-teacher-grid.md`; it still needs GPU execution.
2. **The toy backend cannot rank methods.** It solves only the `easy` split,
   where SFT saturates. Measured: `medium` and `hard` stay at 0% even after 700
   SFT steps. The CPU benchmark is therefore a parity check, not a ranking.
3. **Single-seed GPU results.** Two seeds on CPU, one on GPU. No significance is
   claimed anywhere. The CPU pair quantifies why this matters: the between-seed
   spread there reaches 58 points against a between-arm spread of 4.
4. **Transformers API drift.** CI now tests the Qwen3 introduction boundary
   (4.51.x) and the supported 5.x major. A future rename beyond the explicit
   `<6` cap would still need a reviewed compatibility change.

## Next three actions

1. Publish the CPU-validated v0.2 slice as a draft PR and monitor its checks.
2. Execute the primary protocol-teacher candidate and only use the two
   prespecified one-variable fallbacks if its strict success misses 50%.
3. Run all five benchmark arms at both prespecified seeds, preserve adverse
   results, update the PR and integrate only after local and remote gates pass.

## External blockers

* **Publishing to PyPI or creating a tag/release.** Not authorized for this v0.2
  iteration. The OIDC job is hard-disabled until a trusted publisher is
  registered and a separate reviewed change explicitly enables publication.
* **A GPU CI runner.** `.github/workflows/gpu.yml` is manual-dispatch only and
  targets a self-hosted `cuda` runner that does not exist for this repository.
  The prespecified GPU tests and benchmark therefore run locally on the RTX
  4080, with artifacts and checksums recorded for review.

## Final evidence table

See the [v0.2 takeover checkpoint](#v02-takeover-checkpoint-in-progress) for
current evidence and the historical table for the v0.1 baseline. Known
limitations are enumerated in `docs/limitations.md`.
