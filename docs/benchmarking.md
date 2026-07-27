# Benchmarking

miniVERL ships a matched-budget benchmark harness. It runs several training
configurations ("arms") that differ only in the keys you tell it to change,
holds everything else constant, and writes one JSON file and one Markdown file
describing both the results and the controls.

Source files behind this document:

- `src/miniverl/evaluation/schema.py` - the config and result schemas
- `src/miniverl/evaluation/benchmark.py` - the harness
- `src/miniverl/evaluation/evaluator.py` - standalone re-evaluation
- `src/miniverl/evaluation/export.py` - sanitized single-run export
- `src/miniverl/agent/loop.py` - `RolloutStats`, where most metrics originate
- `recipes/benchmark_calc.yaml` - the shipped example benchmark

## Every metric miniVERL records, and its exact definition

### Rollout metrics

These come from `RolloutStats` in `src/miniverl/agent/loop.py`. One
`RolloutStats` instance accumulates over a batch of trajectories; `n` below is
`max(rollouts, 1)`.

| field | definition |
| --- | --- |
| `rollouts` | number of trajectories folded in |
| `solved` | trajectories whose `verification.solved` is true |
| `success_rate` | `solved / n` |
| `avg_turns` | `sum(len(trajectory.turns)) / n` |
| `avg_tool_calls` | turns whose recorded tool call is marked valid, divided by `n` |
| `invalid_tool_call_rate` | `invalid_tool_calls / max(tool_calls + invalid_tool_calls, 1)` |
| `generated_tokens` | sum of `trajectory.generated_token_count` |
| `generated_tokens_per_task` | `generated_tokens / n` |
| `tokens_per_solved_task` | `generated_tokens / solved`, or NaN when `solved == 0` |
| `termination_reasons` | counter over `TerminationReason` |
| `failure_categories` | counter over `FailureCategory`, plus `no_final_answer` for trajectories with no verification record |

`invalid_tool_calls` counts two things, both incremented in
`RolloutRunner.rollout`: a turn whose assistant text failed to parse as either
a tool call or a final answer, and a parsed tool call that the environment
rejected (`step.ok` false).

`TerminationReason` is one of `final_answer`, `max_turns`, `max_tokens`,
`parse_error_limit`, `repeated_call_limit`, `environment_error`,
`eos_without_final`. `FailureCategory` is one of `solved`, `wrong_answer`,
`no_final_answer`, `malformed_answer`, `invalid_tool_call`, `unknown_tool`,
`tool_error`, `budget_exhausted`.

### Evaluation payload

`OPDTrainer.evaluate` (`src/miniverl/training/trainer.py`) returns the
`RolloutStats` dictionary plus:

| field | definition |
| --- | --- |
| `tag` | label for this evaluation pass (`baseline`, `final`, `cycleN`, `benchmark-<arm>`, `standalone`) |
| `split` | `train`, `eval` or `test` |
| `tasks` | number of tasks evaluated, capped by `eval.tasks` (falling back to `environment.eval_tasks`) |
| `policy_version` | policy version at evaluation time |
| `global_step` | optimizer steps completed at evaluation time |
| `temperature` | `eval.temperature`; the shipped recipes use `0.0`, which is exact argmax decoding |
| `seconds` | wall clock of the evaluation pass |
| `rollout_tokens_per_second` | `generated_tokens / seconds` |
| `success_by_difficulty` | mean solved rate grouped by `task.difficulty` |
| `memory` | CUDA counters from `miniverl.utils.gpu.snapshot()` |

Evaluation is deterministic per task: the seed for task `i` is
`eval.seed + i`, and greedy decoding at `temperature: 0.0` removes the sampler
entirely.

### Selection metrics

From `aggregate_selection_stats` in `src/miniverl/selection/selectors.py`,
written into each `*_cycle` record in `metrics.jsonl`:

| field | definition |
| --- | --- |
| `total_model_tokens` | model-generated target positions available |
| `selected_model_tokens` | positions actually sent to the teacher |
| `teacher_queried_position_ratio` | `selected_model_tokens / total_model_tokens` |
| `total_critical_tokens` / `selected_critical_tokens` | the same for tool-call and final-answer tokens |
| `selected_by_span_type` | selected positions grouped by span type |

The name `teacher_queried_position_ratio` is deliberate. Reducing selected
positions reduces LM-head projection work, cache size and the number of student
loss positions. It does **not** proportionally reduce teacher FLOPs, because
the teacher still runs a full forward pass over the whole sequence to produce
hidden states. See the module docstring of `selection/selectors.py`.

### Cache metrics

From `CacheCompressionStats` in `src/miniverl/schemas/cache.py`:

| field | definition |
| --- | --- |
| `actual_bytes` | bytes actually written to the teacher-target cache |
| `theoretical_full_logit_bytes` | `num_selected_positions * vocab_size * dtype_bytes_assumed` |
| `compression_ratio` | `theoretical_full_logit_bytes / actual_bytes` |
| `bytes_per_selected_position` | `actual_bytes / num_selected_positions` |

The baseline is a dense `[selected_positions, vocab]` dump, not a
`[batch, seq_len, vocab]` dump. The larger baseline would make the ratio look
better and would not describe anything miniVERL ever writes.

### Memory metrics

`miniverl.utils.gpu.snapshot()` reads `torch.cuda.memory_allocated`,
`memory_reserved`, `max_memory_allocated` and `max_memory_reserved` after a
`torch.cuda.synchronize()`, and reports `cuda_available: false` with zeroed
counters when there is no CUDA device. The benchmark harness takes the maximum
`peak_allocated_bytes` and `peak_reserved_bytes` over all metric records that
report `cuda_available: true`, and reports `None` for both when no record does.

### Per-arm result fields

`ArmResult` in `src/miniverl/evaluation/schema.py` is the full per-arm record:

| field | source |
| --- | --- |
| `name`, `description` | the arm definition in the benchmark YAML |
| `mode` | `run.mode` after merging the arm overrides |
| `seed` | the seed this repetition used |
| `run_id`, `run_dir` | the run directory this arm produced |
| `loss_mode`, `divergence`, `selector` | resolved config values |
| `top_k` | `manifest["objective"]["top_k"]`: the student vocabulary size in `exact_full_vocab` mode, otherwise `min(loss.top_k, vocab_size)` |
| `optimizer_steps` | `TrainResult.global_step` |
| `policy_version` | policy version reached |
| `tasks` ... `tokens_per_solved_task` | the held-out evaluation payload above; `tokens_per_solved_task` is stored as `null` when it was NaN |
| `selected_training_tokens` | `selected_model_tokens` from the last `*_cycle` metrics record; `0` when the arm produced no such record, as with `train.cycles: 0` |
| `teacher_queried_position_ratio` | `selected_model_tokens / total_model_tokens` from the same record; `null` when there is no such record. For an SFT arm this describes supervised oracle-trace positions, not teacher queries, because SFT never calls a teacher |
| `cache_bytes`, `cache_compression_ratio` | teacher-cache stats, `null` when no cache was opened |
| `peak_allocated_bytes`, `peak_reserved_bytes` | maxima described above, `null` on CPU |
| `seconds` | wall clock around this arm's `train()` plus `evaluate()` |
| `baseline_success_rate` | reserved for a shared starting score. The harness currently always writes `null`: `_cold_start` runs with `eval.enabled: false` and returns no score, because the cold start is measured by the `cold-start-only` arm on the benchmark's own split instead |
| `measurement_status` | `measured`; `miniverl export-benchmark` writes `measured_cpu_only` when the run did not use CUDA |

`BenchmarkResult.aggregate()` groups arms by name and reports
`success_rate_mean`, `success_rate_min`, `success_rate_max`, the seed count,
and a `single_seed` boolean. It computes no confidence intervals and no
p-values.

## What "matched budget" means here

`run_benchmark` records a `controlled` dictionary built from the **base**
recipe, before any arm override is applied. These are the quantities the
benchmark asserts were shared:

- `environment`, `difficulty`, `split_seed`, `train_tasks`, `eval_tasks`,
  `test_tasks` - identical task splits, generated prompt-disjointly by
  `make_splits`
- `eval_split`, `eval_temperature`, `eval_seed` - identical held-out evaluation
- `max_trajectory_tokens` (`rollout.max_total_tokens`), `max_turns`
- `effective_batch_trajectories` (`train.gradient_accumulation_steps`),
  `rollouts_per_cycle`
- `optimizer`, `learning_rate`, `lr_schedule`
- `cold_start_cycles`, `cold_start_mode`, `shared_initial_checkpoint`
- `seeds`
- `arms_differ_only_in` - the full override mapping, arm name to override dict

Quantities that cannot be matched by construction are measured and reported per
arm instead of being equalized:

- student generated tokens (`generated_tokens_per_task`,
  `tokens_per_solved_task`)
- selected training tokens (`selected_training_tokens`)
- teacher query ratio (`teacher_queried_position_ratio`)
- teacher-cache size and compression ratio
- peak CUDA memory
- wall clock (`seconds`)

An OPD arm samples its own trajectories, so it cannot produce the same token
count as an SFT arm reading oracle traces. Reporting those numbers is the
honest alternative to pretending they are equal.

### Matching is a property of your config, not an assertion

The harness records what happened; it does not refuse to run an unmatched
comparison. `controlled` is built from the base recipe, so if an arm overrides
`train.cycles`, `train.rollouts_per_cycle` or
`train.gradient_accumulation_steps`, that arm's real budget differs from the
`controlled` block. The evidence that it did is always present:
`arms_differ_only_in` lists the overrides and every result row carries its own
`optimizer_steps`.

For `recipes/benchmark_calc.yaml` the budgets work out as follows, computed
from the merged configs rather than asserted:

| arm | mode | loss mode | divergence | top_k | selector | optimizer steps |
| --- | --- | --- | --- | --- | --- | --- |
| (shared cold start) | sft | - | - | - | - | 200 |
| `cold-start-only` | sft | bucketed_topk_tail | reverse_kl | 16 | all_model_tokens | 0 |
| `sft-continued` | sft | bucketed_topk_tail | reverse_kl | 16 | all_model_tokens | 60 |
| `offline-kd` | offline_kd | bucketed_topk_tail | reverse_kl | 16 | all_model_tokens | 60 |
| `opd-bucketed-k16` | opd | bucketed_topk_tail | reverse_kl | 16 | all_model_tokens | 60 |
| `opd-exact` | opd | exact_full_vocab | reverse_kl | 1 | all_model_tokens | 60 |
| `opd-bucketed-forward-kl` | opd | bucketed_topk_tail | forward_kl | 16 | all_model_tokens | 60 |
| `opd-tool-and-final` | opd | bucketed_topk_tail | reverse_kl | 16 | tool_and_final | 60 |

`top_k: 1` on the exact arm is not a truncation. `RunConfig` normalizes
`loss.top_k` to 1 whenever `loss.mode` is `exact_full_vocab`, because `top_k`
has no meaning there; the manifest then records the student vocabulary size as
the effective `top_k`.

The step counts follow from `rollouts_per_cycle: 8` and
`gradient_accumulation_steps: 8` in the base recipe, which gives
`ceil(8 / 8) = 1` optimizer step per cycle. Verify any benchmark's budget
before trusting it:

```bash
python - <<'PY'
from miniverl.config.models import RunConfig
from miniverl.evaluation.benchmark import _load_base, deep_merge
from miniverl.evaluation.schema import BenchmarkConfig

spec = BenchmarkConfig.from_yaml("recipes/benchmark_calc.yaml")
base = _load_base(spec)
for arm in spec.arms:
    cfg = RunConfig.model_validate(deep_merge(base, arm.overrides))
    accum = cfg.train.gradient_accumulation_steps
    per_cycle = max(1, (cfg.train.rollouts_per_cycle + accum - 1) // accum)
    total = per_cycle * (cfg.train.cycles + cfg.train.sft_warmup_cycles)
    print(f"{arm.name:26s} {cfg.run.mode.value:11s} steps={total}")
PY
```

## The shared cold-start checkpoint

`cold_start_cycles` in the benchmark YAML runs one SFT cold start per seed,
before any arm. `_cold_start` deep-merges these settings over the base recipe:

```yaml
run:   {mode: sft, seed: <seed>, name: "<benchmark>-coldstart"}
train: {cycles: <cold_start_cycles>, sft_warmup_cycles: 0,
        eval_every_cycles: 0, save_every_cycles: 0}
cache: {reuse_across_policy_versions: false, strict_policy_version: true}
report: {enabled: false}
eval:  {enabled: false}
```

The resulting `checkpoints/final` directory is loaded into every arm at that
seed. The cold start does not evaluate itself — `eval.enabled: false` above —
because on a real model each evaluation pass costs minutes of generation. The
shared starting score is instead the `cold-start-only` arm, which is evaluated
on the benchmark's own `eval_split` like every other arm.

The load is deliberately weights-only:

```python
load_checkpoint(
    checkpoint,
    backend=trainer.student,
    optimizer=trainer.optimizer,
    device=trainer.student.device,
    include_optimizer=False,
    include_rng=False,
)
```

`include_optimizer=False` matters. If the cold start's Adam moments were
restored, every arm would inherit momentum pointing in the direction the SFT
cold start was already moving, which advantages whichever arm most resembles
that cold start. `include_rng=False` keeps each arm on the RNG stream implied
by its own seed rather than replaying the cold start's stream. Each load emits
a `benchmark_cold_start_loaded` event into the arm's `events.jsonl` with the
note `weights only; optimizer state and RNG intentionally not restored`.

Set `cold_start_cycles: 0` to skip the cold start entirely; every arm then
starts from freshly initialized weights and `shared_initial_checkpoint` is
recorded as `false`.

## Running a benchmark

```bash
# CPU, toy models
miniverl benchmark recipes/benchmark_calc.yaml --output runs/benchmarks

# One 16 GB GPU, the pinned Qwen3 pair
miniverl benchmark benchmarks/configs/gpu_calc_hard.yaml --output runs/benchmarks
```

`benchmarks/configs/gpu_calc_hard.yaml` is the GPU counterpart: it uses
`recipes/qwen_consumer_gpu_calc.yaml` as its base, 12 cold-start cycles, one
seed, the `test` split, `difficulty: hard`, and four arms
(`cold-start-only`, `sft-continued`, `opd-bucketed-k64` and
`opd-privileged-context`, the last differing only in
`models.teacher.mode`). One seed means no significance claim; see
[Single-seed significance](#single-seed-significance).

Options, all from `src/miniverl/cli.py`:

- `--output PATH` - output directory; defaults to `output_dir` in the
  benchmark YAML (`runs/benchmarks` in the shipped example)
- `--notes TEXT` - free text stored in the result and rendered into the
  Markdown
- `--json` - print the whole result as JSON instead of a table

`miniverl benchmark` requires the training extra
(`pip install "miniverl[train]"`); the CLI raises a `MissingDependencyError`
with the exact install command otherwise.

The example benchmark runs 7 arms at 2 seeds plus 2 cold starts, all on toy CPU
models. Its wall clock on any particular machine is not measured here.

### Writing your own benchmark config

```yaml
schema_version: 1
name: my-comparison           # 1-80 characters, used for the output filenames
description: >-
  What question this comparison answers.
base: toy_cpu.yaml            # path relative to this file, or an inline mapping
cold_start_cycles: 200        # 0 disables the shared cold start
eval_split: test              # train | eval | test
seeds: [1234, 20260727]       # at least one; more seeds means a variance range
output_dir: runs/benchmarks

arms:
  - name: baseline            # arm names must be unique
    description: ...
    overrides: {}             # deep-merged into the base recipe
  - name: variant
    overrides:
      loss: {divergence: forward_kl}
```

`overrides` is deep-merged by `deep_merge`: nested mappings are merged
recursively, everything else is replaced. Put **only** the keys the arm is
supposed to differ in there. The harness force-overrides `run.seed`,
`run.name` and `report.enabled` on every arm after your overrides are applied.

Validate the merged configs before spending GPU time; each arm is a normal
`RunConfig` and is rejected at parse time if the combination is contradictory
(for example `run.mode: opd` with `cache.reuse_across_policy_versions: true`).

## Reading the output

`run_benchmark` writes two files into the output directory:

- `<name>.json` - the full `BenchmarkResult`
- `<name>.md` - `render_benchmark_markdown` output

It also leaves one complete run directory per arm per seed, named
`<name>-<arm>-s<seed>`, plus `<name>-coldstart-s<seed>` for each cold start.
Those directories hold the usual artifacts: `config.original.yaml`,
`config.resolved.yaml`, `manifest.json`, `environment.json`, `metrics.jsonl`,
`events.jsonl`, `trajectories.jsonl`, `eval_trajectories.jsonl`,
`teacher-cache/`, `checkpoints/`, `eval.json`.

### The JSON

Top level (`BenchmarkResult`, `extra="forbid"`):

| key | contents |
| --- | --- |
| `schema_version` | `1`; a different value is rejected on load |
| `miniverl_version` | the version that produced the file |
| `name`, `description`, `notes` | from the benchmark config and `--notes` |
| `created_at` | UTC ISO-8601, second resolution |
| `git_commit` | resolved by reading `.git` directly; `null` outside a checkout |
| `hardware` | `gpu`, `os`, `cpu_count` |
| `software` | `python` version and the tracked package versions |
| `controlled` | the held-constant block described above |
| `arms` | a list of `ArmResult`, one per arm per seed |
| `seeds` | the seed list |

To read it without writing code:

```bash
python -c "import json;d=json.load(open('runs/benchmarks/calc-matched-budget.json'));\
print(json.dumps(d['controlled'], indent=2))"
```

### The Markdown

`<name>.md` contains, in order: the header with miniVERL version, git commit,
timestamp and hardware line; a per-arm results table; an aggregate table with
mean, min and max success per arm; a fenced JSON block with the entire
`controlled` dictionary; and the notes if any.

Two things in the rendering are load-bearing. When `len(result.seeds) == 1`
the header line gains `**single seed -- no significance claimed**`, and the
same warning is printed by the CLI. The paragraph after the controls block
states which quantities were measured rather than matched, so the table cannot
be quoted without its caveat.

## Submitting a result

`miniverl benchmark` produces a multi-arm comparison. For a single run on your
own hardware, use the export command instead:

```bash
miniverl train recipes/qwen_consumer_gpu_calc.yaml
miniverl export-benchmark runs/<run-id> --notes "RTX 4080, driver 596.49"
```

`export_run` in `src/miniverl/evaluation/export.py` reads the run directory,
validates it, and writes `benchmark-submission.json` (or `--out PATH`)
containing a single-arm `BenchmarkResult`. It refuses to export a run with no
evaluation results and tells you to run `miniverl eval --run <run-dir>` first
or to set `eval.enabled: true`.

Sanitization is the point of that command. `sanitize_hardware` keeps only GPU
availability, name, VRAM, capability, driver version and device count, OS name
and release, machine architecture and CPU count. `run_dir` is reduced to the
directory *name*, so no absolute path from your machine survives. Nothing else
from the environment record is copied. Read the file before you publish it
anyway.

The JSON Schema for the result format is generated from the same Pydantic model
that writes it, so the two cannot drift:

```bash
miniverl schema --out benchmarks/schema/benchmark-result.schema.json
```

The repository ships `benchmarks/README.md`, `benchmarks/configs/` (with
`cpu_toy_calc.yaml` and `gpu_calc_hard.yaml`) and
`benchmarks/schema/benchmark-result.schema.json`, which is byte-identical to
the output of `miniverl schema`. `benchmarks/results/` exists but is empty: no
hardware result has been submitted yet, including from the development machine.
`.gitignore` carries an explicit `!benchmarks/results/*.json` line, commented
`Committed benchmark results are small, schema-validated JSON and are wanted.`

A submission should state, at minimum: the recipe used, the exact GPU and
driver, the miniVERL version and git commit (all already in the file), and
whether the arms were matched. If you changed the recipe, include the diff.

## How to lie with these numbers, and how we avoid it

Every trap below is one that this harness makes possible. The mitigation is
named in each case, and none of them is automatic.

### Unmatched steps

**The lie.** Give the arm you like more optimizer steps, or a larger effective
batch, and report only final success rates.

**Why it is possible here.** `controlled` is derived from the base recipe.
Nothing stops an arm from overriding `train.cycles`,
`train.rollouts_per_cycle` or `train.gradient_accumulation_steps`.

**The mitigation.** Every result row carries its own `optimizer_steps`, the
Markdown table has a `steps` column, and `arms_differ_only_in` records the full
override mapping. Before quoting a table, check that the `steps` column is
constant across the arms being compared, or that the difference is the point.
The `cold-start-only` arm in the shipped benchmark has 0 steps on purpose: it
is the common starting point, not a competitor.

### Cherry-picked seeds

**The lie.** Run five seeds, report the best.

**Why it is possible here.** The harness runs the seeds listed in `seeds:`.
Nothing prevents you from editing that list after seeing results.

**The mitigation.** `seeds` is recorded at the top level of the result and
inside `controlled`. `aggregate()` reports `success_rate_min` and
`success_rate_max` next to the mean, and the Markdown renders all three, so a
single lucky seed inside a multi-seed arm is visible as a wide min-max range.
The number of seeds per arm is printed as its own column. If you rerun with a
different seed list, publish both files.

### Teacher-FLOPs claims

**The lie.** "Selecting 30% of positions cuts teacher compute by 70%."

**Why it is false.** The teacher runs a full forward pass over the whole
sequence regardless of how many positions are selected; selection only affects
the LM-head projection, the cache payload and the number of student loss
positions.

**The mitigation.** The metric is named `teacher_queried_position_ratio` and
never "teacher compute saved", and the docstring of
`src/miniverl/selection/selectors.py` states the accounting explicitly. If you
want a teacher-compute claim, measure wall clock and peak memory, both of which
the harness already reports per arm.

### Single-seed significance

**The lie.** One seed per arm, a 4-point gap, and the word "improves".

**The mitigation.** `aggregate()` sets `single_seed: true`, the Markdown header
appends `**single seed -- no significance claimed**`, and the CLI prints:

```
single seed - no statistical significance is claimed. Add more entries to
`seeds:` for a variance estimate.
```

miniVERL computes no confidence intervals and no hypothesis tests; do not add
them in prose.

### Attributing a cold start's gains to the method

This one is not hypothetical. On the one measured 16 GB run of
`recipes/qwen_consumer_gpu_calc.yaml` (run id `rtx4080-calc-opd`), held-out
greedy evaluation on 12 calculator tasks went from 0.0 percent to 100.0 percent
across 16 optimizer steps in 481.1 s. The 8-cycle SFT cold start did most of
that work: the first OPD rollout batch already scored 83.3 percent. That run
demonstrates that the pipeline works end to end on a consumer GPU. It is not
evidence that OPD beats SFT, and the medium-difficulty calculator task
saturates.

**The mitigation.** The `cold-start-only` arm exists so the shared starting
point appears as a row in the table, evaluated on the same split as every other
arm, rather than as an unstated prior. Quote the delta over that row, not the
delta over zero.

### Comparing across machines or commits

**The mitigation.** `hardware`, `software`, `miniverl_version` and
`git_commit` are recorded in every result file. Two results with different
values in any of those fields are not a matched comparison, whatever the
`controlled` block says.

### Reporting a benchmark you did not run

`measurement_status` on every arm is `measured`, and
`manifest["measurement_status"]` in each run directory records
`simulated_results: "none"`. There is no simulation path in this codebase; any
number presented as a miniVERL measurement should be traceable to a run
directory.

## See also

- `docs/reproducibility.md` - seeding, determinism, and reproducing a published
  run
- `docs/troubleshooting.md` - what to do when a benchmark arm fails
- `docs/memory.md` - memory strategies and the peak-VRAM numbers
