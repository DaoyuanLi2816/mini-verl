# Benchmarking

miniVERL ships a controlled benchmark harness with an explicit comparison
axis. It runs several training configurations ("arms"), rejects undeclared
config differences before model allocation, and writes one JSON file and one
Markdown file describing the results, complete resolved controls and measured
quantities that were not matched.

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

The published v0.1/v0.2 result files predate destructive trainer teardown.
Their per-arm `peak_allocated_bytes` measurements came from live tensors during
that arm, but historical `peak_reserved_bytes` may include CUDA caching
allocator state retained from an earlier arm in the same process. The result
JSON is preserved rather than silently rewriting measured history. New harness
runs close each trainer inside a function-level context, drop model/optimizer
references, collect garbage, empty the CUDA cache, and only then construct the
next arm. Use new lifecycle-isolated runs for cross-arm reserved-memory
comparisons.

### Per-arm result fields

`ArmResult` in `src/miniverl/evaluation/schema.py` is the full per-arm record:

| field | source |
| --- | --- |
| `name`, `description` | the arm definition in the benchmark YAML |
| `mode` | `run.mode` after merging the arm overrides |
| `seed` | the seed this repetition used |
| `run_id`, `run_dir` | the run directory this arm produced |
| `objective`, `opd_freshness`, `loss_mode`, `divergence`, `selector`, `top_k` | the mode-aware run manifest; SFT records `sft_cross_entropy` and null divergence/top-k |
| `resolved_config_digest`, `structured_diff` | compatibility fields for the fully defaulted pre-allocation arm config and its complete declared diff, including harness bookkeeping |
| `declared_config_digest`, `scientific_config_diff` | digest of the pre-allocation arm config and only the differences declared as experimental treatments |
| `runtime_resolved_config_digest`, `runtime_resolution_diff` | digest after `auto` decisions are frozen and changes such as `models.device: auto -> cuda` or `memory.strategy: auto -> resident`; these are execution provenance, not treatments |
| `harness_config_diff` | run name/seed/id and report toggles introduced by the benchmark harness; never interpreted as scientific differences |
| student/teacher model IDs and revisions, tokenizer fingerprint, context mode | the actual run manifest |
| `teacher_adapter` | validated adapter identity, hashes and policy evaluation, or null |
| `top_k` | `manifest["objective"]["top_k"]`: the student vocabulary size in `exact_full_vocab` mode, otherwise `min(loss.top_k, vocab_size)` |
| `optimizer_steps` | `TrainResult.global_step` |
| `policy_version` | policy version reached |
| `tasks` ... `tokens_per_solved_task` | the held-out evaluation payload above; `tokens_per_solved_task` is stored as `null` when it was NaN |
| `selected_training_tokens_total`, `model_generated_training_tokens_total`, `teacher_queried_positions_total` | numerators summed over every cycle; SFT teacher-query fields are null |
| `selected_position_ratio`, `teacher_queried_position_ratio` | ratios of summed numerators and denominators, never an average of cycle ratios |
| `cache_current_bytes`, `cache_bytes_written_total`, `cache_compression_ratio` | current cache footprint, cumulative bytes written despite pruning, and compression |
| `peak_allocated_bytes`, `peak_reserved_bytes` | maxima described above, `null` on CPU |
| `train_seconds`, `evaluation_seconds`, `wall_seconds` | separately measured phase and enclosing wall times |
| `baseline_success_rate` | reserved for a shared starting score. The harness currently always writes `null`: `_cold_start` runs with `eval.enabled: false` and returns no score, because the cold start is measured by the `cold-start-only` arm on the benchmark's own split instead |
| `measurement_status` | explicit status for time, VRAM, cache and policy-competence fields |

`BenchmarkResult.aggregate()` groups arms by name and reports
`success_rate_mean`, `success_rate_min`, `success_rate_max`, the seed count,
and a `single_seed` boolean. It computes no confidence intervals and no
p-values.

## Explicit design and budget axis

Schema-v2 benchmark configs separate four layers:

```yaml
base: ...
common_overrides: ...
cold_start_overrides: ...
allowed_differences: [...]
budget_axis: optimizer_steps
arms:
  - name: ...
    overrides: ...
```

Before a run directory is created or a model is loaded, `run_benchmark`
resolves the common config, the cold-start config and every arm for every seed.
It computes deterministic leaf-level structured diffs from the common config
and rejects any path not declared in `allowed_differences`. Harness-only paths
(`run.name`, `run.seed`, `run.run_id`, `report.enabled`) are allowlisted
internally.

New results record three disjoint views: `scientific_config_diff` for declared
treatments, `runtime_resolution_diff` for load-time decisions, and
`harness_config_diff` for bookkeeping. `structured_diff` remains as a
compatibility view for existing schema-v2 readers and is not labeled a
scientific diff.

The top-level `controlled` block points to the complete
`common_declared_config` and its digest; the legacy-named
`common_resolved_config` remains an identical compatibility field. Neither is
reconstructed from a hand-picked subset. The cold-start record carries its own
config, digest, environment/difficulty, checkpoint digest and time.

`budget_axis: optimizer_steps` means **equal optimizer updates**, not equal
compute. Quantities that cannot be matched by construction are measured:

- student generated tokens (`generated_tokens_per_task`,
  `tokens_per_solved_task`)
- selected training positions (`selected_training_tokens_total`)
- teacher query ratio (`teacher_queried_position_ratio`)
- teacher-cache current/cumulative bytes and compression ratio
- peak CUDA memory
- train, evaluation and wall time

An OPD arm samples its own trajectories, so it cannot produce the same token
count as an SFT arm reading oracle traces. Reporting those numbers is the
honest alternative to pretending they are equal.

### Controls are enforced before allocation

An undeclared change in difficulty, schedule, effective batch size or any other
leaf is a `ConfigError` that explicitly says validation happened before model
loading. Intentional differences must be added to `allowed_differences`.
Different step counts can be declared—for example the zero-step
`cold-start-only` reference—but the result still records each actual count.

For `recipes/benchmark_calc.yaml` the budgets work out as follows, computed
from the merged configs rather than asserted:

| arm | mode | loss mode | divergence | configured top_k | selector | optimizer steps |
| --- | --- | --- | --- | --- | --- | --- |
| (shared cold start) | sft | - | - | - | all_model_tokens | 450 |
| `cold-start-only` | sft | - | - | - | all_model_tokens | 0 |
| `sft-continued` | sft | - | - | - | all_model_tokens | 60 |
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
from miniverl.evaluation.benchmark import resolve_benchmark_configs
from miniverl.evaluation.schema import BenchmarkConfig

spec = BenchmarkConfig.from_yaml("recipes/benchmark_calc.yaml")
_, cold, arms = resolve_benchmark_configs(spec)
print("cold", cold.environment.difficulty, cold.train.cycles)
for arm, cfg, diff in arms:
    accum = cfg.train.gradient_accumulation_steps
    per_cycle = max(1, (cfg.train.rollouts_per_cycle + accum - 1) // accum)
    total = per_cycle * (cfg.train.cycles + cfg.train.sft_warmup_cycles)
    print(f"{arm.name:26s} {cfg.run.mode.value:11s} steps={total} diff={len(diff)}")
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

# Strictly offline after preloading every pinned model, tokenizer and adapter
miniverl benchmark benchmarks/configs/gpu_calc_hard.yaml \
  --output runs/benchmarks --offline
```

`benchmarks/configs/gpu_calc_hard.yaml` is the GPU counterpart: it uses
`recipes/qwen_consumer_gpu_calc.yaml` as its base, 12 cold-start cycles, the
`test` split, `difficulty: hard`, two prespecified seeds, and five arms:
`cold-start-only`, `sft-continued`, `opd-raw-teacher`,
`opd-privileged-context` and `opd-protocol-sft-teacher`. The final arm is gated
on a recorded teacher tool-policy evaluation; see
[`teacher-adapters.md`](teacher-adapters.md). The primary config pins the public
Hub adapter to immutable revision
`23323751318135484c06c043b1f9b9e7016dd89f`; the `_local_adapter` config changes
only its source/path for offline use.

The completed two-seed RTX 4080 result is published as
[`gpu-calc-hard-equal-update-v2.json`](../benchmarks/results/gpu-calc-hard-equal-update-v2.json),
with a [human-readable table](../benchmarks/results/gpu-calc-hard-equal-update-v2.md)
and [SVG comparison](gpu-calc-hard-equal-update-v2.svg). To rebuild the portable
artifacts from the preserved run directory:

```bash
python scripts/publish_benchmark_artifacts.py \
  runs/benchmarks/gpu-calc-hard-equal-update-v2.json \
  --results-dir benchmarks/results \
  --svg docs/gpu-calc-hard-equal-update-v2.svg \
  --driver-version 596.49
```

The publisher removes machine-local absolute paths before recomputing config
digests, validates the schema-v2 model, and makes the chart carry a prefix of
the exact source JSON SHA-256.

Options, all from `src/miniverl/cli.py`:

- `--output PATH` - output directory; defaults to `output_dir` in the
  benchmark YAML (`runs/benchmarks` in the shipped example)
- `--notes TEXT` - free text stored in the result and rendered into the
  Markdown
- `--offline` - prohibit all network access and require every model, tokenizer
  and pinned adapter file to exist locally or in the Hugging Face cache
- `--json` - print the whole result as JSON instead of a table

`miniverl benchmark` requires the training extra
(`python -m pip install ".[train]"` from a source checkout); the CLI raises a
`MissingDependencyError` with the exact install command otherwise.

The example benchmark runs 7 arms at 2 seeds plus 2 cold starts, all on toy CPU
models. Its wall clock on any particular machine is not measured here.

### Writing your own benchmark config

```yaml
schema_version: 2
name: my-comparison           # 1-80 characters, used for the output filenames
description: >-
  What question this comparison answers.
base: toy_cpu.yaml            # path relative to this file, or an inline mapping
common_overrides:
  environment: {difficulty: hard}
cold_start_overrides:
  environment: {difficulty: medium}  # explicit transfer design, if intended
cold_start_cycles: 200        # 0 disables the shared cold start
allowed_differences: [run.mode, train.cycles, loss.divergence]
budget_axis: optimizer_steps
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

Each layer is deep-merged: nested mappings are merged recursively and everything
else is replaced. Every arm difference must be declared. The harness
force-overrides `run.seed`, `run.name` and `report.enabled` after arm overrides.

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
| `schema_version` | `2` for new harness output; preserved `1` artifacts remain readable |
| `miniverl_version` | the version that produced the file |
| `name`, `description`, `notes` | from the benchmark config and `--notes` |
| `created_at` | UTC ISO-8601, second resolution |
| `git_commit` | resolved by reading `.git` directly; `null` outside a checkout |
| `invocation`, `budget_axis` | exact command arguments and declared comparison axis |
| `hardware` | `gpu`, `os`, `cpu_count` |
| `software` | `python` version and the tracked package versions |
| `cold_start` | resolved config/digest, environment/difficulty, and per-seed checkpoint digest/time |
| `common_declared_config`, `common_declared_config_digest` | complete fully defaulted, pre-allocation shared config and SHA-256 |
| `common_resolved_config`, `common_resolved_config_digest` | compatibility aliases retained for existing schema-v2 readers |
| `controlled` | pointer/digest plus declared difference paths |
| `arms` | a list of `ArmResult`, one per arm per seed |
| `seeds` | the seed list |

To read it without writing code:

```bash
python -c "import json;d=json.load(open('runs/benchmarks/cpu-toy-calc-equal-update-v2.json'));\
print(json.dumps(d['controlled'], indent=2))"
```

### The Markdown

`<name>.md` contains the version, git commit, budget axis, hardware, seeds, a
per-arm table and the resolved-control caveat. Complete resolved configs,
digests and structured diffs stay in JSON rather than being truncated into a
human table.

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

The repository ships configs, a byte-reproducible generated schema and
preserved CPU/RTX 4080 result artifacts. Historical v1 and the published v2
result stay byte-for-byte unchanged; the v1 migration/erratum and the v2
memory-provenance caveat are documented in `benchmarks/README.md`.

A submission should state, at minimum: the recipe used, the exact GPU and
driver, the miniVERL version and git commit (all already in the file), and
whether the arms were matched. If you changed the recipe, include the diff.

## How to lie with these numbers, and how we avoid it

Every trap below is one that this harness makes possible. The mitigation is
named in each case, and none of them is automatic.

### Undeclared schedule differences

**The lie.** Give the arm you like more optimizer steps, or a larger effective
batch, and report only final success rates.

**How schema v2 prevents it.** The complete resolved config for each arm is
diffed against the complete resolved common config before a model is loaded.
An arm cannot change `train.cycles`, `train.rollouts_per_cycle`,
`train.gradient_accumulation_steps` or any other leaf unless that path is
listed in `allowed_differences`.

**What still needs review.** A declared difference can still make a comparison
unfair. Every result row carries its own `optimizer_steps`, and the JSON stores
the exact `structured_diff`. Before quoting a table, check that the declared
`budget_axis` is the question you intended to ask and that the measured counts
and times support the comparison. The `cold-start-only` arm in the shipped
benchmark has 0 steps on purpose: it is the common starting point, not a
competitor.

### Cherry-picked seeds

**The lie.** Run five seeds, report the best.

**Why it remains possible.** The harness faithfully runs the seeds listed in
`seeds:`; it cannot know whether that list was chosen after seeing results.

**The mitigation.** `seeds` is recorded at the top level of the result.
`aggregate()` reports `success_rate_min` and
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
