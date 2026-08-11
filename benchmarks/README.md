# Benchmarks

Everything in this directory is measured. Nothing is estimated, extrapolated or
projected. If a configuration was not run, it says "not run" rather than being
quietly omitted.

```text
benchmarks/
├── configs/     benchmark specifications (BenchmarkConfig YAML)
├── results/     measured results (BenchmarkResult JSON + Markdown)
├── schema/      the JSON Schema that every result file must satisfy
└── README.md    this file
```

## Running one

```bash
# CPU, toy models, minutes
miniverl benchmark recipes/benchmark_calc.yaml --output runs/benchmarks

# One 16 GB GPU, the real Qwen3 pair, equal optimizer updates
miniverl benchmark benchmarks/configs/gpu_calc_hard.yaml --output runs/benchmarks

# Offline equivalent after downloading the pinned protocol adapter
miniverl benchmark benchmarks/configs/gpu_calc_hard_local_adapter.yaml \
  --output runs/benchmarks
```

Each run writes `<name>.json` and `<name>.md` into the output directory. The
Markdown is for humans; the JSON is the artifact to submit.

The primary GPU config uses the public protocol-teacher adapter at immutable
Hub revision `23323751318135484c06c043b1f9b9e7016dd89f`. The local-adapter
config is the explicit offline equivalent.

## What the budget axis means

The v2 configuration declares `budget_axis: optimizer_steps`. Every arm
receives, by construction:

* the same environment, difficulty and `split_seed`, so the same task instances;
* the same **initial weights** — one shared supervised cold start, loaded
  **weights only** so no Adam momentum leaks from the cold start into whichever
  arm happens to resemble it;
* the same number of optimizer steps and the same effective batch size
  (`gradient_accumulation_steps` trajectories per step);
* the same maximum trajectory length, turn limit and rollout bounds;
* the same held-out split, evaluation seed and greedy temperature.

This is an **equal-optimizer-update** comparison, not a generic matched-compute
claim. Arms differ only in paths declared under `allowed_differences`; resolved
leaf diffs are checked before any model is loaded. Four quantities cannot be
matched by construction, so they are measured and reported per arm:

| Quantity | Why it cannot be matched |
| --- | --- |
| student generated tokens | the policy decides how much it writes |
| selected training tokens | depends on the selector and on what the policy emitted |
| teacher queried-position ratio | same |
| wall clock | different arms do genuinely different work |

## How to lie with these numbers

Listed so you can check that this repository does not.

* **Unmatched steps.** Give the favoured arm more optimizer steps. Countered by
  the shared budget above; `optimizer_steps` is in every row, so check it.
* **A different starting point.** Let one arm keep the cold start's optimizer
  state. Countered by loading weights only.
* **Cherry-picking a seed.** Countered by reporting every seed's row *and* the
  aggregate, and by flagging `single_seed` when there is only one.
* **Claiming saved teacher FLOPs.** Reducing selected positions reduces the LM
  head, the cache and the loss positions — not the teacher's forward pass. The
  field is named `teacher_queried_position_ratio` for that reason.
* **Reporting a saturated task.** If every arm scores near the ceiling, the
  benchmark measures nothing. Both saturated results in this repository say so in
  their notes.

## Results in this repository

| File | Hardware | What it shows |
| --- | --- | --- |
| `results/alignment-external-v1.json` | RTX 4080 16 GB | **Preregistered early stop, not a method comparison.** Two declared lineages and eight candidates all measured 0/64 retained JSONNav utility against an unchanged 20% floor. No checkpoint, teacher, continuation arm or final-test result exists. [Study, correction provenance and evaluator boundary.](../docs/alignment-external/alignment-external-v1.md) |
| `results/alignment-lab-v1.json` | RTX 4080 16 GB | Alignment Lab v1: six methods from one Qwen3-0.6B SFT checkpoint, three seeds and 48 paired policy tasks per arm. The SFT start saturated alignment and utility; no continuation method improved it, so the pilot recommends no online teacher querying. [Report, figures and caveats.](../docs/alignment-lab/alignment-lab-v1.md) |
| `results/consumer-runtime-v1.json` | RTX 4080 16 GB | Preregistered eight-cell systems matrix for sequential/batch-2/batch-4/auto updates under dual-model and shared-backbone ownership. Batch-4 reached 3.866 and 3.475 trajectories/s at 3.035 and 2.227 GiB reserved; all equivalence gates passed. [Figure, method and caveats.](../docs/consumer-runtime-v1.md) |
| `results/recoverybench-v1-equal-updates.json` | RTX 4080 16 GB | RecoveryBench v1 primary schema-v3 result: six arms, three seeds and eight equal continuation updates. Frozen-student KD reached 23.2% strict success versus 10.9% for strict fresh OPD. [Analysis, figures and caveats.](../docs/recoverybench/recoverybench-v1.md) |
| `results/recoverybench-v1-equal-selected-tokens.json` | RTX 4080 16 GB | Three-method secondary view at the first optimizer boundary at or beyond 6,224 selected positions. All methods stopped after eight updates; overshoot is retained. |
| `results/recoverybench-v1-equal-wall-time.json` | RTX 4080 16 GB | Preserved cycle-capped wall diagnostic. SFT and frozen KD reached the eight-cycle ceiling before the internal 50-second timer, while fresh OPD crossed it in one indivisible step; this is not exact equal-time evidence. |
| `results/gpu-calc-hard-equal-update-v2.json` | RTX 4080 16 GB | Primary schema-v2, five-arm, two-seed equal-update comparison. The protocol-trained teacher prevented the raw/privileged-teacher collapse and reached 100% on both seeds, tying SFT rather than beating it. [Chart and interpretation.](../docs/rtx4080-baselines.md#protocol-teacher-equal-update-comparison-schema-v2) |
| `results/cpu-toy-calc-matched.json` | CPU only | **Parity, not ranking.** All seven arms run to completion under identical budgets on the toy backend. The accuracy differences are within noise and must not be read as a ranking; see the note in the file. |
| `results/rtx4080-calc-hard-matched.json` | RTX 4080 16 GB | Legacy single-seed equal-update comparison on the chained calculator split with the real Qwen3 pair. **On-policy distillation lost**, with both a standard and a privileged-context teacher; the transcript-level diagnosis and v1 erratum are in [`docs/rtx4080-baselines.md`](../docs/rtx4080-baselines.md#legacy-equal-update-comparison-schema-v1). |

### Alignment Lab v1

All 18 completed arms start from one checksummed SFT checkpoint and retain the
same 48 ordered final-test tasks within each of three preregistered seeds. The
task-level JSONL contains 864 rows. DPO provenance includes the external pinned
TRL 1.8.0 training cost; teacher-query ratio is explicitly a selected-position
ratio, not teacher-backbone FLOPs.

| Artifact | SHA-256 |
| --- | --- |
| `alignment-lab-v1.json` | `584752dccb91654109c357b8ebb12681a12a9c1476a9ba539dd35e4d860a22ef` |
| `alignment-lab-v1-task-results.jsonl` | `8d7fc723436d7377d196fc44046d960e3cb7f0aa81e03d49ef05b627eb84630f` |
| `alignment-lab-v1-state-supervision.json` | `9e08129ba4cd9e460c189b94b4e421d881ba69e3938f02eac95d251f50c88788` |
| `../paper/alignment-lab-v1/alignment-lab-v1.pdf` | `adbffa967f6b9a25d2cdb0cc4464a93c13db4615a1e91499585fb199285d980b` |

The State × Supervision file is a measured signal diagnostic, not a separately
trained hard-target result. The fresh soft target retains 0.0251% mean
probability mass beyond argmax under matched states, teacher, budget, checkpoint
and seeds; no soft-target quality advantage is claimed. All public Alignment
Cards are under `alignment-cards/alignment-lab-v1/`.

### Consumer Runtime v1

This is a performance and numerical-equivalence experiment, not a task-quality
comparison. The frozen JSON and profiler summary are generated by
`scripts/benchmark_consumer_runtime.py`; the SVG is regenerated and byte-checked
by `scripts/publish_consumer_runtime_artifacts.py`. Revision 1.2 of the public
preregistration locks eager attention and NF4 weights with FP32 compute after
two non-headline diagnostics. The final result records identical trajectory and
teacher-target digests plus 12 passed loss/gradient/update equivalence checks.

| Artifact | SHA-256 |
| --- | --- |
| `consumer-runtime-v1.json` | `a302da31af99f1d29f1efd4e6b3dbeb6ea4ac956bba102ca8a1bee8dff0319eb` |
| `consumer-runtime-v1-profiler.json` | `66111cd7fc876cf1befea3297a1a51bcd99252c0bf8989c029381e1dc155a98b` |
| `../docs/consumer-runtime-v1-pareto.svg` | `98645a668a7832423d28b621262292619615917f037adf7219ff1bf071fb2fea` |

### RecoveryBench v1

RecoveryBench asks whether scoring fresh current-student states improves
SQLite tool-error recovery over distillation on a fixed state set collected
from the cold-start student. It is a mechanism study, not an alignment
benchmark. The primary hypothesis was not supported, and fresh supervision was
substantially more expensive in this implementation.

The three result JSON files are immutable source artifacts. The compact
task-result JSONL, paired analysis and three SVGs are generated from them by
`scripts/publish_recoverybench_artifacts.py`. The final run follows the public
[revision-1.3 preregistration](../docs/recoverybench/preregistration.md); one
earlier partial run with an oracle-schedule defect and one post-run aborted
wall-budget replacement are preserved outside the publication set and are not
used in any headline result.

| Artifact | SHA-256 |
| --- | --- |
| `recoverybench-v1-equal-updates.json` | `6ce2e6837e12b99ebc4fad6d27ce3e69c92e295ff3b9b60e0f68c2d308022384` |
| `recoverybench-v1-equal-selected-tokens.json` | `fe4c9afc799724dfe7a32e631676a1e5177c44559a7374d2ea31da135354f137` |
| `recoverybench-v1-equal-wall-time.json` | `425b0fa568f37b09e61af731d3da5009bd3833bddde6efaf2c66e9dba8355cbe` |
| `recoverybench-v1-task-results.jsonl` | `aff96bffc6da27240a852410ac041bd4d95badf34cad030e6f437be1491a55ad` |
| `recoverybench-v1-analysis.json` | `8a6891f74aed80f07ec00d5ea1909895c579346e1abbb1d5d95a354bb46c6b81` |

### Erratum for the legacy RTX 4080 result

`results/rtx4080-calc-hard-matched.json` is preserved byte-for-byte as a schema
v1 measurement and is now superseded by the v2 experiment specification. Its
continuation comparison remains valid in one important respect: every arm
started from the same checkpoint. Its provenance/accounting fields do not mean
what their names implied:

- the shared cold start was trained on calculator `medium`;
- continuation and held-out evaluation were on `hard`;
- the `controlled` block was copied from the base recipe rather than the
  resolved arm configs, so it incorrectly reports base values such as
  `difficulty: medium`, learning rate `1e-4`, cosine scheduling and 48 test
  tasks instead of the continuation settings;
- `selected_training_tokens` is only the final cycle's count, not the run total;
- SFT's teacher-query ratio is not semantically meaningful because no teacher
  was queried.

Schema v2 makes the transfer design explicit, hashes the resolved cold/arm
configs, sums accounting over the full run and records SFT teacher-query fields
as null. Do not use the legacy `controlled` or token-total fields for a new
comparison.

The historical `peak_allocated_bytes` values were measured during each arm.
Historical `peak_reserved_bytes` may include CUDA allocator state retained from
an earlier sequential arm. The result files are not rewritten for that memory
caveat; future comparisons use the lifecycle-isolated harness.

The original schema-v1 config is readable at
`configs/gpu_calc_hard_legacy_v1.yaml`. Current miniVERL deliberately does not
execute it. Reproduce it only from immutable commit
`3383f2b9a3c595e0fa143fecdc27522ab368b27f`, where its original filename was
`configs/gpu_calc_hard.yaml`.

The toy benchmark was run twice. The first attempt gave every arm a fresh cosine
schedule at the base learning rate after the cold start, which restarted the
learning rate and damaged the model; the second uses a constant, lower rate. Both
facts are recorded here rather than only the second run being shown. Only the
second is published, because the first measured a bug in the harness rather than
a property of any arm -- but it happened, so it is written down.

### Why the toy result is a parity check and nothing more

The published run covers two seeds, and the two-seed spread is what makes the
point:

| arm | seed 1234 | seed 20260727 | mean |
| --- | --- | --- | --- |
| cold-start-only | 66.7% | 8.3% | 37.5% |
| sft-continued | 83.3% | 8.3% | 45.8% |
| offline-kd | 62.5% | 4.2% | 33.3% |
| opd-bucketed-k16 | 79.2% | 8.3% | 43.8% |
| opd-exact | 79.2% | 4.2% | 41.7% |
| opd-bucketed-forward-kl | 79.2% | 8.3% | 43.8% |
| opd-tool-and-final | 79.2% | 4.2% | 41.7% |

Changing the seed moves a single arm by up to **58 percentage points**. Changing
the *objective* moves it by at most 4 points at a fixed seed, and the four OPD
arms are indistinguishable from each other at both seeds. The between-seed
variance is more than an order of magnitude larger than the between-arm
variance, so **no ranking can be read out of this table** -- which is precisely
what it is here to establish. What it does establish is that all seven arms,
including `exact_full_vocab`, run to completion under identical budgets and
produce well-formed trajectories.

One detail worth not mistaking for a defect: `opd-exact` reports `0.00 MiB` of
cache. Exact mode with a resident teacher rebuilds the full-vocabulary
distribution one chunk at a time and deliberately never persists a
`[positions, vocab]` tensor, so an empty cache is the designed behaviour.

## Submitting a result

```bash
miniverl train <recipe>
miniverl export-benchmark runs/<run-id> --out benchmarks/results/<gpu>-<recipe>.json
```

`export-benchmark` sanitizes the run: it keeps the GPU model, VRAM, capability,
driver, OS family and library versions, and drops absolute paths and anything
identifying. **Read the file before you post it.**

Validate it before opening the pull request:

```bash
python - <<'PY'
import json
from miniverl.evaluation.schema import BenchmarkResult
BenchmarkResult.model_validate(json.load(open("benchmarks/results/your-file.json")))
print("valid")
PY
```

Then open a pull request, or use the "Benchmark submission" issue template if you
would rather not. State every deviation from the shipped recipe; a result from a
modified recipe is welcome as long as it says so.

## The schema

RecoveryBench runs write schema v3; current calculator harness runs write v2.
The reader and generated JSON Schema continue to accept preserved v1 and v2
artifacts, and old measurements are never rewritten during migration.
`schema/benchmark-result.schema.json` is generated from the Pydantic model, so
the file and implementation cannot drift:

```bash
miniverl schema --out benchmarks/schema/benchmark-result.schema.json
```

CI regenerates it and fails if the committed copy differs.
