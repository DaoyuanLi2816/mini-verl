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
```

Each run writes `<name>.json` and `<name>.md` into the output directory. The
Markdown is for humans; the JSON is the artifact to submit.

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
| `results/gpu-calc-hard-equal-update-v2.json` | RTX 4080 16 GB | Primary schema-v2, five-arm, two-seed equal-update comparison. The protocol-trained teacher prevented the raw/privileged-teacher collapse and reached 100% on both seeds, tying SFT rather than beating it. [Chart and interpretation.](../docs/rtx4080-baselines.md#protocol-teacher-equal-update-comparison-schema-v2) |
| `results/cpu-toy-calc-matched.json` | CPU only | **Parity, not ranking.** All seven arms run to completion under identical budgets on the toy backend. The accuracy differences are within noise and must not be read as a ranking; see the note in the file. |
| `results/rtx4080-calc-hard-matched.json` | RTX 4080 16 GB | Legacy single-seed equal-update comparison on the chained calculator split with the real Qwen3 pair. **On-policy distillation lost**, with both a standard and a privileged-context teacher; the transcript-level diagnosis and v1 erratum are in [`docs/rtx4080-baselines.md`](../docs/rtx4080-baselines.md#legacy-equal-update-comparison-schema-v1). |

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

New harness runs write schema v2. The reader and generated JSON Schema continue
to accept preserved v1 artifacts; old measurements are never rewritten during
migration. `schema/benchmark-result.schema.json` is generated from the Pydantic
model, so the file and implementation cannot drift:

```bash
miniverl schema --out benchmarks/schema/benchmark-result.schema.json
```

CI regenerates it and fails if the committed copy differs.
