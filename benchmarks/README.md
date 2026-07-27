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

# One 16 GB GPU, the real Qwen3 pair, tens of minutes
miniverl benchmark benchmarks/configs/gpu_calc_hard.yaml --output runs/benchmarks
```

Each run writes `<name>.json` and `<name>.md` into the output directory. The
Markdown is for humans; the JSON is the artifact to submit.

## What "matched budget" means here

Every arm receives, by construction:

* the same environment, difficulty and `split_seed`, so the same task instances;
* the same **initial weights** — one shared supervised cold start, loaded
  **weights only** so no Adam momentum leaks from the cold start into whichever
  arm happens to resemble it;
* the same number of optimizer steps and the same effective batch size
  (`gradient_accumulation_steps` trajectories per step);
* the same maximum trajectory length, turn limit and rollout bounds;
* the same held-out split, evaluation seed and greedy temperature.

Arms differ **only** in the keys listed under `arms_differ_only_in` in the result
file. Four quantities cannot be matched by construction, so they are measured and
reported per arm instead of being pretended away:

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
| `results/cpu-toy-calc-matched.json` | CPU only | **Parity, not ranking.** All seven arms run to completion under identical budgets on the toy backend. The accuracy differences are within noise and must not be read as a ranking; see the note in the file. |
| `results/rtx4080-calc-hard-matched.json` | RTX 4080 16 GB | Single-seed matched comparison on the chained calculator split with the real Qwen3 pair. **On-policy distillation lost**, with both a standard and a privileged-context teacher; the transcript-level diagnosis is in [`docs/rtx4080-baselines.md`](../docs/rtx4080-baselines.md#matched-budget-comparison). |

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

`schema/benchmark-result.schema.json` is generated from the Pydantic model, so
the two cannot drift:

```bash
miniverl schema --out benchmarks/schema/benchmark-result.schema.json
```

CI regenerates it and fails if the committed copy differs.
