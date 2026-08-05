# RecoveryBench v1

RecoveryBench asks a narrow mechanism question: when the starting checkpoint,
teacher, task schedule, optimizer and update count are controlled, does scoring
fresh states visited by the current student improve SQLite tool-error recovery
over distillation on states frozen from the cold-start student?

It is not an alignment benchmark and it does not treat SFT and post-SFT OPD as
interchangeable stages. SFT establishes task and protocol competence; OPD is a
teacher-student mechanism whose transferred behavior depends on the teacher.

## Result

The preregistered primary hypothesis was not supported. Under eight equal
optimizer updates, frozen-student KD outperformed strict fresh-state OPD on both
primary endpoints:

| method | strict success, mean (seed range) | recovery after error, mean |
| --- | ---: | ---: |
| cold start | 10.7% (5.5-21.1%) | 13.6% |
| continued oracle SFT | 4.9% (0.0-9.4%) | 1.8% |
| oracle-state offline KD | **33.1%** (0.0-54.7%) | **31.9%** |
| frozen-student-state KD | **23.2%** (9.4-36.7%) | **22.8%** |
| strict fresh-state OPD | 10.9% (0.0-26.6%) | 9.1% |
| strict fresh-state OPD, 50% positions | 27.3% (10.9-49.2%) | 20.7% |

The task-paired difference, fresh OPD minus frozen-student KD, was -12.24
percentage points for strict success (95% paired bootstrap interval -15.89 to
-8.59; 384 pairs) and -13.79 points for recovery after error (-20.69 to -6.90;
116 pairs where both trajectories had a structured tool error). These intervals
describe paired task outcomes; three student seeds do not by themselves justify
a broad significance or generalization claim.

<picture>
  <source media="(max-width: 900px)" srcset="../recovery-success-mobile.svg">
  <img src="../recovery-success.svg" alt="Three-seed mean strict task success and recovery-after-error rate for every RecoveryBench arm, with each rate printed next to its bar.">
</picture>

## Controlled design

- Student: `Qwen/Qwen3-0.6B` at
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Teacher base: `Qwen/Qwen3-1.7B` at
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Qualified teacher adapter:
  [`DaoyuanLi/mini-verl-qwen3-1.7b-sqlite-recovery-teacher`](https://huggingface.co/DaoyuanLi/mini-verl-qwen3-1.7b-sqlite-recovery-teacher/tree/eb2747895ec32dab47c5b50c2d8aa9c0d9701e0d)
  at immutable revision `eb2747895ec32dab47c5b50c2d8aa9c0d9701e0d`;
  adapter weights SHA-256
  `5355f7007efb904d1b45a1aeb9b73b479b6f52025ab92502ab7895706155b2ba`.
- Seeds: `1234`, `20260727`, `20260801`; fixed split seed `20260801`.
- Tasks: 256 train, 96 eval and 128 final test tasks from 12 structurally
  disjoint templates.
- Same cold-start checkpoint and paired final-test task IDs within each seed.
- Primary: eight continuation updates. Secondary: 6,224 selected positions and
  a preregistered 50-second continuation target.

The selected teacher passed the independent eval-only gate at 90.6% strict
success, 81.2% recovery after error, 100% parse-valid tool calls and 87.1% tool
execution success. The historical calculator teacher failed the same gate. No
test outcome was used to select the teacher.

## Budget views and cost

The equal-selected-position view reached the 6,224 target at optimizer-step
boundaries for all nine runs. Overshoot ranged from 0 to 646 positions. Because
all three methods reached the boundary after eight steps, its quality outcomes
match the primary view while independently preserving task-level execution
evidence.

The nominal wall-time view needs a precise caveat. Fresh OPD crossed the
50-second continuation gate in one indivisible step (88.36-121.30 seconds on
the internal continuation timer; 114.89-162.67 seconds for the complete train
call). SFT and frozen KD instead completed the preregistered eight-cycle ceiling
before their internal continuation timer crossed 50 seconds; their complete
train calls averaged 51.92 and 51.63 seconds. This is a cycle-capped wall-time
diagnostic, not an exact equal-time comparison. The completed artifact is
preserved as run, per the steering instruction not to restart or reconfigure the
frozen final experiment.

Under equal updates, strict fresh OPD averaged 686.80 continuation seconds,
versus 52.10 seconds for frozen KD. The budget-50 position selector queried
49.77% of model-generated positions but averaged 720.76 seconds; selecting fewer
positions did not reduce the required teacher backbone forwards and did not
reduce wall time in this implementation.

Teacher preparation cost is reported separately: 2,310.75 seconds once,
462.15 seconds amortized over five students, or 231.07 seconds over ten. A
687.05-second diagnostic is excluded from that selected-teacher preparation
total and remains recorded.

<picture>
  <source media="(max-width: 900px)" srcset="../cost-quality-pareto-mobile.svg">
  <img src="../cost-quality-pareto.svg" alt="Three-seed mean recovery-after-error rate paired with continuation training time as a fraction of the slowest arm, with both values printed next to their bars.">
</picture>

<picture>
  <source media="(max-width: 900px)" srcset="../fresh-vs-frozen-mobile.svg">
  <img src="../fresh-vs-frozen.svg" alt="Paired strict-task-success and recovery-after-error differences between fresh-state OPD and frozen-state KD. Both the mean difference and the upper 95 percent bootstrap bound stay below zero and are printed as text.">
</picture>

## Interpretation

RecoveryBench isolates the value and cost of fresh student-visited states. On
this Qwen3 pair, SQLite recovery task family and RTX 4080, fresh-state OPD did
not justify its added cost. Oracle-state offline KD had the highest three-seed
mean, while the 50%-position fresh arm was highly seed-variable. This result
does not show that OPD is generally ineffective, that offline KD always wins,
or that ordinary task success is the right endpoint for alignment.

The calculator benchmark remains a separate case study: teacher protocol
qualification prevented collapse but only tied continued SFT on a saturated
task. Together, the two studies motivate qualification and pilot evidence
before choosing an online teacher-student mechanism.

## Immutable artifacts

| artifact | SHA-256 |
| --- | --- |
| [equal updates](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/recoverybench-v1-equal-updates.json) | `6ce2e6837e12b99ebc4fad6d27ce3e69c92e295ff3b9b60e0f68c2d308022384` |
| [equal selected tokens](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/recoverybench-v1-equal-selected-tokens.json) | `fe4c9afc799724dfe7a32e631676a1e5177c44559a7374d2ea31da135354f137` |
| [wall-time diagnostic](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/recoverybench-v1-equal-wall-time.json) | `425b0fa568f37b09e61af731d3da5009bd3833bddde6efaf2c66e9dba8355cbe` |
| [compact task results](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/recoverybench-v1-task-results.jsonl) | `aff96bffc6da27240a852410ac041bd4d95badf34cad030e6f437be1491a55ad` |
| [paired analysis](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/recoverybench-v1-analysis.json) | `8a6891f74aed80f07ec00d5ea1909895c579346e1abbb1d5d95a354bb46c6b81` |

The three raw task-artifact sets contain 36 files and 4,608 trajectories. Their
individual hashes remain embedded in the schema-v3 result JSON. The
preregistration is commit `7087b3a333463b88a62ffed73daee2c85d039145`, revision
1.3 digest `9c4c2ec19a56cebb2b2c1c0f3c7e504a9285467c99ae1590488251fbf2ff3934`.

Reproduce the committed analysis and figures with:

```bash
python scripts/publish_recoverybench_artifacts.py \
  --equal-updates benchmarks/results/recoverybench-v1-equal-updates.json \
  --equal-selected-tokens benchmarks/results/recoverybench-v1-equal-selected-tokens.json \
  --equal-wall-time benchmarks/results/recoverybench-v1-equal-wall-time.json
```

See the [preregistration](preregistration.md),
[machine-readable schema](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/schema/recoverybench-result.schema.json)
and [technical report](https://github.com/DaoyuanLi2816/mini-verl/blob/main/paper/recoverybench-v1/README.md) for the full
method and limitations.
