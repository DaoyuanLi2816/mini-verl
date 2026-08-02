# RecoveryBench v1 preregistration

This document records the experimental choices for RecoveryBench before any
final `test`-split model run. The canonical machine-readable record is
[`benchmarks/preregistration/recoverybench-v1.yaml`](../../benchmarks/preregistration/recoverybench-v1.yaml).

## Question and endpoints

RecoveryBench asks whether supervision on fresh states visited by the current
student improves recovery from tool errors compared with distillation on fixed
states collected once from the shared cold-start student. Recovery after a tool
error and strict task success are primary; validity, efficiency, query volume,
time, and peak VRAM are reported rather than hidden behind one score.

The comparison uses three fixed student seeds, one paired task schedule, the
same cold-start checkpoint within each seed, and one protocol-qualified teacher
for every KD/OPD arm. The historical calculator raw-teacher result is not a
RecoveryBench arm.

## Locked design

- Seeds: `1234`, `20260727`, and `20260801`; split seed `20260801`.
- Splits: 256 train, 96 eval, and 128 final test tasks.
- Templates: registry v1, digest
  `396c49c14641f8a282706e635f164ac72aa6f42d5c5b0fe2abd0c8f667168242`.
- Training schedule digest:
  `0d937493eea9daadc36e7d29c7c4dace6e35b80a43e27e8af3b038b465f9a806`.
- Primary budget: eight continuation optimizer updates after 24 shared SFT
  cold-start updates.
- Headline arms: cold start only, continued SFT, oracle offline KD,
  frozen-student offline KD, strict fresh-state OPD, and a strict OPD arm capped
  at 50% of full model-generated teacher-query positions.

The two secondary numeric budgets are deliberately not guessed. They are
derived once from eval-only calibration by the exact rules in the YAML, then
committed before any final test run. Step-boundary overshoot is retained.

## Teacher gate

Candidates are tried in the recorded order on eval only. The first candidate
with at least 80% strict success, 75% recovery after error, 95% parse-valid tool
calls, and 70% successful tool execution is frozen at an immutable Hub revision.
Failed candidates and their costs remain part of the record.

Revision 1.1, still before any final-test run, locks candidate inference and
all downstream teacher use to an NF4 base. Candidate adapters are trained by
QLoRA; a portability check showed that applying the same adapter to an
unquantized base deploys a different policy. The completed full-precision
candidate-A reapplication is retained as a failed, noncanonical diagnostic.
Teacher qualification uses deterministic decoding with five turns, 96 new
tokens per turn, and 1,536 total tokens. The separately locked 64/896 limits
apply to student-arm training and final-test rollouts, not to the teacher gate.

## Analysis and invalidation

Every arm runs all three seeds. Reports retain per-seed values, mean, range, and
10,000-replicate paired task bootstrap intervals for recovery and strict
success. Three seeds alone do not justify a significance claim.

The experiment is invalid if test results are inspected before the teacher,
templates, schedule, hyperparameters, budgets, plots, and analysis code are
frozen, or if any declared provenance check fails. Completed negative or failed
arms are never removed. A correction creates a new versioned artifact rather
than rewriting the original.
