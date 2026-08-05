# Alignment Lab v1: a saturated tool-policy case study

## Abstract

Alignment Lab v1 compares a frozen SFT checkpoint, continued alignment SFT,
DPO, offline soft teacher distillation, standard OPD and verifier-gated OPD
from one checksummed Qwen3-0.6B SFT checkpoint. The final test uses 48 paired
deterministic sandbox policy tasks and three preregistered seeds. Every method
uses the same starting policy and ordered final tasks. The SFT checkpoint
scored 100% policy compliance and 100% retained tool utility in every seed,
leaving no headroom for a continuation method to improve. No method beats that
baseline. Observed mean regressions were retained for: continued SFT, standard OPD, verifier-gated OPD. The experiment therefore identifies cost,
query differences and method-specific regressions but no incremental
alignment-quality advantage. The correct pilot decision is to turn online
teacher querying off for this recipe, not to infer that OPD is broadly
ineffective.

## Study design

- Model: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`
- Shared starting checkpoint: `7304922281268a687dd1c75ba918e26c64c8207b5701db78c368afd20d80ae89`
- Policy: Minipolicy v1, `9a9316bea117928d115eff86291982d7386e6ca2d7127aacb933e508d322c8a8`
- Final test: 48 paired tasks per arm, seeds 1234, 20260727 and 20260801
- Budget: four continuation updates over four effective examples, except the
  frozen SFT checkpoint; DPO uses pinned TRL 1.8.0
- Hardware: one NVIDIA GeForce RTX 4080; cross-GPU generalization was not tested

The policy-conditioned teacher sees the private deterministic rubric. The
student never receives that context. All actions are synthetic sandbox actions;
no real destructive operation is executed. IFEval, XSTest, HarmBench and
RewardBench are represented by pinned metadata adapters only and are **not**
measured endpoints in this artifact.
“Preference win rate” is the deterministic Minipolicy paired outcome, not a
human-preference measurement. Harmful compliance and over-refusal are sandbox
policy checks, not a broad safety result.

## Final result

| method | alignment | harmful compliance | over-refusal | tool utility | teacher query ratio | continuation GPU time | peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT checkpoint | 100.0% | 0.0% | 0.0% | 100.0% | n/a | 0.0 s | 0.63 GiB |
| continued SFT | 94.4% | 0.0% | 0.0% | 88.9% | n/a | 3.9 s | 0.96 GiB |
| DPO | 100.0% | 0.0% | 0.0% | 100.0% | n/a | 8.6 s | 1.64 GiB |
| offline soft distillation | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% | 26.6 s | 0.94 GiB |
| standard OPD | 98.6% | 0.0% | 0.0% | 97.2% | 100.0% | 76.7 s | 0.94 GiB |
| verifier-gated OPD | 97.9% | 0.0% | 0.0% | 95.8% | 46.8% | 66.0 s | 0.87 GiB |

<picture class="alignment-figure">
  <source media="(max-width: 900px)" srcset="../delta-from-sft-mobile.svg">
  <img src="../delta-from-sft.svg" alt="Forest chart of alignment and retained-tool-utility percentage-point deltas from the saturated SFT checkpoint. Every method's three seeds and their means are drawn at their exact values and printed as text; no continuation method lands above the zero baseline.">
</picture>

<picture class="alignment-figure">
  <source media="(max-width: 900px)" srcset="../outcome-cost-matrix-mobile.svg">
  <img src="../outcome-cost-matrix.svg" alt="Row matrix of alignment, retained tool utility, teacher-query ratio, continuation GPU time and peak VRAM for every method, with each value printed next to its bar and non-teacher query ratios marked not applicable rather than zero.">
</picture>

### Metric coverage

<div class="coverage" markdown="0">
<p class="coverage-statement">Both measured sandbox safety checks tied at zero while retained utility still regressed. External IFEval, XSTest, HarmBench and RewardBench endpoints were not executed.</p>
<div class="coverage-scroll">
<table class="coverage-table">
<caption>Alignment Lab v1 metric coverage. Every measured seed is printed; the two sandbox rates are identical across all seeds and methods.</caption>
<thead><tr><th scope="col">Method</th><th scope="col">Harmful compliance<span>seed values</span></th><th scope="col">Over-refusal<span>seed values</span></th><th scope="col">Retained tool utility change<span>mean (seeds), pp</span></th></tr></thead>
<tbody>
<tr><th scope="row" data-label="Method"><span class="coverage-swatch" style="background:#A7A9AC"></span>SFT checkpoint</th><td data-label="Harmful compliance">0% · 0% · 0%</td><td data-label="Over-refusal">0% · 0% · 0%</td><td data-label="Retained tool utility change"><b>0.0</b> <span class="coverage-seeds">(0.0 / 0.0 / 0.0)</span></td></tr>
<tr><th scope="row" data-label="Method"><span class="coverage-swatch" style="background:#0072B2"></span>continued SFT</th><td data-label="Harmful compliance">0% · 0% · 0%</td><td data-label="Over-refusal">0% · 0% · 0%</td><td data-label="Retained tool utility change"><b>-11.1</b> <span class="coverage-seeds">(0.0 / 0.0 / -33.3)</span></td></tr>
<tr><th scope="row" data-label="Method"><span class="coverage-swatch" style="background:#CC79A7"></span>DPO</th><td data-label="Harmful compliance">0% · 0% · 0%</td><td data-label="Over-refusal">0% · 0% · 0%</td><td data-label="Retained tool utility change"><b>0.0</b> <span class="coverage-seeds">(0.0 / 0.0 / 0.0)</span></td></tr>
<tr><th scope="row" data-label="Method"><span class="coverage-swatch" style="background:#E69F00"></span>offline soft distillation</th><td data-label="Harmful compliance">0% · 0% · 0%</td><td data-label="Over-refusal">0% · 0% · 0%</td><td data-label="Retained tool utility change"><b>0.0</b> <span class="coverage-seeds">(0.0 / 0.0 / 0.0)</span></td></tr>
<tr><th scope="row" data-label="Method"><span class="coverage-swatch" style="background:#D55E00"></span>standard OPD</th><td data-label="Harmful compliance">0% · 0% · 0%</td><td data-label="Over-refusal">0% · 0% · 0%</td><td data-label="Retained tool utility change"><b>-2.8</b> <span class="coverage-seeds">(0.0 / 0.0 / -8.3)</span></td></tr>
<tr><th scope="row" data-label="Method"><span class="coverage-swatch" style="background:#009E73"></span>verifier-gated OPD</th><td data-label="Harmful compliance">0% · 0% · 0%</td><td data-label="Over-refusal">0% · 0% · 0%</td><td data-label="Retained tool utility change"><b>-4.2</b> <span class="coverage-seeds">(0.0 / -12.5 / 0.0)</span></td></tr>
</tbody>
</table>
</div>
<ul class="coverage-scope">
<li><b>Sandbox endpoint measured:</b> yes, for every row — deterministic Minipolicy v1 harmful-compliance and over-refusal checks.</li>
<li><b>External safety benchmark executed:</b> no, for every row — IFEval, XSTest, HarmBench and RewardBench are pinned metadata only.</li>
</ul>
</div>

<details>
<summary>Figure provenance</summary>

- Result SHA-256: `584752dccb91654109c357b8ebb12681a12a9c1476a9ba539dd35e4d860a22ef`
- Task-level result SHA-256: `8d7fc723436d7377d196fc44046d960e3cb7f0aa81e03d49ef05b627eb84630f`
- Three seed identities: `1234`, `20260727`, `20260801`

</details>

The starting checkpoint defines a ceiling; overlapping continuation points are
not evidence of algorithmic equivalence, and every non-overlapping regression
is retained. DPO cost includes its external pinned TRL training job; evaluation
time is excluded from the continuation-GPU-time axis. Teacher-query ratio
counts selected target positions and does not imply a proportional reduction
in teacher backbone FLOPs.

## State x Supervision diagnostic

The six-arm result directly observes oracle hard targets (continued SFT),
frozen-state preference supervision (DPO), frozen-state soft distributions
(offline distillation) and fresh-state soft distributions (standard OPD).
The frozen-soft and fresh-soft means are reported, but a task ceiling can make
their difference uninformative. The required hard-state comparisons remain
explicit:

- frozen hard vs fresh hard: teacher-argmax/student-token agreement
  `1.0000` vs `1.0000`
  (fresh minus frozen `+0.0000`)
- frozen soft vs fresh soft: bucketed teacher entropy
  `0.0023` vs `0.0022` nats
  (fresh minus frozen `-0.0002`)
- fresh hard vs fresh soft: matched soft targets retain
  `0.025%`
  mean probability mass beyond the teacher argmax

No soft-target advantage is claimed. Verifier-gated OPD is separately treated
as a localized soft-supervision method, not relabeled as a hard-target cell.

## Verifier-gated OPD and pilot decision

The `policy-critical-span-v1` gate was calibrated on eval and frozen before the
test read. It records a decision for every example/span. Gating reduces queried
positions relative to standard OPD. Any resulting policy or utility regression
is retained rather than hidden; gating does not establish a general quality
gain or a proportional compute saving.

The versioned `alignment-pilot-v1` rule returns
`recommendation: insufficient_evidence`, followed by the operational decision:
do not spend online teacher-query cost on this already-saturated recipe. A more
discriminating policy suite would be required before choosing DPO, offline
distillation or either OPD variant.

The pilot binds 48 tasks, three seeds, measured continuation time, peak VRAM and
teacher-query fraction. Free-running teacher policy competence, the resulting
teacher-student policy gap, distribution-level top-k overlap, independent gate
precision and a population uncertainty interval were **not measured**; the JSON
records them as `null`, never as zero. The no-continuation result follows the
versioned less-than-2% headroom rule before those missing fields are consulted.

## Preserved deviation and negative evidence

Completed final-test regressions:

- `verifier_gated_opd` seed `20260727`: 93.8%; 3 failed task(s), policy categories `safe_error_recovery`.
- `continued_sft` seed `20260801`: 83.3%; 8 failed task(s), policy categories `safe_error_recovery`.
- `standard_opd` seed `20260801`: 95.8%; 2 failed task(s), policy categories `safe_error_recovery`.

The first seed-1234 SFT baseline evaluated the first 24 tasks because the base
recipe initially allocated only 24 test tasks. The run then stopped before any
other method evaluated test. Preregistration revision 1.4 publicly froze a
recovery rule: preserve tasks 0-23, evaluate only tasks 24-47 once, and combine
the disjoint segments. The result contains all 48 unique task IDs with zero
repeated test tasks. The interrupted continued-SFT construction run is retained
outside the headline result and was never evaluated.

The primary negative finding is preserved: no continuation method improves the
already-saturated SFT checkpoint, and any completed regression remains in the
headline result. It would be misleading to turn a lower teacher-query ratio
into a quality claim.

## Scope and limitations

- One small model family, one deterministic sandbox policy suite, one GPU and
  three seeds do not support broad safety or population claims.
- The suite's deterministic validators are valuable for auditability but are
  too easy for the common SFT checkpoint.
- External safety, preference, instruction-following and general-capability
  suites were not executed; their licenses/revisions are metadata only here.
- GPU time and peak VRAM are observed values for this machine and software
  stack, not forecasts for other cards.
- Localized or verifier-qualified distillation is not claimed as novel.

## Reproducibility and artifacts

- Result SHA-256: `584752dccb91654109c357b8ebb12681a12a9c1476a9ba539dd35e4d860a22ef`
- Task-level result SHA-256: `8d7fc723436d7377d196fc44046d960e3cb7f0aa81e03d49ef05b627eb84630f`
- Preregistration SHA-256: `71307dbfe9a5bb20c686307cafce8bd254c07af8b69c1bf1c6ec0dbf53a8cde0`
- Immutable calculator artifact SHA-256: `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`

Regenerate and compare every figure and public Alignment Card with:

```bash
python scripts/publish_alignment_lab_artifacts.py --check
```

The machine-readable result records raw run-artifact hashes, DPO provenance,
the two-segment recovery, all 18 completed arms and all 864 task-level rows.
