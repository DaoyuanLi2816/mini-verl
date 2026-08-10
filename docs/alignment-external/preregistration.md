# Preregistration: external alignment comparison v1

This document and
[`benchmarks/preregistration/alignment-external-v1.yaml`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/preregistration/alignment-external-v1.yaml)
are the contract for the v0.7.0 study. Once the commit carrying them is public,
the models, methods, task ids, evaluator revisions and thresholds below are
frozen, and the final test may be read exactly once.

Everything here was written before any final-test task was scored. The machine
-readable file is the authority; this page explains the reasoning.

## The question

> Starting from the same non-saturated SFT policy, can continued alignment SFT,
> DPO, offline teacher distillation, standard on-policy distillation, or
> verifier-gated on-policy distillation improve externally measured alignment
> behaviour while controlling over-refusal, retained utility, teacher-query
> cost, GPU time and peak VRAM on one RTX 4080?

It is a question about *when a method is worth its cost*, not a search for a
winner. A result where no method beats continued SFT answers it. So does one
where OPD improves harmful-compliance and pays for it in over-refusal.

## What would make this study worthless

Stating these first, because each is a way to produce a publishable-looking
result that means nothing:

**A saturated starting point.** Alignment Lab v1 began from a policy already at
100% on its suite. Every continuation arm then measured the same ceiling. The
selection gate below exists to prevent a repeat, and it requires headroom in
*both* directions — room to improve on alignment and room to lose on utility.

**Selection on the outcome.** Choosing the best-scoring checkpoint or teacher
would tune the study to its own result. Both are chosen by taking the first
candidate in a committed order that clears a gate fixed in advance.

**A leaked final test.** If a pre-final-test decision were made on a task the
final test also scores, the single read would already be spent. Every selection
suite withholds the frozen final-test ids and records how many.

**Endpoints that secretly agree.** RewardBench contains 404 XSTest-derived
rows. Left in, one behaviour change would move both the over-refusal rate and
the preference win rate, and the two would read as independent corroboration.
They are excluded.

**An unqualified judge.** A 2B safety classifier and a 0.4B pairwise ranker
each have to clear an agreement floor on a calibration subset before their
endpoint counts. Below the floor the endpoint reports `not_applicable`, not a
score.

## Endpoints

Four categories, each pinned to an ungated source at a 40-hex revision. Full
licences, redistribution decisions and rejected candidates are in
[benchmark governance](benchmark-governance.md).

| category | source | evaluator |
| --- | --- | --- |
| instruction following | `google/IFEval` | deterministic rules, no model |
| over-refusal | `natolambert/xstest-v2-copy` | string-match refusal classifier |
| harmful compliance | `JailbreakBench/JBB-Behaviors` | `granite-guardian-3.0-2b` |
| preference / reward | `allenai/reward-bench` | `llm-blender/PairRM`, both orders |

HarmBench, StrongREJECT and AdvBench are gated and could not be pinned;
Llama-Guard-3-1B requires manual approval no reader can be guaranteed. The
harmful-compliance endpoint is therefore **not** reported as HarmBench — its
official classifier is a fine-tuned 13B model, outside the compute contract.

## Methods

All five continuation arms start from the identical selected checkpoint digest
and share, within a seed, the same prompts, splits, sequence limits, optimizer
update count and decoding settings.

```text
starting-sft-checkpoint     zero continuation updates, one deterministic result
continued-alignment-sft     chosen response as a hard target
dpo                         the same pairs, pinned TRL
offline-soft-kd             teacher scores a fixed prompt set once per seed
standard-opd                fresh student states, teacher scores what it visited
verifier-gated-opd          same mechanism, teacher applied only where the
                            preregistered gate fires
```

Seeds are `1234`, `20260727`, `20260801`. The baseline is deterministic and is
evaluated once; it is **not** duplicated three times and presented as seed
variance.

Training data is `Anthropic/hh-rlhf` (MIT), which shares no prompt with any
endpoint. Over-long examples are dropped, never truncated.

## Analysis, fixed in advance

Every arm reports all three seeds, the mean, the min and max, task-level paired
differences against the shared baseline, and 95% paired bootstrap intervals at
a fixed analysis seed.

Three seeds do not support a significance claim, and none is made. Tasks and
training seeds are not pooled as interchangeable samples.

A method is called Pareto-superior only when no displayed primary metric is
worse beyond the tolerance and at least one is better. The retained-utility
non-inferiority tolerance is **2 percentage points**, frozen here.

No composite "alignment score" is the headline. Reducing harmful compliance by
refusing benign prompts is not an improvement, and the two rates are always
shown together.

## If it fails

Preregistered outcomes that are published rather than worked around:

* no candidate clears the saturation gate → publish checkpoint-selection
  failure, run nothing downstream;
* no teacher clears qualification → run the non-teacher baselines, keep the
  failed gates, and have `miniverl pilot` recommend against OPD;
* a judge misses its agreement floor → that endpoint is `not_applicable` for
  every arm;
* an implementation defect invalidates a run → the whole affected set moves to
  `superseded/`, the preregistration gets a public amendment, and every
  affected method and seed reruns from the same starting checkpoint. A
  favourable-looking result from an invalidated execution is not kept.

## Compute

48 GPU hours total on one RTX 4080, 14.5 GiB peak reserved, at most 512
generation tasks per model. The frozen profile uses 508. Measured costs and the
per-phase budget are tracked in `PROJECT_STATE.md`.
