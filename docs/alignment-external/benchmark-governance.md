# External benchmark governance

This page records where every external alignment endpoint comes from, what
licence it carries, what this project redistributes, and what each measurement
does *not* establish. It is the auditable half of
[`benchmarks/external-alignment/registry.yaml`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/external-alignment/registry.yaml),
which is the machine-readable half.

Every revision, licence and gating status below was read from the Hugging Face
Hub API on 2026-08-08. None of it is quoted from a paper, because a paper
describes the benchmark as it was, not as the repository serves it today.

## What runs

| Category | Source | Licence | Revision | Evaluator |
| --- | --- | --- | --- | --- |
| Instruction following | [`google/IFEval`](https://huggingface.co/datasets/google/IFEval) | Apache-2.0 | `966cd89` | Deterministic rules, no model |
| Over-refusal | [`natolambert/xstest-v2-copy`](https://huggingface.co/datasets/natolambert/xstest-v2-copy) | CC-BY-4.0 | `b71afe2` | String-match refusal classifier, no model |
| Harmful compliance | [`JailbreakBench/JBB-Behaviors`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | MIT | `886acc3` | [`ibm-granite/granite-guardian-3.0-2b`](https://huggingface.co/ibm-granite/granite-guardian-3.0-2b) (Apache-2.0, 2B) |
| Preference / reward | [`allenai/reward-bench`](https://huggingface.co/datasets/allenai/reward-bench) | ODC-By | `168d848` | [`llm-blender/PairRM`](https://huggingface.co/llm-blender/PairRM) (MIT, 0.4B) |

Every source is ungated. That is a hard requirement, not a preference: a gated
source cannot be downloaded by a reader trying to reproduce the study, so the
registry validator rejects one.

## What was rejected, and why

Three of the benchmarks a reader might expect are absent. They were evaluated
and refused, and the reasons are recorded so the choice is auditable rather
than looking arbitrary.

**HarmBench, StrongREJECT and AdvBench** (all `walledai/*`) are gated. A
download attempt on 2026-08-08 returned `DatasetNotFoundError` asking for
access. Accepting dataset terms is an authorisation the maintainer gives on
their own account, and a reader would face the same gate, so none of the three
can be a pinned source here.

**`meta-llama/Llama-Guard-3-1B`** fits the parameter limit and would otherwise
be a natural safety classifier, but it is `gated: manual`. Per-user manual
approval means reproduction depends on a decision this project cannot grant.

## Why the harmful endpoint is not called HarmBench

It uses JailbreakBench behaviours judged by Granite Guardian, and it is named
that. Using the HarmBench name would misdescribe the measurement twice over:

- HarmBench's official classifier is a fine-tuned 13B model, well outside a
  one-GPU contract that caps a judge at 3B;
- StrongREJECT's reference judge is a paid API model, which the contract
  forbids outright.

A keyword heuristic or a substituted 2B classifier is not HarmBench, and this
project does not publish it under that name. What it *is* stays on the label:
100 fixed behaviours, no adaptive attack generation, one 2B judge.

## Redistribution

Nothing in this repository republishes benchmark prompt text. For every
endpoint the repository commits:

- the dataset identifier and its exact revision;
- the selected row identifiers and a digest over them;
- evaluator identity and revision;
- aggregate metrics and per-task scores.

It does not commit prompts, and it does not commit model generations. For the
harmful-compliance endpoint in particular, generations are stored as SHA-256
digests only, so a task-level artifact can prove *which* text was scored
without republishing a completion to a jailbreak request.

A reader reproduces the study by fetching the pinned revisions themselves under
the licences above.

## What these numbers do not establish

Each endpoint is bounded, and the bounds are part of the result rather than a
footnote to it.

**Instruction following.** IFEval checks surface constraints — length, casing,
keyword presence, formatting. It says nothing about whether an answer is
correct or useful. The scorer here is an independent implementation of all 25
instruction types at the pinned revision, leaving 0 of 834 instructions
unscored; two verifiers are marked `approximate` and named in every result,
because the reference uses `nltk` for sentence segmentation and `langdetect`
for language identification. `langdetect` is unreliable on short text — "hola
amigo" identifies as Somali — so the 31 language-constrained prompts carry more
evaluator noise than the rest.

**Over-refusal.** Two rates are reported and never averaged: refusing a *safe*
prompt and refusing an *unsafe* one. A policy that refuses everything scores
perfectly on one and uselessly on the other, which is the whole point of
XSTest. The classifier is string matching, not the paper's GPT-4 judge, and its
disagreement against a hand-labelled subset is published with the results.

**Harmful compliance.** One judge, 100 fixed behaviours, no adaptive attacks.
An `unclear` verdict is counted separately and excluded from the rate rather
than folded into "refused", because an undecided classifier is missing
evidence, not a refusal.

**Preference.** PairRM is asked in both orders. A pair it orders inconsistently
is a tie, not a coin flip, and the position-disagreement rate is published
beside every win rate. A PairRM preference is a model's preference; it is never
described as human preference.

## Judge qualification

Both model evaluators are qualified before they count, against a prespecified
calibration subset with floors fixed in the preregistration. A judge below its
floor disqualifies its endpoint, which then reports `not_applicable` — not a
score that nobody should trust, and not a floor lowered after seeing the
number.

## Compute contract

No required model exceeds 1.7B trainable parameters or 3B for an evaluator, no
paid API is used, and final evaluation makes no network call: every model loads
with `local_files_only=True` and `trust_remote_code=False`.
