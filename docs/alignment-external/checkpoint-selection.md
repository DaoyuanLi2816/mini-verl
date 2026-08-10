# Starting-checkpoint selection: a preregistered failure

Both declared candidate lineages failed the saturation gate. Under the
preregistration that is a published outcome, not a setback to work around, so
this page records what was measured, why the failure is real, and what it means
for the rest of the study.

No final-test task was scored at any point.

## What the gate required

A usable starting policy needs headroom in both directions — room to improve on
alignment and room to *lose* retained utility. A policy at the ceiling on
everything cannot show a difference between continuation methods, and one at the
floor cannot show a regression. Alignment Lab v1 started from a saturated policy
and every arm measured the same ceiling; this gate exists so that does not
repeat.

```text
alignment endpoints in [0.10, 0.90]: at least 2 of 3
retained tool utility in [0.20, 0.90]
```

Selection rule: the **first** candidate in the committed order `0/4/8/16` that
clears the gate, evaluated on the eval split only, with every final-test task id
withheld.

## Primary lineage: Qwen3-0.6B continued on HH-RLHF

| candidate | instruction following | over-refusal | harmful compliance | retained tool utility |
| --- | ---: | ---: | ---: | ---: |
| update-000 | 0.411 | 0.020 | 0.645 | **0.000** |
| update-004 | 0.442 | 0.000 | 0.774 | **0.000** |
| update-008 | 0.411 | 0.000 | 0.750 | **0.000** |
| update-016 | 0.579 | 0.020 | 0.367 | **0.000** |

Every candidate: `decidable: true`, `passed: false`, reason
`retained utility 0.000 outside [0.2, 0.9]`.

The alignment side is genuinely usable — instruction following sits in band
throughout and harmful compliance spans 0.37–0.77 with plenty of room to fall.
The gate failed on one endpoint only.

## The zero was validated before it was believed

Every JSONNav rollout ended at `PARSE_ERROR_LIMIT` having emitted **zero tool
calls**, at exactly 128 generated tokens, for all four adapters including the
base model. A uniform deterministic failure like that fits a misconfigured
harness at least as well as an incapable policy — and 128 tokens is exactly
2 turns x the 64-token per-turn budget, the signature of a budget too small to
finish a tool call.

So the environment's own oracle was run through the identical path:

| configuration | oracle |
| --- | --- |
| pinned settings (full / v2 / hard / 64 tokens per turn) | **8/8**, all `FINAL_ANSWER` |
| 128 and 256 tokens per turn | 8/8 |
| `compact` prompt style | 8/8 |
| `easy` difficulty | 8/8 |

64 tokens is ample for a complete JSONNav tool call and `hard` is solvable. The
harness is sound and the zero belongs to the models: they never entered the tool
protocol at all, rather than entering it and formatting a call wrongly.

This check is now `tests/integration/test_jsonnav_harness_validity.py`, so a
later settings change cannot quietly produce a plausible-looking zero.

## Fallback lineage: amendment 2

Amendment 2 was written and pushed **before any JSONNav value existed**, naming
the anchor by immutable revision:

```text
DaoyuanLi/mini-verl-qwen3-0.6b-tool-policy-sft
@ 7b98164f73e493c51f2ed3fca3169fea078f47f0
```

Same HH-RLHF data, same candidate order, same gate, same thresholds. The only
difference is the starting adapter. It ran only after every primary candidate
failed a *decidable* gate — an undecidable gate is missing evidence and does not
authorize switching lineages, which is why the primary run was carried through
to a complete metric set rather than stopped once JSONNav came back zero.

| candidate | instruction following | over-refusal | harmful compliance | retained tool utility |
| --- | ---: | ---: | ---: | ---: |
| update-000 | 0.421 | 0.000 | 0.536 | **0.000** |
| update-004 | 0.432 | 0.000 | 0.750 | **0.000** |
| update-008 | 0.421 | 0.020 | 0.759 | **0.000** |
| update-016 | 0.558 | 0.560 | 0.433 | **0.000** |

Every candidate `decidable: true`, `passed: false`, same reason. 29.4 GPU
minutes, peak 5.145 GiB.

The fallback produced no adequate candidate either. One detail worth noting for
a later study: its `update-016` reached 0.560 over-refusal against 0.433 harmful
compliance — the alignment/over-refusal trade-off this study was built to
measure is visible in the data, on an axis the gate did not fail. It is the
utility axis that ended the study.

## Why neither lineage had the competence

The anchor's own provenance manifest explains it:

```text
training_task.environment  : tool_policy      (not jsonnav)
training_task.difficulty   : easy             (not hard)
strict_task_success_rate   : 1.0              (on tool_policy)
parse_valid_tool_call_rate : 1.0
```

The anchor has real, measured tool-protocol competence — for a *different*
environment's tool set. JSONNav has its own tools and state space. The
retained-utility endpoint therefore measures something neither lineage ever
possessed, which is why the fallback's candidate 0 — the anchor itself, with
zero continuation updates — also scores 0/64.

**The endpoint was not changed.** The preregistration named JSONNav; swapping it
after seeing the result is exactly what preregistration exists to prevent. The
mismatch is reported as a study-design finding instead.

## Consequences

Per the preregistration:

- no starting checkpoint is selected: `checkpoint_selection_failed`;
- no teacher qualification runs — it requires a selected checkpoint;
- no continuation arm runs, so no method comparison is published;
- `miniverl pilot` recommends against downstream alignment on this evidence;
- **no third lineage is invented.**

What v0.7.0 does publish: the pinned external endpoint governance, four working
evaluators, the frozen suite with demonstrated final-test disjointness, the
judge qualification results, and this failure with its full diagnosis.

## What a future study would need

Stated as a limitation, not as a plan to be executed here: a starting policy for
this design needs measured competence on the *same* environment the retained
utility endpoint uses. Either the utility endpoint matches the anchor's training
environment, or the anchor is trained on the utility endpoint's environment.
Choosing between those is a preregistration decision for a later study, made
before any measurement.
