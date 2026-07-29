# Prespecified protocol-teacher grid

Written before the first v0.2 protocol-teacher training run. The selection
target is held-out strict tool-policy success, not SFT loss and not the
downstream OPD benchmark.

All candidates use the pinned Qwen3-1.7B base, seed 1234, NF4 LoRA rank 16,
the deterministic `hard` calculator oracle traces, eight traces per optimizer
update, greedy evaluation on the same 24 held-out test tasks, and otherwise the
complete config in `recipes/qwen3_1.7b_protocol_teacher_sft.yaml`.

That `test` choice is retained as historical provenance, not recommended
methodology. The same 24 tasks were later used in the downstream v0.2
benchmark, so they were not a completely untouched final test set. Candidate A
passed on the first prespecified attempt and no fallback tuning occurred.
Future grids select on `eval` and reserve `test` for downstream reporting.

| candidate | optimizer updates | learning rate | changed from A |
| --- | ---: | ---: | --- |
| A (primary) | 24 | `1e-4` | none |
| B (longer) | 48 | `1e-4` | `train.cycles` only |
| C (lower LR) | 24 | `5e-5` | `train.learning_rate` only |

Execution rule:

1. Run A once.
2. If A reaches at least 50% strict success, export A and do not train B or C.
3. If A misses the gate, run both B and C from the same pinned base.
4. Select the highest strict success; break ties by valid tool-call rate, then
   final-answer format validity, then lower training wall time.
5. Preserve every attempted run. Do not choose a teacher by looking at the OPD
   arm, and do not call a passing teacher stronger without reporting its exact
   held-out metrics.

For B and C, copy the primary recipe and change only the field named in the
table. The resolved config, git commit, checkpoint digest and final evaluation
are preserved in the run and exported-adapter manifests, so the one-variable
diff is independently checkable.

## Execution record

Candidate A was run once on 2026-07-27 and reached **100.0% strict held-out
success (24/24)**, 100.0% parse-valid tool-call rate, 100.0% execution success
and 100.0% final-answer format validity. It therefore passed the 50% gate. Per
the rule above, A was exported
and candidates B and C were not run. The downstream OPD result was not consulted
when making that decision; hashes and the full competence record are in
[`teacher-adapters.md`](teacher-adapters.md).
