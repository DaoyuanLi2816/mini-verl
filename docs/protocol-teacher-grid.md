# Prespecified protocol-teacher grid

Written before the first v0.2 protocol-teacher training run. The selection
target is held-out strict tool-policy success, not SFT loss and not the
downstream OPD benchmark.

All candidates use the pinned Qwen3-1.7B base, seed 1234, NF4 LoRA rank 16,
the deterministic `hard` calculator oracle traces, eight traces per optimizer
update, greedy evaluation on the same 24 held-out test tasks, and otherwise the
complete config in `recipes/qwen3_1.7b_protocol_teacher_sft.yaml`.

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
