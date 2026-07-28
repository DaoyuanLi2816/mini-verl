# Protocol-trained teacher adapters

miniVERL can train a Qwen3 LoRA policy on deterministic oracle tool traces,
export it as a standard PEFT adapter, and attach that adapter to a frozen
teacher base model during distillation. Adapter weights are generated artifacts
and are never committed to git. The verified v0.2 adapter is public at
[`DaoyuanLi/mini-verl-qwen3-1.7b-protocol-teacher`](https://huggingface.co/DaoyuanLi/mini-verl-qwen3-1.7b-protocol-teacher),
pinned by miniVERL at immutable revision
`23323751318135484c06c043b1f9b9e7016dd89f`.

## Train and evaluate the teacher policy

```bash
miniverl validate recipes/qwen3_1.7b_protocol_teacher_sft.yaml
miniverl train recipes/qwen3_1.7b_protocol_teacher_sft.yaml --dry-run
miniverl train recipes/qwen3_1.7b_protocol_teacher_sft.yaml \
  --run-id qwen3-1.7b-protocol-teacher
```

The final held-out evaluation, not SFT loss, is the competence record. It writes
strict task success as the primary metric and also records diagnostic lenient
success, valid tool-call rate/count, final-answer format validity and average
turns. Protocol-token accuracy is `null` for free-running trajectories because
there is no aligned token target; the measurement status states that explicitly.
The primary candidate and conditional one-variable fallbacks were fixed before
training in [`protocol-teacher-grid.md`](protocol-teacher-grid.md).

### Measured v0.2 candidate

Candidate A completed 24 optimizer updates on the RTX 4080 and scored **100.0%**
strict success on all 24 held-out tasks, with 100.0% lenient diagnostic success,
100.0% valid tool-call rate, 100.0% final-answer format validity, 33 tool calls
and 2.375 average turns. It passed the prespecified 50% gate, so candidates B
and C were not run.

The exported artifact is intentionally not committed to Git. Its public Hub
copy and reviewable identity are:

| field | SHA-256 |
| --- | --- |
| source checkpoint tree | `e9c42893b861e371dd48e2c151940a198e22eff2f91649ca6a5303c525c5ee4c` |
| `adapter_config.json` | `ca94a103c86a20f0297579a1d05c3ca971a6f1303b2e356b8dd33c644502e939` |
| `adapter_model.safetensors` | `8df7e7bc1b8283b910aa13bc4173083ae20c838bcacb366d7dbcabc7b310b994` |
| miniVERL adapter manifest | `502bca7489c6fe161ebf198d2a1b4622123d4f958885a7e4714c6a02a2e1ac43` |

This competence measurement selected the teacher without looking at downstream
OPD. In the subsequent two-seed comparison, its OPD arm reached 100% on both
seeds and tied the SFT arm; see
[`rtx4080-baselines.md`](rtx4080-baselines.md#protocol-teacher-equal-update-comparison-schema-v2).

## Export

```bash
miniverl export-adapter \
  --run runs/qwen3-1.7b-protocol-teacher \
  --checkpoint runs/qwen3-1.7b-protocol-teacher/checkpoints/final \
  --out artifacts/qwen3-1.7b-protocol-teacher
```

PEFT writes `adapter_config.json` and `adapter_model.safetensors`.
miniVERL adds `miniverl_adapter_manifest.json` with:

- pinned base identity/revision and tokenizer fingerprint;
- source run and checkpoint tree digest;
- LoRA configuration, miniVERL version and git commit;
- training environment/task/protocol;
- the final policy-competence evaluation;
- SHA-256 checksums for both PEFT files.

No pickle format is used.

## Load a frozen teacher from the Hub

```yaml
models:
  teacher:
    model_id: Qwen/Qwen3-1.7B
    revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
    tokenizer_revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
    adapter:
      source: hub
      path: DaoyuanLi/mini-verl-qwen3-1.7b-protocol-teacher
      revision: 23323751318135484c06c043b1f9b9e7016dd89f
      require_policy_evaluation: true
      minimum_strict_success_rate: 0.5
```

For offline use, download the same immutable revision and switch only the
adapter source/path:

```bash
hf download DaoyuanLi/mini-verl-qwen3-1.7b-protocol-teacher \
  --revision 23323751318135484c06c043b1f9b9e7016dd89f \
  --local-dir artifacts/qwen3-1.7b-protocol-teacher
```

```yaml
adapter:
  source: local
  path: artifacts/qwen3-1.7b-protocol-teacher
  require_policy_evaluation: true
  minimum_strict_success_rate: 0.5
```

Before the base model is allocated, miniVERL validates the PEFT files, type,
target modules, base identity/revision, tokenizer fingerprint, checksums and
competence record. Hub metadata and weights are downloaded from the pinned
adapter revision and pass the same checks as a local directory. After
attachment, every teacher parameter is frozen and checked again.

The `minimum_strict_success_rate` in the shipped GPU benchmark is an operational
gate against supervising with a clearly broken protocol policy. Passing it does
not make the teacher “stronger” by definition; the exact teacher metrics are
carried into run and benchmark provenance and must be compared with the student
under the same environment.
