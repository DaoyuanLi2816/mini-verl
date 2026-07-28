# Protocol-trained teacher adapters

miniVERL can train a Qwen3 LoRA policy on deterministic oracle tool traces,
export it as a standard PEFT adapter, and attach that adapter to a frozen
teacher base model during distillation. Adapter weights are generated artifacts
and are never committed to git.

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

## Load a frozen teacher

```yaml
models:
  teacher:
    model_id: Qwen/Qwen3-1.7B
    revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
    tokenizer_revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
    adapter:
      source: local
      path: artifacts/qwen3-1.7b-protocol-teacher
      require_policy_evaluation: true
      minimum_strict_success_rate: 0.5
```

Before the base model is allocated, miniVERL validates the local files, PEFT
type, target modules, base identity/revision, tokenizer fingerprint, checksums
and competence record. After attachment, every teacher parameter is frozen and
checked again. A Hub adapter is also supported when both its adapter revision
and base revision are pinned.

The `minimum_strict_success_rate` in the shipped GPU benchmark is an operational
gate against supervising with a clearly broken protocol policy. Passing it does
not make the teacher “stronger” by definition; the exact teacher metrics are
carried into run and benchmark provenance and must be compared with the student
under the same environment.
