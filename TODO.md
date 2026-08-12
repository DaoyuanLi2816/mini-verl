# TODO

This is the post-v0.6 roadmap. Completed RecoveryBench, Consumer Runtime,
Alignment Lab and pinned verl-bridge work lives in `PROJECT_STATE.md` and
`CHANGELOG.md`, not in this active list.

## Scientific follow-up

- [ ] [External alignment issue #39](https://github.com/DaoyuanLi2816/mini-verl/issues/39):
      v0.7.0 preregistered and executed the starting-checkpoint gate, which
      failed before teacher/method training. Method-level coverage, evaluator
      qualification and the reserved final test remain unexecuted; keep the
      issue open without treating a follow-up study as committed work.
- [ ] Evaluate a less saturated task family with at least three prespecified
      seeds, eval-only selection and a reserved one-read test split.
- [ ] Extend matched-budget JSON-navigation and SQLite evidence only when the
      design adds information beyond the published RecoveryBench study.
- [ ] Repeat the measured GPU recipes on Linux; do not treat the expected
      throughput improvement as measured until those artifacts exist.

## Runtime scope

- [ ] Cross-tokenizer distillation with an explicit alignment contract.
- [ ] Engine-backed decoding beyond the current padded local-HF prompt batches;
      tool-environment multi-turn generation remains deliberately sequential.
- [ ] Additional tested model families beyond Qwen2/Qwen3.
- [ ] Entropy-aware divergence mixing after a prespecified experiment; current
      code records teacher entropy but does not implement the method.
- [ ] A maintained GPU CI runner. GPU tests remain opt-in until real CUDA
      infrastructure is configured.

Multi-GPU training, Ray, FSDP, DeepSpeed, vLLM and PPO/GRPO remain intentionally
out of scope; use verl or another distributed training system for those needs.
