# TODO

This is the post-v0.2.1 research and engineering backlog. Completed release,
publication, protocol-teacher and two-seed benchmark work lives in
`PROJECT_STATE.md` and `CHANGELOG.md`, not in this active list.

## Scientific follow-up

- [ ] Select future protocol-teacher candidates on the `eval` split and reserve
      `test` for downstream reporting.
- [ ] Evaluate on a less saturated task family with at least three prespecified
      seeds; retain SFT, raw-teacher and protocol-aligned controls.
- [ ] Run JSON-navigation and SQLite real-model comparisons with the same
      matched-budget and immutable-artifact discipline.
- [ ] Repeat the measured GPU recipes on Linux; do not treat the expected
      throughput improvement as measured until those artifacts exist.

## Runtime scope

- [ ] Cross-tokenizer distillation with an explicit alignment contract.
- [ ] Padded multi-trajectory batching; today gradient accumulation supplies
      the effective batch size.
- [ ] Additional tested model families beyond Qwen2/Qwen3.
- [ ] Entropy-aware divergence mixing after a prespecified experiment; current
      code records teacher entropy but does not implement the method.
- [ ] A maintained GPU CI runner. GPU tests remain opt-in until real CUDA
      infrastructure is configured.

Multi-GPU training, Ray, FSDP, DeepSpeed, vLLM and PPO/GRPO remain intentionally
out of scope; use verl or another distributed training system for those needs.
