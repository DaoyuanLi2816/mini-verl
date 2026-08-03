# TODO

This is the post-v0.6 roadmap. Completed RecoveryBench, Consumer Runtime,
Alignment Lab and pinned verl-bridge work lives in `PROJECT_STATE.md` and
`CHANGELOG.md`, not in this active list.

## Scientific follow-up

- [ ] [Preregister and execute real external alignment endpoints](https://github.com/DaoyuanLi2816/mini-verl/issues/39) (for example
      IFEval, XSTest, HarmBench and RewardBench) in a future release; keep them
      separate from Alignment Lab v1's sandbox-policy checks.
- [ ] Evaluate a less saturated task family with at least three prespecified
      seeds, eval-only selection and a reserved one-read test split.
- [ ] Extend matched-budget JSON-navigation and SQLite evidence only when the
      design adds information beyond the published RecoveryBench study.
- [ ] Repeat the measured GPU recipes on Linux; do not treat the expected
      throughput improvement as measured until those artifacts exist.

## Runtime scope

- [ ] Cross-tokenizer distillation with an explicit alignment contract.
- [ ] Batched or engine-backed rollout decoding; v0.4 batches update forwards,
      while rollout generation remains deliberately sequential.
- [ ] Additional tested model families beyond Qwen2/Qwen3.
- [ ] Entropy-aware divergence mixing after a prespecified experiment; current
      code records teacher entropy but does not implement the method.
- [ ] A maintained GPU CI runner. GPU tests remain opt-in until real CUDA
      infrastructure is configured.

Multi-GPU training, Ray, FSDP, DeepSpeed, vLLM and PPO/GRPO remain intentionally
out of scope; use verl or another distributed training system for those needs.
