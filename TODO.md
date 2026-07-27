# TODO

Everything the v0.1.0 scope required is done; see `PROJECT_STATE.md` for the
evidence table. This file is what is left, split into what is blocked on someone
with credentials and what is genuinely future work.

## Blocked on credentials or infrastructure

These cannot be completed from a commit. Each one names the exact command or
step, so nothing has to be rediscovered.

- [ ] **Publish to PyPI.** Register a trusted publisher on the PyPI project
      (`DaoyuanLi2816` / `mini-verl` / `release.yml` / environment `pypi`), then
      add the `publish` job described in `docs/release-checklist.md`. Re-check
      that the name `miniverl` is still free first; it was on 2026-07-27
      (`https://pypi.org/pypi/miniverl/json` returned 404). If taken, publish as
      `mini-verl-opd` and change only `[project] name`.
- [ ] **Push the repository and create the `v0.1.0` release.** The tag must be
      `v0.1.0`; the release workflow will refuse it otherwise.
- [ ] **Repository settings.** Description:
      `On-policy distillation for tool-using LLM agents on one consumer GPU.`
      Topics: `llm`, `on-policy-distillation`, `knowledge-distillation`,
      `agentic-rl`, `tool-use`, `qlora`, `consumer-gpu`, `post-training`,
      `qwen`, `llm-agents`, `verl`. Social preview: `docs/banner.svg` rendered
      to PNG.
- [ ] **A GPU CI runner.** `.github/workflows/gpu.yml` targets a self-hosted
      runner labelled `cuda`. Until one is registered the workflow stays queued,
      which is expected. The same tests were run locally; results are in
      `docs/rtx4080-baselines.md`.

## Measurement gaps worth closing

Not required for the release, but each would make a claim stronger.

- [ ] **More seeds on GPU.** Every GPU number is single-seed. Three seeds on
      `benchmarks/configs/gpu_calc_hard.yaml` would turn "measured" into
      "measured with a variance estimate".
- [ ] **A Linux measurement.** Decoding on the measured Windows machine is
      kernel-launch bound (a 14-token prefill costs about the same as a cached
      one-token step). The same recipe on Linux would very likely be faster, but
      that is a prediction and is **not measured**; the docs say so.
- [ ] **A teacher that knows the protocol.** The GPU benchmark shows that
      distilling toward a raw instruct teacher can reduce task success. An arm
      that SFTs the teacher first would separate "OPD does not help here" from
      "this teacher does not help here".
- [ ] **The JSON-navigation and SQLite recipes on GPU.** Only the calculator
      environment has been run end to end on real models.

## Roadmap (deliberately not implemented in v0.1.0)

Listed in `docs/limitations.md` too. Nothing below exists in the code.

- [ ] Cross-tokenizer distillation. Currently rejected with an explicit error.
- [ ] Padded multi-sequence batching. Today one trajectory per forward pass, so
      `gradient_accumulation_steps` is the batch size.
- [ ] Entropy-aware divergence mixing (arXiv:2603.07079). miniVERL already
      records per-token teacher entropy, which is the input such a method needs.
- [ ] More model families. Only Qwen3 and Qwen2 are tested.
- [ ] More environments. The registry and the example make this the cheapest
      contribution to accept.
- [ ] Multi-GPU. Out of scope on purpose; use verl.
