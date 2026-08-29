# Competitive landscape

miniVERL's v0.11 development line is focused on current-policy rollout speed,
grouped samples and deterministic reward composition on one consumer GPU. This
page records the upstream state used to make engineering choices; it is not a
leaderboard. A capability observed in source is kept separate from a workload
measured on the local RTX 4080.

Inspected: **2026-08-29**.

| Project | Exact source inspected | Capability observed in source or stable documentation | miniVERL state at the start of v0.11 | Locally measured | Caveat |
| --- | --- | --- | --- | --- | --- |
| [mini-opd](https://github.com/thu-nics/mini-opd) | `b47eaa1728fc3a7ff0b0b627ec2c93d60f07aa16` (`main`; no GitHub release) | SGLang HTTP rollouts, `n_rollouts` (default 4), task rewards, group reward processing, macro-batching, optional one-step-stale background rollout, and checkpoint weight reload | Local HF prompt rollouts; no typed grouped-sample contract and no external generation engine | **No** — source audit only | Its README starts SGLang in a separate terminal. Async measurements would not be compared with miniVERL's strict synchronous path without a separate staleness label. |
| [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) | stable `v0.7.0`, commit `81aa43dda4765b0429cf31dab44441e4e4383911`; current `main` also inspected at `46dfb92fb5ceed5b2d593d652361b10d0acc6ef6` | A documented single-GPU OPD example, vLLM and SGLang generation, refit/weight synchronization, multi-teacher MOPD, and broader Ray/distributed scope | One-GPU local trainer and a fail-closed portable verl bridge; no NeMo RL runtime integration | **No** — install and one-step feasibility remain to be measured | “Single GPU” is not evidence that the published example fits this repository's 16 GiB RTX 4080 envelope. The stable MOPD objective and runtime differ from miniVERL's closed profiles. |
| [SGLang](https://github.com/sgl-project/sglang) | stable `v0.5.18`; current `main` `0a585d5bb108cab8f0922b483d7f55812f05e245` | Generation server and engine APIs considered for a managed rollout backend | Not integrated | **No** | Public support requires a local lifecycle, raw-token, synchronization, cache-invalidation and teardown qualification; an API being present is not that qualification. |
| [vLLM](https://github.com/vllm-project/vllm) | stable `v0.28.0`; current `main` `cacc429f62c3738c9c95093e9bd410e96103221a` | Batched generation and model-serving APIs considered for a managed rollout backend | The verl-shaped config vocabulary may mention vLLM, but miniVERL currently executes local HF generation | **No** | No vLLM process or distributed verl job has been run by the v0.11 line yet. Support depends on measured policy refresh, cache invalidation, memory and teardown behavior. |

## Selection rule

`hf_reference` remains the compatibility oracle. A batched local `hf_cached`
backend is the first implementation target. SGLang and vLLM will be spiked on
the same WSL2 RTX 4080 workload; only one becomes a supported v0.11 backend,
and only if it has a measured advantage over `hf_cached` together with strict
policy synchronization and lifecycle cleanup. Until then, both are
**not measured** rather than recommended.

The comparison workload, metrics and invalidation rules are preregistered in
[`rollout-runtime-v2.yaml`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/preregistration/rollout-runtime-v2.yaml).
