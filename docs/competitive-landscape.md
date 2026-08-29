# Competitive landscape

miniVERL's v0.11 development line is focused on current-policy rollout speed,
grouped samples and deterministic reward composition on one consumer GPU. This
page records the upstream state used to make engineering choices; it is not a
leaderboard. A capability observed in source is kept separate from a workload
measured on the local RTX 4080.

Inspected: **2026-08-29**.

| Project | Exact source inspected | Capability observed in source or stable documentation | miniVERL state at the start of v0.11 | Locally measured | Caveat |
| --- | --- | --- | --- | --- | --- |
| [mini-opd](https://github.com/thu-nics/mini-opd) | `b47eaa1728fc3a7ff0b0b627ec2c93d60f07aa16` (`main`; no GitHub release) | SGLang HTTP rollouts, `n_rollouts` (default 4), task rewards, group reward processing, macro-batching, optional one-step-stale background rollout, and checkpoint weight reload | Typed grouped samples, deterministic rewards, local `hf_cached`, and managed vLLM direct-GKD generation | **No** — source audit only | Its documented separate-server and optional stale-async paths do not match miniVERL's managed strict-sync workload. No performance comparison is claimed. |
| [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) | stable `v0.7.0`, commit `81aa43dda4765b0429cf31dab44441e4e4383911`; current `main` `46dfb92fb5ceed5b2d593d652361b10d0acc6ef6` | A documented single-GPU OPD example, vLLM and SGLang generation, refit/weight synchronization, multi-teacher MOPD, and broader Ray/distributed scope | One-process consumer-GPU runtime with a portable, fail-closed verl bridge; no NeMo RL integration | **No** — setup and one-step run were not attempted after the source audit | Runtime, objective and distributed dependencies differ. There is no miniVERL-versus-NeMo performance claim. |
| [SGLang](https://github.com/sgl-project/sglang) | stable `v0.5.18` at `71de97b264b04dcd514cf904003028aefe9775c8`; current `main` `cdbfe90b4a6c728e03e6520862d792501b3a97bb` | Raw-token generation, batching and adapter lifecycle APIs suitable for a spike | Not selected | **Attempted; no throughput measurement** | The WSL2 spike stopped at FlashInfer/system-CUDA compatibility and a Triton fallback expecting `/usr/local/cuda/bin/nvcc`. Missing throughput is not represented as zero. |
| [vLLM](https://github.com/vllm-project/vllm) | stable `v0.28.0` at `2cf0a6915ce544dc493a0990f2ea38d81601128a`; current `main` `fd5d3aea9470bb92376eb2d9f5d64cd8f23de31b` | Batched raw-token generation, CUDA Graph execution and dynamic LoRA load/unload used by the managed backend | Selected external engine for the development-line direct-GKD path | **Yes** — full 24-cell WSL2 RTX 4080 workload | 626.6–836.4 tokens/s and 3.08–5.97× over `hf_cached` in required cells; 11,931 MiB peak total GPU memory. PG-k1 remains on `hf_cached` because log-probability conformance failed. |

## Selection rule

`hf_reference` remains the compatibility oracle and `hf_cached` remains the
service-free local backend. vLLM 0.28.0 passed the external-engine value,
strict-sync, memory and teardown gates for direct GKD, so it is the selected
engine. Eight policy refreshes used unique identities and showed no monotonic
memory growth. Its sampled-token log-probabilities exceeded the NF4 tolerance,
so PG-k1 continues to use `hf_cached`.

The optimized `hf_cached` backend reached the preregistered 2× speedup over
`hf_reference` in every 256/512-token cell. Its first failed candidate remains
published as negative evidence. mini-opd and NeMo RL were not run locally, and
no relative performance conclusion is made about either.

The comparison workload, metrics and invalidation rules are preregistered in
[`rollout-runtime-v2.yaml`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/preregistration/rollout-runtime-v2.yaml).
