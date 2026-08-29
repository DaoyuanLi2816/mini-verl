# Rollout Runtime v2

Rollout Runtime v2 gives prompt-based OPD a typed boundary between the live
actor and generation. Existing recipes continue to use `hf_reference`. A new
recipe can opt into the local cached path:

```yaml
rollout:
  backend: hf_cached
  samples_per_prompt: 1
  prompt_batch_size: 4
  max_padded_tokens: 4096
  synchronization: strict
  compile_backend: true  # measured WSL2/CUDA fast path; first run compiles
  record_logprobs: true
```

`hf_cached` performs one padded prefill for each physical batch, then advances
the model with one token per active row and the returned KV cache. Each logical
sample owns its CPU generator, so OOM bisection and physical repartitioning do
not enter the seed derivation. Greedy and stochastic requests retain EOS, text
stop, maximum-token and sampled-token-log-probability provenance.

`compile_backend: true` uses a generation-only Inductor decoder with CUDA
graphs disabled. It has a substantial first-run compilation cost, so it is
explicit rather than a legacy default. Set it to `false` for the eager cached
path or on hosts without a working CUDA compiler toolchain.

## Policy binding

Before generation, the runtime synchronizes a `PolicySnapshot` that binds the
parameter version, model revision, tokenizer structure, adapter manifest and
live trainable-tensor digests, precision, quantization, backend version,
profile identity and execution plan. A request for any other identity fails
before model execution. Lifecycle transitions are explicit: `new` →
`synchronized` → `quiesced` or `closed`.

## Grouped prompt rollouts

The grouped profiles accept `actor_rollout_ref.rollout.n > 1` for Parquet
prompt sources. Each prompt receives a stable group identity and `n` independent
current-policy trajectories. The direct-GKD and sampled-k1 objectives train
those trajectories independently; there is no group baseline, GRPO estimator
or reward normalization hidden behind `n`.

Trajectory schema v3 binds the prompt digest, group and sample indices,
generation seed, backend and policy identity into every record. A complete
group is journaled and appended as one transaction. Checkpoints retain the
prompt cursor, group cursor, trajectory count, policy version and backend sync
identity, so an interrupted group is regenerated without skipping prompts;
replaying an already committed identical group is a no-op rather than a
duplicate append. Cache and export identities carry the same `n`, schema and
seed-derivation versions.

The runtime reports unique prompts, groups, trajectories, generated tokens,
physical generation batches and per-group reward variance when rewards exist.
Logical `n` stays separate from physical batch partitioning.

Use `verl-opd-v0.8-single-gpu-grouped-v1` or
`verl-opd-v0.8-single-gpu-pg-k1-grouped-v1`. The original measured profiles
remain fixed at `n=1`; the grouped profiles are conformance-only until a later
evidence stage measures them. Grouped rollouts currently require a Parquet
prompt source. Environment-backed recipes remain `n=1`.

## Managed vLLM for direct GKD

Install the optional engine in the same Linux or WSL2 environment as the
training stack:

```bash
pip install "miniverl[train,rollout-vllm]"
```

Bind it into the immutable execution plan for the grouped direct-GKD profile:

```bash
miniverl plan \
  --profile verl-opd-v0.8-single-gpu-grouped-v1 \
  --config verl-opd.yaml \
  --accept-local-reinterpretations \
  --rollout-backend vllm \
  --out plan.json
miniverl run --plan plan.json
```

miniVERL starts the pinned vLLM server on an ephemeral localhost port, exports
the live PEFT adapter under a policy-version-and-digest name, generates raw
token IDs, and terminates the complete process group before teacher scoring or
the actor update. Prefix caching is disabled, and a failed policy refresh
closes the server before another request can run.

The development-line evidence supports this backend for direct GKD. Use
`hf_cached` for PG-k1: vLLM's BF16 rollout and the NF4 training actor agreed on
all 32 greedy probe tokens, but the maximum sampled-token log-probability
difference was 0.0194 against the preregistered 0.01 limit.

## Measured RTX 4080 envelope

The exact `0.11.0.dev0` wheel ran the frozen 24-cell workload on WSL2 with
Qwen3-0.6B NF4/BF16, Transformers 5.14.1 and vLLM 0.28.0. Each cell has one
warmup and three measured repetitions.

| Response | Sampling | `hf_cached` / `hf_reference` | vLLM / `hf_cached` | vLLM tokens/s |
| ---: | --- | ---: | ---: | ---: |
| 64 | greedy | 0.65–0.76× | 1.73–1.80× | 112.7–114.9 |
| 64 | seeded stochastic | 1.83–1.87× | 1.94–2.06× | 109.3–114.2 |
| 256 | greedy | 0.71–0.85× | 1.71–1.77× | 111.8–117.4 |
| 256 | seeded stochastic | 0.84–1.85× | 2.04–4.55× | 113.1–114.5 |
| 512 | greedy | 0.67–1.05× | 1.65–1.87× | 114.8–117.7 |
| 512 | seeded stochastic | 1.84–1.90× | 1.90–2.05× | 108.5–114.4 |

vLLM peaked at 11,820 MiB of total GPU memory, confirmed eight unique policy
refreshes without monotonic growth, and closed its localhost port. Initial
managed synchronization, including server startup and first adapter load, took
17.57 seconds. The isolated startup field was overwritten by later no-op start
checks in this measurement, so no standalone startup number is reported.

The external-engine value gate passed in every 256/512-token cell. The separate
`hf_cached >= 2× hf_reference` gate did not: the best seeded-stochastic cells
reached about 1.9× and the greedy path was usually slower than the legacy
oracle. That failed gate blocks v0.11.0 publication while `hf_cached` is
profiled and optimized. The [benchmark report](benchmarks/rollout-runtime-v2.md)
and [machine-readable result](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/rollout-runtime-v2-rtx4080.json)
contain every cell and raw artifact hash.
