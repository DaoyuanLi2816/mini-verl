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

`compile_backend: true` uses a generation-only Inductor decoder. With an NF4
training actor, the runtime owns a BF16 rollout mirror and copies the exact live
LoRA tensors into it at every policy version. The first run compiles and warms
the decoder; set it to `false` for the eager cached path or on hosts without a
working CUDA compiler toolchain.

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

miniVERL starts the pinned vLLM server with CUDA Graph execution on an ephemeral localhost port, exports
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

| Response | `hf_cached` / `hf_reference` | vLLM / `hf_cached` | `hf_cached` tokens/s | vLLM tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 2.39–4.30× | 2.99–5.71× | 124.2–242.5 | 626.6–806.0 |
| 256 | 2.38–4.30× | 3.19–5.97× | 125.9–240.8 | 685.6–836.4 |
| 512 | 2.54–4.23× | 3.08–5.88× | 124.3–248.7 | 693.4–821.2 |

Both preregistered speed gates passed in every required 256/512-token cell.
`hf_cached` peaked at 6,504 MiB and vLLM at 11,931 MiB of total GPU memory.
Both backends confirmed eight unique policy refreshes without monotonic growth
and closed cleanly. The local NF4/mirror probe passed; vLLM's sampled-token
log-probabilities remained outside the PG-k1 tolerance, which keeps its scope
to direct GKD. The [benchmark report](benchmarks/rollout-runtime-v2.md) and
[machine-readable result](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/rollout-runtime-v2-v0.11.0-candidate-rtx4080.json)
contain every cell and raw artifact hash. The earlier failed candidate remains
available beside it as preserved negative evidence.
