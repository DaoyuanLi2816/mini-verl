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

The v0.11 baseline result remains the pre-change `hf_reference` measurement.
It is not evidence for `hf_cached` or grouped-rollout performance. External
engine support is a separate release-chain stage.
