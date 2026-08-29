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

The v0.11 baseline result remains the pre-change `hf_reference` measurement.
It is not evidence for `hf_cached` performance. Grouped `n>1` publication and
external-engine support are separate release-chain stages.
