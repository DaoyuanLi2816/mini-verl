# Rollout Runtime v2 benchmark method

This is a systems benchmark for one WSL2 RTX 4080. It measures rollout phases,
memory, current-policy synchronization and lifecycle behavior; it does not
measure task quality or establish an algorithm ranking.

The preregistration is
[`benchmarks/preregistration/rollout-runtime-v2.yaml`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/preregistration/rollout-runtime-v2.yaml).
The actor and teacher snapshots, token workload, sample seeds, response bounds,
warmups, repetitions and invalidation rules are fixed there before the first
baseline measurement.

## Comparisons

1. `hf_reference` freezes the pre-v0.11 behavior and acts as the compatibility
   oracle.
2. `hf_cached` must preserve the oracle's greedy tokens, stop decisions and
   policy identity while adding batched incremental KV-cache decoding.
3. One external engine may be qualified after separate SGLang and vLLM spikes.

`samples_per_prompt=4` in the baseline means four independently repeated
requests. It is not evidence that the later grouped trajectory, transaction or
resume contract already exists.

## Timing and rates

Cold start, prefill, decode, policy synchronization, teacher scoring, actor
update, complete cycle and teardown are separate fields. Prompt throughput,
decode throughput and full-cycle throughput are never combined into one number.
Cells that cannot expose a phase retain a structured `not_measured` status
instead of a fabricated zero.

Each measured cell records actual prompt and generated token counts. Rates use
those counts, not configured maximums. CUDA is synchronized at timing
boundaries, peak memory is reset per cell, and failures/OOM downshifts remain in
the artifact.

## Correctness and scope

Greedy output IDs, stop reasons and policy identity are exact comparisons.
Sampled-token log-probabilities use the preregistered dtype-aware tolerances.
The workload also tests batch partitions `[4]`, `[2,2]` and `[1,1,1,1]`.

The result applies to the recorded RTX 4080, WSL2/Linux stack and exact model
revisions. It does not imply gains on other GPUs, a task-quality improvement,
or execution of distributed verl.
