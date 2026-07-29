# Memory on one personal GPU

miniVERL runs a teacher and a student on one card. This document describes the
two placement strategies, how `auto` picks between them, the out-of-memory retry
contract, and the projection trick that keeps vocabulary-sized tensors small.

Source: `src/miniverl/training/memory.py`, the swap and resident sections of
`src/miniverl/training/trainer.py`, `src/miniverl/models/hf.py`,
`src/miniverl/losses/chunked.py` and `src/miniverl/utils/gpu.py`.

---

## Contents

- [`resident`](#resident)
- [`swap`](#swap)
- [`auto`](#auto)
- [Why `swap` is refused for quantized models](#why-swap-is-refused-for-quantized-models)
- [The OOM retry contract](#the-oom-retry-contract)
- [Peak allocated vs peak reserved](#peak-allocated-vs-peak-reserved)
- [Selected-position projection](#selected-position-projection)
- [Measured: one OPD cycle on an RTX 4080](#measured-one-opd-cycle-on-an-rtx-4080)
- [Knobs to turn when you run out of memory](#knobs-to-turn-when-you-run-out-of-memory)
- [Roadmap](#roadmap)

---

## `resident`

Teacher and student both stay on the accelerator for the whole run. Cheapest in
wall-clock, because nothing is ever moved.

It is also the only strategy that supports the `exact_hidden` teacher shape. In
`src/miniverl/teachers/local.py`, `LocalTeacherScorer` is constructed with
`keep_exact_resident=(plan.strategy is MemoryStrategy.RESIDENT)`. When that flag
is set and `loss.mode` is `exact_full_vocab`, the scorer keeps only the teacher's
`[N, H]` hidden states and a reference to its LM head, and rebuilds the
`[chunk, V]` teacher distribution on demand during the update. That requires the
teacher's LM head to still be callable, which a swapped-out teacher is not.

Under `resident` the trainer sets `_teacher_on_device = True` at construction and
never calls `_teacher_to_device` or `_teacher_off_device`.

## `swap`

One model on the accelerator at a time. Slower, but it fits pairs that
`resident` cannot. The sequence per training cycle, from `_run_cycle` in
`src/miniverl/training/trainer.py`:

1. **Roll out.** `_collect` samples trajectories from the student, which is on
   the accelerator. The teacher was loaded onto `"cpu"` at startup
   (`teacher_device = device if plan.strategy is RESIDENT else "cpu"`).
2. **Evict the student.** `_student_off_device()` takes a CPU copy of the
   trainable weights via `student.trainable_state_dict()`, calls
   `move_optimizer_state(optimizer, "cpu")`, calls `student.release()` (which
   moves the model to host memory), and — when
   `memory.empty_cache_between_phases` is true — calls `gpu.empty_cache()` to
   return the caching allocator's blocks to the driver.
3. **Admit the teacher.** `_teacher_to_device()` calls
   `teacher.to_device(plan.device)`.
4. **Score.** `_build_samples` runs the teacher over the student's states and
   writes compressed `top-k + tail` targets to the on-disk teacher cache.
5. **Evict the teacher.** In a `finally` block, `_teacher_off_device()` calls
   `teacher.release()`, moving it back to host memory and emptying the cache.
6. **Readmit the student.** `_student_on_device()` calls
   `student.to_device(plan.device)` and `move_optimizer_state(optimizer,
   plan.device)`, then `load_trainable_state_dict(student_state)` restores the
   weights captured in step 2.
7. **Re-attach the targets.** `_reload_targets_from_cache` reads each
   trajectory's targets back from the cache and builds a fresh
   `BucketedTargetProvider` on `plan.device`.
8. **Update.** `_optimize` runs the optimizer steps against the cached targets.

Steps 5 and 6 are in a `finally`, so a failure during scoring still restores the
student rather than leaving the run with no model on the device.

`move_optimizer_state` exists because without it `swap` would move the model to
host memory while leaving the Adam moments pinned on the GPU — which is most of
what an optimizer costs.

Because a swapped teacher cannot serve `exact_hidden`, `loss.mode:
exact_full_vocab` under `swap` must persist a `[positions, V]` tensor instead.
For a large vocabulary that is refused by a guard rail:

```
loss.mode=exact_full_vocab with memory.strategy=swap must persist a
[positions, 151936] teacher tensor, which exceeds the
loss.exact_max_vocab=8192 guard rail.
  hint: use loss.mode=bucketed_topk_tail for large vocabularies, or set
  memory.strategy=resident so the exact teacher distribution can be rebuilt
  one chunk at a time, or set loss.allow_large_exact=true if you really mean it
```

## `auto`

`resolve_strategy` turns `auto` into a concrete strategy and records why. It
never silently changes anything that affects the mathematical objective — the
only thing it decides is where the weights live.

The rule, in order:

1. **Strategy set explicitly.** Return it unchanged.
   Reason: `memory.strategy was set explicitly to resident` (or `swap`).
2. **Device is not CUDA.** Return `resident`.
   Reason: `auto -> resident: no CUDA device, host memory is not partitioned`.
3. **Free VRAM below the headroom.** If `free_vram_gib() <
   memory.auto_swap_vram_headroom_gb` (default `2.0`), return `swap`.
   Reason: `auto -> swap: only 14.72 GiB free before loading the teacher, below
   the 1000.00 GiB headroom` (numbers are the measured free VRAM and the
   configured headroom).
4. **Try the resident placement.** `teacher_fits` actually attempts to build the
   teacher on the device. If it raises a CUDA OOM, the callback empties the cache
   and returns `False`, and `auto` returns `swap`.
   Reason: `auto -> swap: the teacher did not fit alongside the student (14.72
   GiB were free)`.
5. **Otherwise `resident`.**
   Reason: `auto -> resident: 14.72 GiB free after loading the student`.

There is one earlier short-circuit, applied in `OPDTrainer.from_config` before
`resolve_strategy` runs. If either model is quantized and the strategy is `auto`,
the strategy is forced to `resident` and the reason is overwritten with:

```
auto -> resident: a quantized model cannot be moved off the accelerator, so swap is unavailable
```

### Where the decision is recorded

`MemoryPlan.to_dict()` carries `strategy`, `projection_chunk_size`, `device`,
`reason`, `oom_retries_used` and `chunk_size_history`. It reaches the user in
four places:

| artifact | contents |
| --- | --- |
| `manifest.json`, key `memory` | The full `MemoryPlan.to_dict()`. Written **once at startup**, so `oom_retries_used` is always `0` and `chunk_size_history` always `[]` there. |
| `config.resolved.yaml` | The resolved `memory.strategy` and the startup `loss.chunk_size`, so a re-run of the resolved file reproduces the placement without re-deciding it. |
| `events.jsonl`, `run_start` event | `strategy`, plus `device`, `mode`, `loss_mode` and `divergence`. |
| `metrics.jsonl`, per optimizer step | `projection_chunk_size` — the **live** value, so this is where a post-retry chunk size shows up. |

The `manifest.json` block from a completed CPU toy run (`miniverl demo --fast`),
showing the shape of the record:

```json
"memory": {
  "chunk_size_history": [],
  "device": "cpu",
  "oom_retries_used": 0,
  "projection_chunk_size": 128,
  "reason": "memory.strategy was set explicitly to resident",
  "strategy": "resident"
}
```

`scripts/gpu_smoke.py` prints `memory_strategy`, `memory_reason`,
`projection_chunk_size` and `oom_retries_used` as JSON, which is the quickest way
to see the decision without opening a run directory.

## Why `swap` is refused for quantized models

bitsandbytes 4-bit and 8-bit parameters are pinned to the device they were
quantized on. `HFBackend.load` passes `device_map={"": device}` alongside the
`BitsAndBytesConfig`, and `HFBackend.to_device` refuses any subsequent move:

```
a bitsandbytes-quantized model cannot be moved between devices after loading
  hint: use memory.strategy: resident for quantized models, or load the model
  unquantized
```

`HFBackend.release` is correspondingly conditional: it only moves the model to
`"cpu"` when `capabilities.quantization == "none"`. For a quantized model it
empties the CUDA cache and leaves the weights where they are.

Rather than let a run discover this at cycle 1, `OPDTrainer.from_config` checks
the combination up front. If either `models.student.quantization` or
`models.teacher.quantization` is not `none` and `memory.strategy` is `swap`, it
raises `ConfigError` before loading anything:

```
memory.strategy=swap cannot be used with a quantized model: bitsandbytes
4-bit/8-bit parameters are pinned to the device they were quantized on and
cannot be moved to host memory and back.
  hint: use memory.strategy: resident (a 0.6B QLoRA student plus a bf16 1.7B
  teacher fits in 16 GB), or set both quantization fields to 'none' if you
  really need swap
```

The same combination under `auto` is not an error — it resolves to `resident`
with the reason quoted in the previous section.

## The OOM retry contract

`run_with_oom_retry` wraps each optimizer step. On a CUDA OOM it **only ever
halves `loss.chunk_size`**. Sequence lengths, batch sizes, models, divergences
and objectives are never changed behind your back.

### Why halving the chunk is mathematically neutral

`chunked_selected_position_loss` divides every chunk's weighted sum by the
**global** weight sum, so the sum of the chunk losses equals the unchunked loss
regardless of how the positions are grouped. For the gradient, the selected
hidden states are detached into a leaf `work` tensor; each chunk backpropagates
into `work.grad`, freeing its `[chunk, V]` intermediates immediately; and a
single `hidden_states.backward(gradient=work.grad)` at the end pushes the
accumulated gradient through the backbone exactly once. `work.grad` therefore
equals the unchunked gradient.

`tests/unit/test_chunked_equivalence.py` asserts both properties. Its
`test_chunked_gradients_match_unchunked` case is parameterized over chunk sizes
1, 5 and 37 across `forward_kl`, `reverse_kl` and `jsd`.

The chunk size is therefore purely a memory/throughput knob, which is what makes
it the one thing the retry logic is allowed to touch.

### The loop

`attempts = memory.oom_retries + 1`. On each CUDA OOM the handler runs the
`cleanup` callback (the trainer passes `optimizer.zero_grad(set_to_none=True)`),
calls `empty_cache()`, and computes `next_chunk = max(chunk // 2,
memory.min_chunk_size)`. It stops if `next_chunk == chunk` (the floor was
reached) or if this was the last attempt.

Every successful retry emits an `oom_chunk_retry` event to `events.jsonl` with
`old_chunk`, `new_chunk` and the note `objective unchanged; only the projection
chunk size shrank`.

Two measured traces with `oom_retries: 3`, `min_chunk_size: 16`, starting chunk
256:

| scenario | chunks attempted | outcome |
| --- | --- | --- |
| succeeds at 64 | 256, 128, 64 | Returns. `plan.chunk_size` becomes `64`, `oom_retries_used` is `2`, `chunk_size_history` is `[256]`. |
| always OOMs | 256, 128, 64, 32 | Raises `GpuMemoryError`. `plan.chunk_size` stays `256` and `chunk_size_history` stays `[]`, because the plan is only written back on success. |

The chunk size that finally succeeded is written back into the plan, so the rest
of the run keeps using it instead of re-discovering the limit on every step.

Note that with 3 retries from 256 the loop never reaches `min_chunk_size: 16` —
it exhausts its attempts at 32 first. The floor is a separate stop condition:
starting from chunk 32 with `min_chunk_size: 16`, the loop attempts 32 then 16
and then stops, regardless of how many retries remain, because halving 16 clamps
back to 16.

A `RuntimeError` or `MemoryError` that is not an OOM is re-raised immediately and
never retried. `is_oom_error` recognises `torch.cuda.OutOfMemoryError` and any
exception whose message contains `out of memory`.

### When retries run out

```
CUDA ran out of memory and the 3 equivalence-preserving retries were exhausted
(projection chunk size reached 32).
  hint: reduce rollout.max_total_tokens, reduce
  train.gradient_accumulation_steps, lower loss.top_k, switch
  models.student.quantization to nf4, enable
  models.student.gradient_checkpointing, or set memory.strategy: swap.
  Original error: <the original CUDA message>
```

The run fails. It does not silently shrink your batch, truncate your sequences
or switch your objective in order to keep going.

## Peak allocated vs peak reserved

`gpu.snapshot()` reports both, because they answer different questions:

- **allocated** (`torch.cuda.max_memory_allocated`) is what the live tensors
  need. It is the number to compare against a model's parameter and activation
  footprint when you are reasoning about whether a change helped.
- **reserved** (`torch.cuda.max_memory_reserved`) is what the caching allocator
  took from the driver. It is the number that actually decides whether the next
  allocation OOMs, because the allocator does not hand blocks back until
  `empty_cache()` is called, and a reserved block that is the wrong size for the
  next request is unusable.

Reserved is always at least allocated, and the gap is fragmentation plus
allocator slack. Reporting only allocated would understate how close a run is to
the edge; reporting only reserved would make an unfragmented run look worse than
it is.

`MemorySnapshot.to_dict()` writes `allocated_bytes`, `reserved_bytes`,
`peak_allocated_bytes`, `peak_reserved_bytes`, the two GiB conveniences rounded
to 4 decimals, `total_bytes` and `cuda_available`. It is embedded under
`memory` in every per-step and per-cycle record in `metrics.jsonl`.

When `memory.reset_peak_stats_each_cycle` is true (the default),
`gpu.reset_peak_stats()` runs before each optimizer step, so each step's peaks
are measured on their own rather than inheriting the high-water mark of model
loading.

## Selected-position projection

The LM head is **never** run over a whole sequence. Both scoring and training
call the decoder backbone directly for `[1, T, H]` hidden states, gather the
selected prediction positions into `[N, H]`, and project only those. Even
generation projects a single position per step.

`HFBackend.hidden_states_at` is the gather; `chunked_selected_position_loss`
does the projection in slices of `chunk_size`, so the largest vocabulary-sized
tensor alive at any moment is `[chunk_size, V]`.

### The tensors this avoids

With the pinned Qwen3 pair, `vocab_size` is **151936** (both models; note that
`len(tokenizer)` is 151669 — the configured vocabulary is padded). The recipe's
`rollout.max_total_tokens` is 704.

A naive full-sequence logit tensor at `T = 704`:

```
704 x 151936 = 106,962,944 elements
  fp32: 427,851,776 bytes = 408.0 MiB
  bf16: 213,925,888 bytes = 204.0 MiB
```

At `T = 768`, the figure quoted in the `models/hf.py` docstring:

```
768 x 151936 = 116,686,848 elements
  fp32: 466,747,392 bytes = 445.1 MiB
```

That is per forward pass, per sequence, and the backward pass would need its own
gradient tensor of the same shape. miniVERL never allocates either.

What it builds instead, at `chunk_size: 256`:

```
256 x 151936 = 38,895,616 elements, fp32: 155,582,464 bytes = 148.4 MiB
```

And the halving ladder the OOM retry walks:

| `chunk_size` | `[chunk, 151936]` fp32 |
| ---: | ---: |
| 256 | 148.4 MiB |
| 128 | 74.2 MiB |
| 64 | 37.1 MiB |
| 32 | 18.5 MiB |
| 16 | 9.3 MiB |

The gathered hidden states are negligible by comparison. At `N = 256`:

```
student  [256, 1024] bf16 = 524,288 bytes   = 0.500 MiB
teacher  [256, 2048] bf16 = 1,048,576 bytes = 1.000 MiB
```

This asymmetry is the point. Under `resident` with `exact_full_vocab`, the
teacher scorer keeps only the 1.0 MiB `[N, H]` tensor and reconstructs the
148 MiB `[chunk, V]` distribution on demand, one chunk at a time.

Two costs are worth naming honestly. Selecting fewer positions with the `hybrid`
or `uniform_ratio` selectors reduces `N`, and therefore the cache size and the
number of projections — but it does **not** proportionally reduce teacher FLOPs,
because the teacher still runs a full backbone forward pass to produce the
hidden states. And `chunk_size` bounds the vocabulary tensor only; it does
nothing about the backbone activations, which is why the fallback list in the
`GpuMemoryError` hint reaches for sequence length and gradient checkpointing
instead.

## Measured: one OPD cycle on an RTX 4080

Produced by `scripts/gpu_smoke.py` with the historical raw-teacher recipe
`recipes/qwen_consumer_gpu_calc_raw_teacher.yaml`.

```bash
python scripts/gpu_smoke.py \
  --recipe recipes/qwen_consumer_gpu_calc_raw_teacher.yaml \
  --output runs/gpu-smoke
```

The script overrides the recipe budgets before running: `cycles=1`,
`rollouts_per_cycle=2`, `gradient_accumulation_steps=2`, `sft_warmup_cycles=1`,
`eval_every_cycles=0`, `save_every_cycles=0`,
`rollout.max_new_tokens_per_turn=48`, `train_tasks=16`, `eval_tasks=2`,
`test_tasks=2`. Everything else comes from the recipe: Qwen3-0.6B student with
NF4 double quantization plus LoRA (`r=16`, `alpha=32`, seven target modules) and
gradient checkpointing, bf16 Qwen3-1.7B teacher, `sdpa` attention,
`loss.mode: bucketed_topk_tail`, `reverse_kl`, `top_k: 64`, `chunk_size: 256`,
`selector: hybrid` at `ratio: 0.6`, `optimizer: adamw8bit`,
`memory.strategy: auto`.

Machine: Windows 11 Pro 10.0.22631, RTX 4080 (16376 MiB), driver 596.49,
CPython 3.12.13, torch 2.13.0+cu130, transformers 5.14.1, peft 0.19.1,
bitsandbytes 0.50.0.

| measurement | value |
| --- | --- |
| peak CUDA allocated | **4.251 GiB** |
| peak CUDA reserved | **4.762 GiB** |
| memory strategy | `resident` |
| reason | `auto -> resident: a quantized model cannot be moved off the accelerator, so swap is unavailable` |
| projection chunk size | 256 |
| OOM retries used | 0 |
| student trainable parameters | 10,092,544 (LoRA) |
| student total parameters | 385,941,504 (NF4-packed) |
| teacher total parameters | 1,720,574,976 |

Two things to read from this. The reserved-to-allocated ratio is 1.12, so
fragmentation is mild. And 4.762 GiB reserved on a 15.99 GiB card leaves roughly
11.2 GiB unused, which is why `chunk_size` never had to shrink — the headroom
here is large enough that the knobs below were not needed for this
configuration.

Do not extrapolate this figure to a different pair, sequence length, `top_k` or
selector. It is one measurement of one configuration.

Related throughput measurements on the same machine, for sizing rollout budgets.
Single-sequence decode of 64 new tokens from a 36-token prefix:

| configuration | tok/s | peak allocated |
| --- | ---: | ---: |
| nf4, deterministic | 11.19 | 0.862 GiB |
| nf4, non-deterministic | 11.29 | not recorded |
| bf16 LoRA, deterministic | 12.84 | 1.170 GiB |
| bf16 LoRA, non-deterministic | 14.12 | not recorded |

A 14-token prefill costs 37.0 ms against 30.9 ms for a cached 1-token step, so
decoding here is kernel-launch bound rather than compute bound. Reproduce the
probe with `python scripts/gpu_probe_throughput.py`; the same table and its
caveats are in
[limitations.md](limitations.md#throughput-numbers-are-platform-specific-and-mostly-measure-kernel-launches).

## Knobs to turn when you run out of memory

In the order to try them. The first four preserve the objective exactly; the
rest change what is being optimized, so measure before and after.

| # | knob | effect | cost |
| --- | --- | --- | --- |
| 1 | `loss.chunk_size` (halve it) | Shrinks the `[chunk, V]` projection tensor linearly. From 256 to 128 saves 74 MiB per live chunk. | Fewer, larger kernel launches become more, smaller ones. Loss and gradient are unchanged (asserted by `tests/unit/test_chunked_equivalence.py`). |
| 2 | `models.student.gradient_checkpointing: true` | Trades backbone activation memory for a recomputation pass. | Slower training step. Already on in the 16 GB recipe. |
| 3 | `models.student.quantization: nf4` | NF4 weights with double quantization. | Slower decode: 11.19 tok/s against 12.84 tok/s for bf16 LoRA in the deterministic probe on this machine. Forces `resident` — see above. |
| 4 | `train.gradient_accumulation_steps` (reduce) | Fewer trajectories held live per optimizer step. | Under `mode: opd`, default `opd_freshness: strict` rejects more than one optimizer step per rollout batch. Explicit `opd_freshness: replay` permits it but labels the objective as replay, not genuine OPD. Keep `gradient_accumulation_steps >= rollouts_per_cycle` for strict on-policy updates. |
| 5 | `memory.strategy: swap` | One model on the device at a time. | Slower, and only available when neither model is quantized. Rules out `exact_hidden`. |
| 6 | `loss.top_k` (reduce) | Smaller teacher targets and a smaller on-disk cache. | Changes the objective: fewer teacher probabilities are represented exactly, more mass falls into the tail bucket. |
| 7 | `rollout.max_total_tokens` / `rollout.max_new_tokens_per_turn` (reduce) | Shorter sequences mean smaller backbone activations. | Changes the task: episodes may terminate with `max_tokens` instead of `final_answer`. |
| 8 | `selection.selector: hybrid` with a lower `selection.ratio` | Fewer supervised positions, so fewer projections and a smaller cache. | Changes the objective. Does **not** reduce teacher forward FLOPs. |

Two settings that help indirectly:

- `memory.empty_cache_between_phases` (default `true`) returns cached blocks to
  the driver when the student is evicted under `swap`.
- `train.optimizer: adamw8bit` keeps the Adam moments small, which matters most
  under `swap` where they are copied across the bus twice per cycle.

If you have exhausted this list, `memory.oom_retries` and
`memory.min_chunk_size` control how hard the automatic path tries before giving
up. Raising `oom_retries` only lets the chunk halve further; it will not rescue a
configuration whose backbone activations alone do not fit.

## Roadmap

Not implemented. Listed so the absence is explicit rather than assumed.

- **Multi-GPU and sharded training.** miniVERL places models on a single device
  chosen by `resolve_device`. There is no FSDP, no tensor parallelism and no
  Ray-based placement.
- **Cross-tokenizer distillation.** `build_alignment_map` raises
  `TokenizerMismatchError` when the fingerprints differ. Teacher and student must
  share a tokenizer. See `docs/limitations.md`.
- **Automatic sequence-length or batch-size reduction on OOM.** Deliberately
  excluded: only the projection chunk shrinks, because only that is provably
  equivalence-preserving.
- **Partial-layer offload.** `swap` moves whole models. There is no per-layer
  streaming of teacher weights.
- **Entropy-aware objective mixing.** miniVERL records per-token teacher entropy
  (`teacher_entropy_mean` in `metrics.jsonl`), but does not use it to blend
  forward and reverse KL as in arXiv:2603.07079. The measurement exists; the
  mechanism does not.
