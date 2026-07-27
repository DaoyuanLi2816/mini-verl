# 0005. Resident and swap memory strategies, and the quantized-swap refusal

Status: Accepted, 2026-07-27.

## Context

On-policy distillation needs two models alive across one cycle: the student
generates, the teacher scores the student's own trajectories, and the student
is updated. On a cluster the two live on different devices. On one 16 GB card
they either share the card or take turns.

Which of those is possible depends on the pair, and the wrong answer is an OOM
several minutes into a run. A single-GPU trainer therefore has to make a
placement decision, make it before the expensive part, and be able to explain
it afterwards.

## Decision

`src/miniverl/training/memory.py` defines three strategies.

**`resident`** keeps both models on the accelerator. It is the cheapest in
wall-clock and the only strategy that supports the `exact_hidden` teacher
supervision shape, because that shape requires the teacher's LM head to still
be callable during the update (`LocalTeacherScorer.keep_exact_resident`).

**`swap`** keeps one model on the accelerator at a time. Per cycle: the student
rolls out; the student weights *and optimizer state* move to host memory and
the device memory is released; the teacher scores and writes compressed targets
to the cache; the teacher is released; the student and its optimizer state come
back; the update runs against the cached targets. `move_optimizer_state` exists
because without it the Adam moments -- most of what an optimizer costs -- would
stay pinned on the GPU while the weights left.

**`auto`** resolves to one of the two and records why. On a non-CUDA device it
picks `resident` with the reason "no CUDA device, host memory is not
partitioned". On CUDA it first compares free VRAM against
`memory.auto_swap_vram_headroom_gb` (default 2.0) and picks `swap` if there is
less; otherwise it calls a `teacher_fits` probe that actually attempts the
resident placement and cleans up on OOM. Every branch returns a `MemoryPlan`
carrying a human-readable `reason`, which is printed, written into
`config.resolved.yaml` and recorded in `manifest.json`. `auto` never silently
changes anything that affects the objective.

**Quantized models refuse to swap.** bitsandbytes 4-bit and 8-bit parameters
are pinned to the device they were quantized on and cannot be moved to host
memory and back. `OPDTrainer.from_config` therefore raises `ConfigError` for
`memory.strategy: swap` with any quantization set, and forces `auto` to
`resident` with the recorded reason "a quantized model cannot be moved off the
accelerator, so swap is unavailable". The one-cycle GPU smoke test on this
machine resolved to exactly that: strategy `resident`, that reason, projection
chunk 256, zero OOM retries, peak CUDA allocated 4.251 GiB and peak reserved
4.762 GiB, with a 10,092,544-parameter trainable LoRA on a
385,941,504-parameter NF4-packed student and a 1,720,574,976-parameter teacher.

**OOM retries only halve `loss.chunk_size`**, down to `memory.min_chunk_size`.
The chunk size is the number of selected prediction positions projected through
the LM head at once; the loss and the gradient are identical for any chunk
size, which `tests/unit/test_chunked_equivalence.py` verifies for chunk sizes
1, 5 and 37 across all three divergences. Sequence lengths, batch sizes, models
and objectives are never changed behind the user's back. When halving is
exhausted, `run_with_oom_retry` raises `GpuMemoryError` listing the six knobs
that actually help and the original error. The chunk size that succeeded is
written back into the plan so the rest of the run does not re-discover the
limit every step.

## Consequences

Positive:

- The placement decision, its reason and the number of OOM retries used are all
  in the manifest, so a run's memory behaviour is reconstructable from its
  artifacts.
- `tests/integration/test_resume_and_swap.py::test_swap_and_resident_produce_the_same_update`
  asserts the two strategies produce the same update, so the choice is a
  performance decision and not a correctness one.
- The automatic response to OOM cannot change the mathematics, only the
  throughput.

Negative:

- `swap` costs wall-clock for two extra host-to-device transfers per cycle.
  That cost has not been measured on this machine; the 16 GB recipe uses a
  quantized student and therefore runs `resident`.
- The `auto` probe attempts a real teacher load, so on a machine where the
  teacher does not fit, the fallback path pays a load and an OOM before
  choosing `swap`.
- `swap` plus `exact_full_vocab` is refused above `loss.exact_max_vocab`
  (default 8192), because it would have to persist a `[positions, vocab]`
  tensor. That is a guard rail, not a capability.
- The free-VRAM headroom check is a heuristic: it reads free VRAM at one moment
  and cannot account for another process allocating afterwards.

## Alternatives considered

**Always resident.** Rejected: it makes a fitting teacher a hard requirement
and gives no path for an unquantized pair that does not fit.

**Always swap.** Rejected: it is slower for every pair that would have fit, and
it is impossible for quantized models.

**CPU offload of individual layers (accelerate-style device maps).** Rejected
for v0.1: it interacts with LoRA, gradient checkpointing and the chunked
projection in ways this project has not tested, and it would make peak-memory
numbers harder to attribute.

**Retry OOM by shortening sequences or shrinking the batch.** Rejected. Those
change the objective and the data the model sees. Halving the projection chunk
is the only automatic response that is provably equivalence-preserving.
