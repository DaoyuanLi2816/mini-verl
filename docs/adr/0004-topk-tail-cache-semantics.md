# 0004. Top-k plus tail teacher targets: a coarse-graining, and a lower bound

Status: Accepted, 2026-07-27.

## Context

The Qwen3 vocabulary is large (config `vocab_size` 151936). An exact
full-vocabulary divergence at every selected position materializes a
`[positions, 151936]` float32 tensor for both models, which on a 16 GB card
competes directly with the weights it is supposed to be distilling. A second
constraint: under `memory.strategy: swap` the teacher is evicted from VRAM
before the update runs (ADR 0005), so whatever the update needs must already be
on disk in a form small enough to be worth writing.

This is not a novel problem and the fix is not a novel technique. TRL's
`ServerDistillationTrainer` already exposes `loss_top_k` (default 1) with an
optional tail bucket. The decision recorded here is about naming and honesty,
not invention.

## Decision

The default loss mode is `bucketed_topk_tail`
(`src/miniverl/losses/bucketed.py`). The teacher's distribution over `V`
entries is coarse-grained into `K + 1` buckets: one per teacher top-k token,
plus one aggregate bucket holding the remaining mass. The student is
coarse-grained onto the *same* partition -- its probabilities on the teacher's
top-k ids, plus `1 - sum_k q` in the tail -- and the divergence is computed
between the two `K + 1` category distributions.

Three properties are stated in the module docstring, in the function names, and
in the tests:

1. **This is not full-vocabulary KL.** Every function is named `bucketed_*` so
   no call site can present it as exact.
2. **It is a lower bound.** By the data-processing inequality, coarse-graining
   cannot increase a divergence.
   `tests/unit/test_losses_bucketed.py::test_bucketed_lower_bounds_exact`
   checks this for `k` in {1, 2, 8, 32} across all three divergences, and
   `test_bucketed_is_monotone_non_decreasing_in_k` checks the ordering between.
3. **Equality holds at `k == V`.** `teacher_topk_targets` sets the tail to
   exactly `-inf` there rather than relying on floating-point cancellation, and
   `test_full_k_converges_to_the_exact_loss` asserts agreement with the exact
   loss to atol 1e-5.

Both tails are floored at `log(tail_epsilon)` (`loss.tail_epsilon`, default
1e-9) and both bucket vectors are renormalized before use. Without the floor, a
teacher whose top-k captures all the mass against a student that still leaks
outside it gives `+inf` reverse KL; with it, the penalty is bounded by
`log(1 / tail_epsilon)` nats
(`test_reverse_kl_tail_penalty_is_bounded_by_log_one_over_epsilon`).

The cache (`src/miniverl/cache/store.py`) stores, per selected position, `K`
int32 indices, `K` log-probs, one tail scalar, the target token id and the
weight, with `cache.dtype` choosing float16 or float32 for the log-probs.
`CacheCompressionStats.compute` reports the ratio against a dense
`[positions, vocab]` fp16 dump and its docstring says so explicitly, noting
that this is not a `[batch, seq, vocab]` dump.

`bucketed_teacher_entropy` is computed over the same buckets and labelled as a
lower bound on the true entropy, because merging the tail discards its spread.

## Consequences

Positive:

- Teacher targets become small enough to write to disk and to survive the
  teacher's eviction from VRAM.
- The objective is auditable: the exact loss exists in the same repository
  (`src/miniverl/losses/exact.py`) and the bucketed one is tested against it.
- The cache index records `top_k`, `temperature`, `loss_mode`, the teacher model
  id and revision, the tokenizer fingerprint and the vocabulary size, and
  `TeacherCache.write` refuses an entry whose `top_k` or `temperature` differs.

Negative:

- **The trained objective is not the exact KL.** Every number derived from the
  buckets, including reported teacher entropy, is a lower bound and must be
  labelled that way in reports.
- `top_k` is a hyperparameter that changes the objective, not only the cost. A
  run cached at `top_k: 64` is not comparable to one at `top_k: 8`.
- **It does not reduce student compute.** The student still needs a
  full-vocabulary `log_softmax` over the selected positions, because the tail
  bucket is only meaningful if the student is normalized over the whole
  vocabulary. The saving is teacher-side storage and transfer. Position
  selection likewise reduces cached positions but not teacher FLOPs: the
  teacher still runs a full forward pass to produce the hidden states.

## Alternatives considered

**Always use exact full-vocabulary KL.** Kept as a mode
(`loss.mode: exact_full_vocab`) rather than as the default, and guarded:
`LocalTeacherScorer._check_exact_is_affordable` refuses it under `swap` above
`loss.exact_max_vocab` (default 8192) unless `loss.allow_large_exact` is set,
with an error naming the three ways out. `recipes/toy_exact_full_vocab.yaml`
runs it end to end, affordable because the toy tokenizer has roughly 190
entries; that is what lets the CPU suite check the exact objective. In exact
mode the config rejects an explicitly-set `top_k` other than 1, so a recipe can
never imply a truncation that does not happen.

**Sampled-vocabulary estimators.** Rejected: they trade a stated bound for
variance plus a bias analysis this project has not done.

**Persist full teacher logits to disk.** Rejected: a `[positions, 151936]` fp16
dump per trajectory is large and confers no benefit over the buckets it would
be reduced to anyway.

## Roadmap (not implemented)

Entropy-aware mixing of forward and reverse KL, as proposed in arXiv:2603.07079
("Entropy-Aware On-Policy Distillation of Language Models"), is not
implemented. miniVERL records per-token bucketed teacher entropy, which is the
input such a scheme would need, but nothing consumes it in the loss.
