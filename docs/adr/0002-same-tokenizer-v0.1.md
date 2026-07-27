# 0002. Same tokenizer for student and teacher in v0.1

Status: Accepted, 2026-07-27.

## Context

A token-level distillation loss compares two distributions over the same
vocabulary at the same sequence position. If the student and the teacher
tokenize differently, neither the positions nor the vocabulary indices line up,
and a cached teacher top-k becomes a list of indices into the wrong vocabulary.

Cross-tokenizer distillation is a real research area with real implementations
(KDFlow supports it), and verl's own distillation trainer requires the teacher
to share the student tokenizer. The question for v0.1 is not whether
cross-tokenizer distillation is possible, but whether an unchecked assumption
about tokenizer identity is acceptable. It is not: a mismatch produces targets
that are numerically well-formed and semantically meaningless, which is the
worst class of bug to ship.

## Decision

v0.1 requires that the student and the teacher tokenize identically, and the
requirement is enforced in two independent places.

**At load time**, `build_tokenizer` in `src/miniverl/models/factory.py` loads
one tokenizer for the pair. When the teacher declares its own `tokenizer_id`
(or `model_id`) different from the student's, the teacher tokenizer is loaded
as well and its `fingerprint` compared; a mismatch raises
`TokenizerMismatchError` naming both ids. The teacher's declaration is never
trusted without loading it.

The fingerprint is behavioural, not metadata. `tokenizer_fingerprint` in
`src/miniverl/models/tokenizers.py` hashes a JSON description containing the
adapter kind, the tokenizer class name, `len(tokenizer)`, the EOS and PAD ids,
the sorted additional special tokens, and the token ids produced for a fixed
`PROBE_TEXT` that exercises the ChatML frame, a `<tool_call>` JSON payload, a
`<tool_result>` block and a `<final>` block. Two tokenizers with different
files but identical behaviour on that probe agree; two with the same files but
a different special-token configuration do not.

**At alignment time**, `build_alignment_map` in
`src/miniverl/trajectory/alignment.py` re-checks
`student.tokenizer_fingerprint` against `teacher.tokenizer_fingerprint`, and
then does something stronger: for every selected position it requires the
target token id to be identical on both sides, raising `AlignmentError` on the
first disagreement. That check does not depend on the fingerprint being
correct, so it catches a fingerprint collision or a hand-edited trajectory file
as well.

The pinned pair for the 16 GB recipe is `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca` and `Qwen/Qwen3-1.7B` at revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Both are Apache-2.0 and their
`tokenizer.json` files are byte-identical (sha256
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`).
`tests/gpu/test_gpu_qlora.py::test_the_pinned_pair_shares_one_tokenizer`
asserts this against the downloaded artifacts.

## Consequences

Positive:

- Cached teacher targets index a vocabulary the student shares, so
  `topk_indices` can be stored as bare integers (see ADR 0004).
- Alignment reduces to a per-segment offset, and the offset is verified by
  token-id equality rather than assumed constant.
- A mismatched pair fails at load time with a hint naming a working pair,
  instead of training to a plausible-looking loss curve on nonsense targets.

Negative:

- The teacher must come from the same model family. A stronger teacher with a
  different tokenizer cannot be used at all, which rules out most
  cross-family distillation setups.
- The fingerprint is deliberately strict. Two Qwen checkpoints that differ only
  in `additional_special_tokens` are treated as incompatible even where the
  difference would not matter in practice.
- Note that Qwen3 config reports `vocab_size` 151936 while `len(tokenizer)` is
  151669 (the embedding matrix is padded). The fingerprint uses
  `len(tokenizer)`; the cache index records the backend's `vocab_size`. These
  are different numbers on purpose and must not be conflated.

## Alternatives considered

**Trust the configuration.** Rejected. The failure is silent and the resulting
training run looks healthy.

**Compare tokenizer files by hash.** Rejected as both too strict and too loose:
two repositories can ship behaviourally identical tokenizers in different file
layouts, and identical files with different `AutoTokenizer` keyword arguments
can behave differently. The probe-text fingerprint tests the property that
matters.

**Implement cross-tokenizer alignment now.** Rejected for v0.1. A token-level
alignment between different segmentations needs its own evaluation to be
trustworthy, and shipping it unevaluated alongside a same-tokenizer path would
make it unclear which one a given result came from.

## Roadmap (not implemented)

Cross-tokenizer distillation is not implemented. The error messages raised by
`build_tokenizer` and `build_alignment_map` name it as a roadmap item and point
at `docs/limitations.md`.
