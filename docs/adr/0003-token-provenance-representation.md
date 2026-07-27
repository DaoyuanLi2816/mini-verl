# 0003. Token provenance: typed spans, stored and re-derived masks

Status: Accepted, 2026-07-27.

## Context

An agentic rollout is one flat token sequence containing tokens from three
different sources: the prompt, the policy, and the environment. Tool results in
particular are appended to the same context the model attends over. Training on
them teaches the model to hallucinate tool output, and nothing about the loss
value reveals that it happened. OPSD handles the same hazard by excluding tool
tokens from the loss; miniVERL needs the exclusion to be a checked property
rather than a convention followed by whoever wrote the rollout loop.

A second hazard is the target/prediction off-by-one. The distribution that
predicts the token at index `j` lives at index `j - 1`. A single implicit
conversion in the wrong direction produces a loss that decreases while teaching
the model the wrong thing.

## Decision

A `Trajectory` (`src/miniverl/schemas/trajectory.py`) is a flat `token_ids`
list plus a list of `Span` objects that tile the sequence with no gaps and no
overlaps. Every token belongs to exactly one span, and the span's `SpanType`
decides whether the token may ever be a training target:

- `system`, `user`, `tool_result` are context and can never be targets.
- `assistant_text`, `assistant_tool_call`, `assistant_final` were produced by
  the policy (`MODEL_GENERATED_SPAN_TYPES`).
- `assistant_tool_call` and `assistant_final` are additionally *critical*
  (`CRITICAL_SPAN_TYPES`): the tool JSON and the answer.

`model_generated_mask` and `critical_mask` are **stored in the file and
re-derived by the validator**. The `_validate_structure` model validator
rebuilds both masks from the span partition and raises when the stored values
disagree, with a message that names the hazard ("tool/user/system tokens must
never be marked model-generated"). Storing them means a consumer such as the
report renderer or `miniverl inspect` can use them without walking spans;
re-deriving them means a hand-edited or tampered file fails loudly instead of
training on tool output. `tests/unit/test_token_provenance.py` covers this
directly, including
`test_tampered_mask_marking_tool_output_trainable_is_rejected` and
`test_masks_are_derived_from_spans_not_trusted`.

The position convention is a separate module,
`src/miniverl/trajectory/masks.py`, with two named concepts: *target position*
`j` and *prediction position* `j - 1`. Nothing converts between them
implicitly; every conversion goes through `prediction_positions`, and `j = 0`
is rejected everywhere because no distribution precedes it.
`validate_target_positions` additionally rejects positions that are not
model-generated, duplicated, or out of order.

Context segments own the trailing assistant header. In
`src/miniverl/agent/transcript.py`, `add_context(..., open_next_assistant=True)`
appends `<|im_start|>assistant\n` to the end of the *context* segment that
precedes the model's turn. Generation therefore starts exactly at the first
token of a model span, and the forced scaffolding tokens are never marked
model-generated. If the header belonged to the assistant span instead, the
model would be supervised on tokens it was never free to choose, and the
"critical" token counts in every report would be inflated by the frame.
`tests/unit/test_transcript.py::test_context_segments_carry_the_assistant_header`
pins this.

## Consequences

Positive:

- "Tool outputs are context, not labels" is enforced by a validator that runs
  on every read and every write, including on files produced by an older build.
- Span metadata carries what alignment and reporting need: `segment_key`,
  `turn_id`, `tool_name`, `tool_call_id`, `env_state_id`.
- Selectors and loss weighting can address `critical` tokens as a first-class
  category rather than by string matching the decoded text.

Negative:

- Every builder must account for every token. A span partition with a
  one-token gap is rejected, which makes ad-hoc trajectory construction in
  tests more verbose.
- Validation is O(n) Python per trajectory on both write and read, with two
  boolean lists rebuilt each time. At the sequence lengths used here (the 16 GB
  recipe caps `rollout.max_total_tokens` at 704) this is not a bottleneck, but
  it is not free either.
- The JSONL files carry two boolean arrays that are recomputable, so they are
  larger than strictly necessary.

## Alternatives considered

**Derive masks on read, never store them.** Rejected: consumers that only want
the masks would pay a span walk, and, more importantly, there would be nothing
to cross-check, so a corrupted span list would be accepted as the truth.

**Store only masks, drop spans.** Rejected: masks lose the span type, the turn
id and the tool identity, all of which the selectors, the privileged-context
alignment and the reports need.

**A per-token integer label array.** Rejected for the same reason: an integer
per token cannot carry `segment_key`, which is what makes privileged-context
alignment checkable (ADR 0002 and `alignment.py`).

**Trust the rollout loop and document the rule in a comment.** Rejected. This
is the failure the whole schema exists to prevent.
