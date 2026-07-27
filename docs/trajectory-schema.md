# Trajectory schema

A miniVERL trajectory is a **flat token sequence plus a partition of that
sequence into typed spans**. Every token belongs to exactly one span, and the
span type is the only thing that decides whether a token may contribute to a
training loss.

The schema is defined in `src/miniverl/schemas/trajectory.py`. Records are
serialized as JSON Lines by `src/miniverl/trajectory/io.py` (one JSON object per
line, no pickle and no `torch.save`, so reading a file from a stranger never
executes anything).

`TRAJECTORY_SCHEMA_VERSION` is currently `1`.

---

## Contents

- [Span types and provenance](#span-types-and-provenance)
- [`Trajectory` fields](#trajectory-fields)
- [`Span` fields](#span-fields)
- [`Turn`, `ToolCallRecord`, `ToolResultRecord`, `VerificationRecord`](#turn-toolcallrecord-toolresultrecord-verificationrecord)
- [`TerminationReason`](#terminationreason)
- [Context segments carry the trailing assistant header](#context-segments-carry-the-trailing-assistant-header)
- [Target positions vs prediction positions](#target-positions-vs-prediction-positions)
- [`AlignmentMap`](#alignmentmap)
- [What is validated, and the error each violation produces](#what-is-validated-and-the-error-each-violation-produces)
- [A real JSONL record](#a-real-jsonl-record)
- [Inspecting a file](#inspecting-a-file)

---

## Span types and provenance

`SpanType` has six members. Two derived frozensets classify them:

| `span_type` | trainable (`MODEL_GENERATED_SPAN_TYPES`) | critical (`CRITICAL_SPAN_TYPES`) | meaning |
| --- | --- | --- | --- |
| `system` | no | no | System prompt. Context. |
| `user` | no | no | Task statement. Context. |
| `tool_result` | no | no | Environment observation. Context. |
| `assistant_text` | **yes** | no | Free assistant text produced by the policy (including the text of an unparseable turn). |
| `assistant_tool_call` | **yes** | **yes** | The tool-call block the policy emitted. |
| `assistant_final` | **yes** | **yes** | The final-answer block the policy emitted. |

Three of the six are trainable, and two of those three are additionally
*critical*: the structurally load-bearing tokens, being the tool JSON and the
answer. `Span.is_model_generated` and `Span.is_critical` expose these two
predicates per span.

The distinction matters in two places. `model_generated_mask` gates what may
ever be supervised. `critical_mask` is what the `critical_only` and `hybrid`
selectors in `src/miniverl/selection/selectors.py` keep unconditionally, and it
is the target of `SelectionConfig.critical_weight`.

`tool_result` being non-trainable is the property the whole schema exists to
enforce. Tool output is text the model conditioned on but did not author;
supervising it teaches the policy to hallucinate environment responses.

## `Trajectory` fields

`Trajectory` is a Pydantic model with `extra="forbid"`, so an unknown key is a
hard error rather than a silently ignored field.

| field | type | meaning |
| --- | --- | --- |
| `schema_version` | `int` | Defaults to `TRAJECTORY_SCHEMA_VERSION` (`1`). The reader rejects any other value before attempting to parse the record. |
| `trajectory_id` | `str` | Unique id. `RolloutRunner.rollout` defaults to `"{task_id}:v{policy_version}:s{seed}"`; `oracle_rollout` defaults to `"{task_id}:oracle"`. The trainer overrides the oracle form with a cycle suffix. |
| `task_id` | `str` | Id of the task from the environment split. |
| `environment` | `str` | Environment name, e.g. `calculator`. |
| `token_ids` | `list[int]` | The flat sequence. For model spans these are the **sampled ids verbatim** — nothing is re-tokenized. |
| `attention_mask` | `list[int]` | Same length as `token_ids`; only `0` and `1` are permitted. `TranscriptBuilder.build` writes all ones. |
| `model_generated_mask` | `list[bool]` | `True` where the span type is in `MODEL_GENERATED_SPAN_TYPES`. Stored explicitly and re-derived from the spans by the validator. |
| `critical_mask` | `list[bool]` | `True` where the span type is in `CRITICAL_SPAN_TYPES`. Also re-derived and checked. |
| `spans` | `list[Span]` | The partition. Must be in ascending `start` order and must tile `[0, len(token_ids))` with no gap and no overlap. |
| `turns` | `list[Turn]` | One entry per assistant action. Defaults to empty. |
| `policy_version` | `int >= 0` | Which policy produced the rollout. Incremented once per OPD cycle, so it is what makes a stale teacher-cache entry detectable. Default `0`. |
| `tokenizer_fingerprint` | `str` | Behavioural tokenizer hash. Two tokenizers with the same fingerprint tokenize identically. Alignment refuses to proceed when the student and teacher fingerprints differ. |
| `model_id` | `str` | The model that generated the rollout. |
| `model_revision` | `str \| None` | Pinned revision, when one was configured. |
| `verification` | `VerificationRecord \| None` | The environment verifier's outcome. `None` when the episode never produced a final answer. |
| `termination_reason` | `TerminationReason` | Required. Exactly one per trajectory. |
| `generated_token_count` | `int >= 0` | Total tokens sampled from the policy across all turns. Default `0`. |
| `invalid_tool_calls` | `int >= 0` | Parse errors plus tool steps the environment rejected. Default `0`. |
| `metadata` | `dict[str, Any]` | Free-form. The rollout runner writes `source` (`"policy"`, `"oracle"` or `"privileged_teacher_render"`), and for policy rollouts also `seed`, `temperature`, `difficulty` and `split`. |

Derived views (computed, not stored):

- `length` — `len(token_ids)`.
- `model_token_positions()` — absolute indices where `model_generated_mask` is true.
- `critical_token_positions()` — absolute indices where `critical_mask` is true.
- `spans_of(*span_types)` — spans filtered by type.
- `span_at(position)` — the span containing an absolute index; raises `IndexError` outside the sequence.
- `token_counts_by_span_type()` — token totals keyed by the span-type string.

## `Span` fields

A `Span` is a half-open token range `[start, end)` with a single provenance
type. `extra="forbid"` applies here too.

| field | type | meaning |
| --- | --- | --- |
| `span_type` | `SpanType` | Provenance. Decides trainability. |
| `start` | `int >= 0` | First token index, inclusive. |
| `end` | `int >= 0` | One past the last token index. Must be strictly greater than `start`; an empty span is rejected. |
| `turn_id` | `int >= 0` | Which turn the span belongs to. Must match a `Turn.turn_id` whenever `turns` is non-empty. |
| `text` | `str` | The exact text of the segment. Defaults to `""`; `TranscriptBuilder.add` fills it by decoding the ids when the segment was supplied pre-tokenized. |
| `tool_name` | `str \| None` | Set on `assistant_tool_call` spans. |
| `tool_call_id` | `str \| None` | Correlates an `assistant_tool_call` span with its `tool_result` span. The rollout runner uses `"c{turn_id}"`. |
| `env_state_id` | `str \| None` | Environment state id after the step, recorded on `tool_result` spans. |
| `metadata` | `dict[str, Any]` | Always carries `segment_key` when built by `TranscriptBuilder` — the stable identity used for privileged-context alignment. |

Properties: `length` (`end - start`), `is_model_generated`, `is_critical`.

### Segment-wise tokenization

`TranscriptBuilder` tokenizes each segment on its own and concatenates the ids.
Two consequences follow, and both are deliberate:

1. A token can never straddle a provenance boundary, so "this token was
   generated by the model" is exact rather than approximate.
2. The teacher's render of a shared segment produces byte-identical ids, which
   is what makes privileged-context alignment checkable.

The cost is that concatenated segment ids may differ from tokenizing the whole
string in one call, because no BPE merge can cross a segment boundary.

## `Turn`, `ToolCallRecord`, `ToolResultRecord`, `VerificationRecord`

`Turn` — one assistant action and the environment response it triggered.

| field | type | meaning |
| --- | --- | --- |
| `turn_id` | `int >= 0` | Turn index, starting at `0`. |
| `tool_call` | `ToolCallRecord \| None` | The parsed call, or `None` for the final-answer turn. |
| `tool_result` | `ToolResultRecord \| None` | The environment response. `None` when the call was never executed (parse error, or the repeated-call limit). |
| `is_final` | `bool` | `True` on the turn that emitted the final answer. Default `False`. |

`ToolCallRecord` — a tool call emitted by the policy.

| field | type | meaning |
| --- | --- | --- |
| `call_id` | `str` | Correlation id, matching `Span.tool_call_id`. |
| `name` | `str` | Tool name. The rollout runner records the literal `"<unparsed>"` when the turn failed to parse. |
| `arguments` | `dict[str, Any]` | Parsed JSON arguments. Defaults to `{}`. |
| `raw_text` | `str` | The assistant text exactly as generated. Default `""`. |
| `valid` | `bool` | `False` for a parse error. Default `True`. |
| `parse_error` | `str \| None` | The model-readable parser message. |

`ToolResultRecord` — the environment's response.

| field | type | meaning |
| --- | --- | --- |
| `call_id` | `str` | Matches the originating `ToolCallRecord.call_id`. |
| `ok` | `bool` | Whether the step succeeded. |
| `result` | `str` | Result payload. Default `""`. |
| `error` | `str \| None` | Error text when `ok` is false. |
| `env_state_id` | `str \| None` | Environment state id after the step. |
| `duration_ms` | `float \| None` | Step duration. Not populated by `RolloutRunner`. |

`VerificationRecord` — the environment's exact verifier for a finished rollout.

| field | type | meaning |
| --- | --- | --- |
| `solved` | `bool` | Whether the final answer was correct. |
| `reward` | `float` | Scalar reward. Default `0.0`. |
| `predicted` | `str \| None` | The answer the policy gave. |
| `expected` | `str \| None` | The answer the environment expected. |
| `failure_category` | `str \| None` | Taxonomy label; `"solved"` on success. |
| `detail` | `str \| None` | Free-text explanation. |

## `TerminationReason`

Every trajectory records exactly one, and each cap in the rollout loop has its
own reason so the failure taxonomy in reports is exact.

| value | set when |
| --- | --- |
| `final_answer` | The policy emitted a parseable `<final>` block. The only reason that carries a `verification` record. |
| `max_turns` | The turn budget ran out. This is also the initial value of the loop variable, so it is what a rollout that ends without hitting any other condition reports. |
| `max_tokens` | `rollout.max_total_tokens` was reached before the next turn could start. |
| `parse_error_limit` | Parse errors exceeded `rollout.max_parse_errors`. |
| `repeated_call_limit` | An identical call signature repeated more than `rollout.max_repeated_calls` times. |
| `environment_error` | Defined in the enum. Not emitted by `RolloutRunner`. |
| `eos_without_final` | The backend emitted EOS, or produced zero tokens, without a final answer. |

## Context segments carry the trailing assistant header

`TranscriptBuilder.add_context(..., open_next_assistant=True)` appends the
`<|im_start|>assistant\n` header of the *next* turn to the **context** segment
that precedes it. Symmetrically, `close_previous=True` prepends the
`<|im_end|>\n` that closes the *previous* assistant turn to the following
context segment.

The effect is visible in the real record below. The `user` span is
`[97, 138)` and its text ends with `<|im_start|>assistant\n`; the
`assistant_tool_call` span starts at `138`. The `tool_result` span `[177, 211)`
begins with `<|im_end|>\n` — the marker that closes the assistant's tool-call
turn — and again ends with `<|im_start|>assistant\n`.

Why the framing is arranged this way:

- **Generation begins exactly at the first token of a model span.** The prefix
  handed to `backend.generate` is the full context including the header, so
  index `138` is genuinely the first token the policy chose.
- **No forced scaffolding token is ever marked model-generated.** The header and
  the closing `<|im_end|>` are template text that the sampler never had a choice
  about. Supervising them would train the policy on tokens it did not select,
  and would inflate every trainable-token count.
- The trajectory therefore **ends at the last generated token** — there is no
  trailing `<|im_end|>` after `assistant_final`, because nothing followed it to
  carry one.

## Target positions vs prediction positions

The most dangerous bug in a distillation trainer is an off-by-one between the
token being predicted and the distribution used to predict it. miniVERL keeps
two separate vocabularies for the word "position", defined in
`src/miniverl/trajectory/masks.py`:

- **target position** `j` — the index of the token whose identity is supervised.
- **prediction position** `j - 1` — the index whose output distribution predicts
  the token at `j`.

Nothing converts between them implicitly. Every conversion goes through
`prediction_positions()`, and `j = 0` is always rejected because no distribution
precedes it. `model_target_positions()` excludes index `0` even when it is
marked model-generated.

### Worked example

Indices below are from the real record in the next section
(`calc-train-1:oracle:c-2`, 218 tokens, toy tokenizer). Its trainable target
positions are exactly `{138 … 176} ∪ {211 … 217}`, which is 39 + 7 = 46
positions — matching `model_generated_mask` summing to 46.

| index | token id | decoded | span at this index | a target? | prediction position |
| ---: | ---: | --- | --- | --- | --- |
| 136 | 11 | `assistant` | `user` (context) | no | — |
| 137 | 89 | `\n` | `user` (context) | no | — |
| 138 | 3 | `<tool_call>` | `assistant_tool_call` | yes | 137, inside `user` (**context**) |
| 139 | 89 | `\n` | `assistant_tool_call` | yes | 138, inside `assistant_tool_call` |
| 140 | 182 | `{` | `assistant_tool_call` | yes | 139, inside `assistant_tool_call` |
| 176 | 4 | `</tool_call>` | `assistant_tool_call` | yes | 175, inside `assistant_tool_call` |
| 177 | 2 | `<\|im_end\|>` | `tool_result` (context) | no | — |
| 211 | 7 | `<final>` | `assistant_final` | yes | 210, inside `tool_result` (**context**) |
| 212 | 89 | `\n` | `assistant_final` | yes | 211, inside `assistant_final` |
| 217 | 8 | `</final>` | `assistant_final` | yes | 216, inside `assistant_final` |

Two rows carry the whole point. At `j = 138` and `j = 211` the *prediction*
position lands inside a **context** span. That is correct and necessary: the
distribution that predicts the first token of an assistant turn is the one
produced after reading the assistant header. This is precisely why the header
must live in the context segment — if it were part of the model span, index
`138` would no longer be the first sampled token and the mask would claim the
policy authored template text.

Rows 136, 137 and 177 show the other half of the rule. Index 177 is **not**
supervised even though it directly follows 39 model tokens, because the
`<|im_end|>` that closes the assistant's turn belongs to the following
`tool_result` context span. Indices 136 and 137 are the assistant header, which
for the same reason belongs to the preceding `user` span.

Reproduce the table with:

```python
from miniverl.trajectory.io import read_trajectories
from miniverl.trajectory.masks import model_target_positions, prediction_positions

traj = read_trajectories("runs/demo/trajectories.jsonl")[0]
targets = model_target_positions(traj.model_generated_mask)
predictions = prediction_positions(targets)
for j, p in list(zip(targets, predictions))[:3]:
    print(
        j,
        traj.token_ids[j],
        traj.span_at(j).span_type.value,
        "<-",
        p,
        traj.span_at(p).span_type.value,
    )
```

### Guards

`validate_target_positions(target_positions, model_generated_mask)` is the check
that makes "tool outputs are context, not labels" a tested property rather than
a comment. It raises `TrajectoryError` with these exact messages:

| condition | message |
| --- | --- |
| index outside the sequence | `target position 9 is outside [0, 2)` |
| index `0` | `position 0 can never be a training target` |
| not model-generated | `target position 1 is not a model-generated token; system, user and tool-result tokens must never be supervised` |
| repeated index | `target position 1 appears more than once` |
| not ascending | `target positions must be strictly increasing` |

`prediction_positions([0])` separately raises `target position 0 has no
preceding prediction position; position 0 can never be a training target`.

## `AlignmentMap`

Defined in `src/miniverl/schemas/alignment.py`. Six parallel lists of the same
length `N`. Entry `i` states: *the student distribution at
`student_prediction_positions[i]` and the teacher distribution at
`teacher_prediction_positions[i]` both predict token `target_token_ids[i]`, and
it contributes with weight `token_weights[i]`.*

| field | type | meaning |
| --- | --- | --- |
| `trajectory_id` | `str` | The student trajectory this map belongs to. |
| `student_prediction_positions` | `list[int]` | Prediction positions in student space. Must be non-negative and strictly increasing. |
| `teacher_prediction_positions` | `list[int]` | Prediction positions in teacher space. Must be non-negative. Not required to be increasing or equal to the student's. |
| `target_token_ids` | `list[int]` | The supervised token at each entry. Verified to be identical on both sides at construction. |
| `model_token_mask` | `list[bool]` | Whether the entry is a model-generated token. |
| `token_weights` | `list[float]` | Non-negative per-position loss weights. A `False` entry in `model_token_mask` must have weight exactly `0.0`. |
| `span_types` | `list[str]` | Span-type string per entry, used for the per-span-type loss breakdown in metrics. |

Properties: `num_positions`, `total_weight` (the loss normalizer),
`is_identity()` (teacher and student position lists coincide exactly), and
`counts_by_span_type()`.

Weights come from the selector: `SelectionConfig.critical_weight` for positions
in `critical_mask`, `SelectionConfig.other_weight` otherwise.

### The two teacher context modes

Built by `build_alignment_map` in `src/miniverl/trajectory/alignment.py`.

**`standard`** — the teacher sees the byte-identical transcript, so
`teacher_prediction_positions == student_prediction_positions` and
`is_identity()` is `True`. `teacher=None` produces this via
`identity_alignment`.

**`privileged_context`** — `RolloutRunner.privileged_render` rebuilds the
transcript with an extra oracle `system` block inserted before the shared
content. Every student segment keeps its `segment_key` and its exact token ids,
but its absolute positions shift. The offset is **not** assumed constant: spans
are matched by `segment_key` and the offset is recomputed per span as
`tspan.start + (j - span.start)`.

Measured on the record below, with the privileged block occupying 55 tokens:

| entry | student prediction | teacher prediction | target id | span type |
| ---: | ---: | ---: | ---: | --- |
| 0 | 137 | 192 | 3 | `assistant_tool_call` |
| 1 | 138 | 193 | 89 | `assistant_tool_call` |
| 45 | 216 | 271 | 8 | `assistant_final` |

`is_identity()` is `False`, `num_positions` is 46, `total_weight` is 46.0.

In both modes the alignment is accepted only if the **target token ids are
identical on both sides**. That is what makes the same-tokenizer contract
enforceable rather than aspirational. `build_alignment_map` raises:

| condition | error type and message |
| --- | --- |
| tokenizer fingerprints differ | `TokenizerMismatchError`: `student and teacher tokenizers differ (… vs …); miniVERL v0.1 only supports same-tokenizer distillation` |
| a span lacks `segment_key` | `AlignmentError`: `span <type> at [a,b) has no 'segment_key' metadata; …` |
| duplicate teacher segment key | `AlignmentError`: `teacher render contains duplicate segment key …; keys must be unique` |
| student segment missing on the teacher side | `AlignmentError`: `student span … has no counterpart in the teacher render; …` |
| shared segment tokenized to a different length | `AlignmentError`: `segment … has N student tokens but M teacher tokens; the shared content must tokenize identically on both sides` |
| target token ids disagree | `AlignmentError`: `target token mismatch at student position j / teacher position k: a != b` |
| a target lands at teacher position `0` | `AlignmentError`: `a target token landed at teacher position 0, which has no preceding prediction position` |
| weight count differs from position count | `AlignmentError`: `got N weights for M target positions` |

`AlignmentMap`'s own validator adds:

| condition | message |
| --- | --- |
| parallel list length mismatch | `alignment field 'target_token_ids' has length 1, expected 2` |
| negative student position | `student prediction positions must be >= 0` |
| negative teacher position | `teacher prediction positions must be >= 0` |
| non-increasing student positions (including duplicates) | `student prediction positions must be strictly increasing` |
| negative weight | `token weights must be non-negative` |
| non-model token with non-zero weight | `a non-model token was given a non-zero weight; tool/user/system tokens must never contribute to the distillation loss` |

An empty map (`N == 0`) is valid and short-circuits the remaining checks.

## What is validated, and the error each violation produces

Validation runs on **both** write and read. `write_trajectories` serializes
already-validated `Trajectory` objects; `iter_trajectories` re-validates every
line as it streams, so a hand-edited or tampered file fails loudly instead of
silently training on tool output.

`iter_trajectories` raises `SchemaValidationError` in every case below. The
messages are prefixed with `<path>:<lineno>`.

| violation | error message (verbatim, path elided) |
| --- | --- |
| file missing | `trajectory file not found: <path>` (hint: `run 'miniverl demo' or 'miniverl train <recipe>' first`) |
| line is not JSON | `<file>:1 is not valid JSON: <json decoder message>` |
| unknown `schema_version` | `<file>:1 has trajectory schema_version 2, this build reads version 1` |
| empty sequence | `Value error, trajectory has no tokens` |
| mask length mismatch | `Value error, attention_mask has length 217 but token_ids has length 218` |
| attention mask not 0/1 | `Value error, attention_mask must contain only 0/1` |
| no spans | `Value error, trajectory has no spans` |
| spans out of order | `Value error, spans must be stored in ascending start order` |
| gap or overlap | `Value error, spans must tile the sequence without gaps or overlaps: expected next span to start at 97, got 98` |
| spans do not cover the sequence | `Value error, spans cover 211 tokens but the sequence has 218` |
| empty span | `Value error, span assistant_tool_call has empty range [138,138)` (reported under `spans.2`) |
| `model_generated_mask` disagrees with the spans | `Value error, model_generated_mask disagrees with the span partition; tool/user/system tokens must never be marked model-generated` |
| `critical_mask` disagrees with the spans | `Value error, critical_mask disagrees with the span partition` |
| span references an unknown turn | `Value error, span system references unknown turn_id 9` |
| unknown key anywhere in the record | `Extra inputs are not permitted` (reported under the offending key) |

The two mask checks are the load-bearing ones. The masks are stored explicitly
rather than recomputed on read, and the validator re-derives them from the span
partition and rejects any mismatch. Flipping a single `False` to `True` on a
`system` token in a saved file is enough to trigger the first message.

`count_trajectories` is the one function that does **not** validate: it counts
non-empty lines and returns `0` for a missing file.

## A real JSONL record

Produced by `miniverl demo --fast`, which uses the embedded toy backend and
needs no network. The token ids below are toy-tokenizer ids, not Qwen ids.

```bash
miniverl demo --fast --no-report          # writes runs/demo
head -1 runs/demo/trajectories.jsonl
```

The file stores each record as a **single line** (this one is 7,249 bytes).
Pretty-printed and trimmed below: `token_ids`, `attention_mask`,
`model_generated_mask` and `critical_mask` are 218 entries long in the file and
are shown truncated to their first 8, and long `span.text` values are cut with
`...`. Everything else is verbatim.

```json
{
  "schema_version": 1,
  "trajectory_id": "calc-train-1:oracle:c-2",
  "task_id": "calc-train-1",
  "environment": "calculator",
  "token_ids": [1, 69, 89, 144, 174, 160, 91, 76, "..."],
  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, "..."],
  "model_generated_mask": [false, false, false, false, false, false, false, false, "..."],
  "critical_mask": [false, false, false, false, false, false, false, false, "..."],
  "spans": [
    {
      "span_type": "system",
      "start": 0,
      "end": 97,
      "turn_id": 0,
      "text": "<|im_start|>system\nUse tools to solve the task.\n- calculator(expre...",
      "tool_name": null,
      "tool_call_id": null,
      "env_state_id": null,
      "metadata": {"segment_key": "sys"}
    },
    {
      "span_type": "user",
      "start": 97,
      "end": 138,
      "turn_id": 0,
      "text": "<|im_start|>user\nCompute (17 * 10) and report the value.<|im_end|>...",
      "tool_name": null,
      "tool_call_id": null,
      "env_state_id": null,
      "metadata": {"segment_key": "user"}
    },
    {
      "span_type": "assistant_tool_call",
      "start": 138,
      "end": 177,
      "turn_id": 0,
      "text": "<tool_call>\n{\"arguments\": {\"expression\": \"(17 * 10)\"}, \"name\": \"ca...",
      "tool_name": "calculator",
      "tool_call_id": "c0",
      "env_state_id": null,
      "metadata": {"segment_key": "gen:0:block"}
    },
    {
      "span_type": "tool_result",
      "start": 177,
      "end": 211,
      "turn_id": 0,
      "text": "<|im_end|>\n<|im_start|>user\n<tool_result>\n{\"ok\": true, \"result\": \"...",
      "tool_name": null,
      "tool_call_id": "c0",
      "env_state_id": "calc:1",
      "metadata": {"segment_key": "obs:0"}
    },
    {
      "span_type": "assistant_final",
      "start": 211,
      "end": 218,
      "turn_id": 1,
      "text": "<final>\n170\n</final>",
      "tool_name": null,
      "tool_call_id": null,
      "env_state_id": null,
      "metadata": {"segment_key": "gen:1:block"}
    }
  ],
  "turns": [
    {
      "turn_id": 0,
      "tool_call": {
        "call_id": "c0",
        "name": "calculator",
        "arguments": {"expression": "(17 * 10)"},
        "raw_text": "<tool_call>\n{\"arguments\": {\"expression\": \"(17 * 10)\"}, \"name\": \"calculator\"}\n</tool_call>",
        "valid": true,
        "parse_error": null
      },
      "tool_result": {
        "call_id": "c0",
        "ok": true,
        "result": "170",
        "error": null,
        "env_state_id": "calc:1",
        "duration_ms": null
      },
      "is_final": false
    },
    {"turn_id": 1, "tool_call": null, "tool_result": null, "is_final": true}
  ],
  "policy_version": 0,
  "tokenizer_fingerprint": "d8b8c2b5a24f32edc5a90bf6742d508fc16e52e9ca6ff9056102ac6405bce541",
  "model_id": "toy-student",
  "model_revision": null,
  "verification": {
    "solved": true,
    "reward": 1.0,
    "predicted": "170",
    "expected": "170",
    "failure_category": "solved",
    "detail": null
  },
  "termination_reason": "final_answer",
  "generated_token_count": 46,
  "invalid_tool_calls": 0,
  "metadata": {"source": "oracle", "difficulty": "easy", "split": "train"}
}
```

Reading this record against the rules above: 218 tokens, five spans tiling
`[0, 218)` with no gap, 46 tokens marked model-generated (the 39-token tool call
plus the 7-token final answer), all 46 also critical, and the remaining 172
`system` / `user` / `tool_result` tokens (97 + 41 + 34) excluded from the loss.

Note the `<tool_call>` / `<tool_result>` / `<final>` text protocol from
`src/miniverl/agent/protocol.py`. It is deliberately not a vendor
function-calling format: every supported model emits the same three plain-text
blocks, so the same schema, parser and masks work across model families without
special-token surgery or an embedding resize.

## Inspecting a file

`miniverl inspect` is torch-free — it validates and summarizes using nothing but
the base install. The provenance summary it prints is computed from the span
partition, which is the same source the training masks come from, so a file that
claims tool output is trainable is reported that way rather than hidden.

```bash
miniverl inspect runs/demo/trajectories.jsonl --limit 2
```

```
  8 trajectories | 2042 tokens | 263 model tokens (12.9%) | 178 critical
  graded 4 | solved 4 (100.0%)
  policy versions [0, 1, 2]
  termination {'final_answer': 4, 'max_turns': 4}
  tools {'calculator': 4}
  tokenizer ['d8b8c2b5a24f...']
tokens by span type (only assistant_* can enter
                   the loss)
+---------------------------------------------+
| span type           | tokens | in loss      |
|---------------------+--------+--------------|
| system              |    776 | no (context) |
| tool_result         |    685 | no (context) |
| user                |    318 | no (context) |
| assistant_tool_call |    153 | yes          |
| assistant_text      |     85 | yes          |
| assistant_final     |     25 | yes          |
+---------------------------------------------+
```

Options:

| flag | effect |
| --- | --- |
| `--limit N` | Number of per-trajectory rows to show. Default `5`. |
| `--trajectory ID` | Restrict the summary to one trajectory id. |
| `--spans` | With `--trajectory`, print the span table: type, `[start, end)`, token count, whether it enters the loss, and the first 400 characters of text. |
| `--json` | Machine-readable `FileSummary.to_dict()` on stdout. |

The span table for the record above:

```bash
miniverl inspect runs/demo/trajectories.jsonl \
  --trajectory calc-train-1:oracle:c-2 --spans
```

```
| span                | range      | tokens | in loss | text
| system              | [0, 97)    | 97     | no      | <|im_start|>system ...
| user                | [97, 138)  | 41     | no      | <|im_start|>user ...
| assistant_tool_call | [138, 177) | 39     | yes     | <tool_call> ...
| tool_result         | [177, 211) | 34     | no      | <|im_end|> ...
| assistant_final     | [211, 218) | 7      | yes     | <final> 170 </final>
```

The `--json` output adds a `provenance_check` block that names the rule set the
summary was computed under, so a report can be audited without re-reading the
source:

```json
"provenance_check": {
  "trainable_span_types": ["assistant_final", "assistant_text", "assistant_tool_call"],
  "critical_span_types": ["assistant_final", "assistant_tool_call"],
  "context_span_types_excluded_from_loss": ["system", "tool_result", "user"]
}
```

Because every record is schema-validated during the walk, `miniverl inspect`
exiting non-zero is itself the integrity check — a mask that disagrees with its
span partition makes the command fail rather than print a reassuring summary.

### Reading a file programmatically

```python
from miniverl.trajectory.io import iter_trajectories, read_trajectories, count_trajectories

count_trajectories("runs/demo/trajectories.jsonl")  # line count, no validation
trajectories = read_trajectories("runs/demo/trajectories.jsonl")  # list, validated
for traj in iter_trajectories("runs/demo/trajectories.jsonl"):  # streamed, validated
    print(traj.trajectory_id, traj.length, traj.token_counts_by_span_type())
```
