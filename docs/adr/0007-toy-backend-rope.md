# 0007. Rotary position embeddings in the toy backend

Status: Accepted, 2026-07-27.

## Context

miniVERL ships a `toy` model backend
(`src/miniverl/models/toy.py`) so that the whole pipeline -- rollout, teacher
scoring, cache write and read, chunked loss, checkpoint, resume, report --
can be exercised end to end on CPU in seconds, with a vocabulary small enough
that the exact full-vocabulary loss is affordable. The toy models are a test
harness for the machinery, not a capability claim.

For that harness to be useful, the toy models have to be able to *learn the
task at all*. The calculator environment requires the model to copy operands
out of the prompt into a tool-call JSON payload. If the toy model can produce
syntactically valid tool calls but cannot copy, then every end-to-end test
passes on a pipeline that never sees a solved task, and the environment's
verifier is never exercised on a success.

The first implementation used learned absolute position embeddings, which is
the smaller and more obvious choice at this scale.

## Decision

The toy transformer uses rotary position embeddings (RoPE) rather than learned
absolute position embeddings, alongside RMSNorm and a SwiGLU MLP -- structurally
the same family as Qwen and Llama, three orders of magnitude smaller.
`apply_rotary` and the cached `rope_cos` / `rope_sin` tables live in
`src/miniverl/models/toy.py`; `ToyAttention`'s docstring states the reason and
points at this ADR.

The decision was driven by a measurement, not by an appeal to architecture
fashion:

| configuration | training signal | held-out eval success |
| --- | --- | --- |
| learned absolute positions | train loss 0.0006 (memorized) | 0 percent |
| RoPE, 48 oracle traces | -- | 25 percent |
| RoPE, 256 oracle traces, 800 steps | -- | 87.5 percent |
| RoPE, 1024 oracle traces, 800 steps | -- | 18.8 percent |

With absolute positions the toy teacher drove training loss to 6e-4 and still
scored 0 percent on held-out tasks. The failure mode was specific and
diagnostic: every rollout emitted a valid tool call with the wrong operands.
The model had memorized the tool-call *syntax* and had not learned to copy.
Relative position information is what makes copy and induction behaviour
learnable at this size, so the loss curve looked healthy while the capability
was absent.

The last row is included because it is the counter-lesson: 1024 traces at the
same 800-step budget is only about three epochs and scores 18.8 percent.
Diversity helps only up to what the step budget can consume. The toy recipes
are sized accordingly.

For reference, the toy student's SFT convergence on the same task (hidden 96,
3 layers, batch 8) is step 100: 0 percent, 200: 25 percent, 300: 75 percent,
400: 100 percent, 600: 87.5 percent, with the whole run taking 41 s on CPU.
The non-monotonicity at step 600 is real and is another reason the toy backend
is positioned as a machinery harness.

## Consequences

Positive:

- The toy pipeline reaches solved tasks, so the verifier, the reward path, the
  success-rate metric and the report's solved/unsolved rendering are all
  exercised by CPU tests.
- The toy architecture matches the Hugging Face backend's family, so the same
  adapter code paths (attention with a KV cache, an untied LM head) are
  meaningful in both.
- The `max_position_embeddings` bound is still enforced, and overflowing it
  raises with a pointer to `models.<role>.toy.max_position_embeddings` rather
  than indexing past the RoPE table.

Negative:

- The toy backend is slower and has more moving parts than a positional-
  embedding lookup would have.
- RoPE tables are registered as non-persistent buffers, so they are rebuilt at
  construction; changing `max_position_embeddings` or `rope_theta` changes the
  model even though neither appears in a checkpoint's tensor files.
- The toy LM head is deliberately left untied while the Qwen3 pair has
  `tie_word_embeddings: true`, so the tied path is exercised only in the
  Hugging Face tests.
- None of these numbers transfer. They are toy-scale results on one
  environment at one difficulty, and they say nothing about Qwen3-0.6B.

## Alternatives considered

**Keep learned absolute positions and train longer.** Rejected on the
evidence: training loss was already 6e-4: the model had fit the training set.
More steps do not add a relative-position inductive bias.

**Keep absolute positions and change the environment so copying is not
required.** Rejected: copying operands into a tool call is the core behaviour
the agent loop exists to train. Removing it would make the harness pass while
testing nothing that matters.

**Drop the toy backend and test only against a real model.** Rejected: the CPU
test suite is what makes the numerics testable at all, including the exact
full-vocabulary loss, which is not affordable at a 151936-entry vocabulary.
