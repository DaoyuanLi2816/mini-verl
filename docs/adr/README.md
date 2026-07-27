# Architecture Decision Records

These records document the decisions that shaped miniVERL, together with the
evidence behind them and the costs they carry. They are written for a reader
deciding whether to trust the project, so each one states what was rejected and
what the decision makes worse, not only what it makes better.

An ADR is a historical record. If a decision is later reversed, a new ADR
supersedes the old one; the old file stays as it was written.

## Format

Each record has the same sections: Title, Status, Context, Decision,
Consequences (positive and negative), and Alternatives considered. Where a
decision leaves something unimplemented, that appears under an explicit
Roadmap heading and is labelled as not implemented.

Every claim in these records is traceable to code in this repository or to a
measurement recorded in it. Numbers that were not measured are not present.

## Index

| ADR | Title | Status | Primary source |
| --- | --- | --- | --- |
| [0001](0001-single-gpu-no-ray.md) | Single GPU, single process, no Ray | Accepted 2026-07-27 | `pyproject.toml`, `models/factory.py` |
| [0002](0002-same-tokenizer-v0.1.md) | Same tokenizer for student and teacher in v0.1 | Accepted 2026-07-27 | `models/factory.py`, `models/tokenizers.py`, `trajectory/alignment.py` |
| [0003](0003-token-provenance-representation.md) | Token provenance: typed spans, stored and re-derived masks | Accepted 2026-07-27 | `schemas/trajectory.py`, `trajectory/masks.py` |
| [0004](0004-topk-tail-cache-semantics.md) | Top-k plus tail teacher targets: a coarse-graining, and a lower bound | Accepted 2026-07-27 | `losses/bucketed.py`, `cache/store.py` |
| [0005](0005-resident-vs-swap.md) | Resident and swap memory strategies, and the quantized-swap refusal | Accepted 2026-07-27 | `training/memory.py`, `training/trainer.py` |
| [0006](0006-segment-wise-tokenization.md) | Segment-wise tokenization of the transcript | Accepted 2026-07-27 | `agent/transcript.py`, `agent/loop.py` |
| [0007](0007-toy-backend-rope.md) | Rotary position embeddings in the toy backend | Accepted 2026-07-27 | `models/toy.py` |
| [0008](0008-no-pickle-anywhere.md) | No pickle anywhere: safetensors for tensors, JSON for structure | Accepted 2026-07-27 | `cache/store.py`, `training/checkpoint.py`, `trajectory/io.py` |
| [0009](0009-reporting-without-matplotlib.md) | Reporting with hand-rolled inline SVG, no plotting library | Accepted 2026-07-27 | `reporting/charts.py`, `reporting/data.py` |

## How they relate

ADR 0001 sets the scope: one process on one device, which is the constraint
every other record answers to. ADR 0005 is the direct consequence for VRAM, and
ADR 0004 is the direct consequence for the teacher-target format.

ADR 0002, 0003 and 0006 form one chain about correctness of supervision.
Segment-wise tokenization (0006) makes per-token provenance exact; the span
schema (0003) makes it enforceable; the same-tokenizer contract (0002) makes
the student-teacher alignment checkable by comparing token ids rather than by
trusting an offset.

ADR 0008 and 0009 are about the artifacts a run leaves behind: readable without
executing code, and reportable without a network or a GPU.

## Related prior work

miniVERL is not the only implementation of on-policy distillation, and these
records name what already exists rather than working around it:

- [verl](https://github.com/verl-project/verl) (Apache-2.0) implements
  HybridFlow (arXiv:2409.19256) and has first-class on-policy distillation in
  core plus an Agent Loop for multi-turn tool calling, on Ray, from one H100
  upward.
- [TRL](https://github.com/huggingface/trl) has `GKDTrainer` (now under
  `trl.experimental.gkd`), and its `ServerDistillationTrainer` already offers a
  top-k plus optional tail teacher target.
- [KDFlow](https://github.com/songmzhang/KDFlow) (MIT) does on-policy and
  cross-tokenizer KD on Ray, SGLang and FSDP2, with 8-GPU examples.

The papers that motivate specific choices are cited in the records that use
them: arXiv:2602.12275 (on-policy context distillation, the
`privileged_context` teacher mode) and arXiv:2603.07079 (entropy-aware
on-policy distillation, listed under Roadmap in ADR 0004).
