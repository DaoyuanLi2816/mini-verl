# 0001. Single GPU, single process, no Ray

Status: Accepted, 2026-07-27.

## Context

On-policy distillation for tool-using agents is already implemented at cluster
scale. [verl](https://github.com/verl-project/verl) (Apache-2.0, the reference
implementation of HybridFlow, arXiv:2409.19256) ships first-class distillation
in core under `verl/trainer/distillation/` with a `distillation.*` config
namespace, a GKD-style forward-KL objective and a policy-gradient reverse-KL
variant, plus an Agent Loop for multi-turn tool calling. It supports FSDP,
FSDP2 and Megatron-LM training backends with vLLM, SGLang or Hugging Face
rollout, and its sizing documentation starts at one H100 and scales to hundreds
of GPUs. `ray[default]` is an unconditional entry in its `requirements.txt`.
[KDFlow](https://github.com/songmzhang/KDFlow) (MIT) makes the same choice:
Ray plus SGLang plus FSDP2, with examples that assume eight GPUs per node.

That leaves an unserved case rather than an unsolved problem: a single consumer
card. The development machine for this project is one RTX 4080 with 16376 MiB
of VRAM. A Ray head node, a placement group and a resource-aware scheduler add
no capability at one device, but they do add a heavyweight required dependency,
a second failure surface and a layer between the user and the tensor
operations.

## Decision

miniVERL runs in one process on one device. There is no Ray, no
`torch.distributed` process group, no launcher and no sharding. A grep for
`ray`, `torchrun`, `DDP` or `distributed` over `src/miniverl/` returns no
matches outside an unrelated numpy call.

Device selection is a single function, `resolve_device` in
`src/miniverl/models/factory.py`: `models.device` accepts `auto`, `cpu` or
`cuda`, and `auto` resolves to `cuda` when `torch.cuda.is_available()` is true.
Asking for `cuda` when torch sees no device raises `ConfigError` with a pointer
to `miniverl doctor` rather than falling back silently.

The dependency split in `pyproject.toml` follows from the same decision. The
base install is torch-free (`typer`, `rich`, `pydantic`, `pyyaml`, `jinja2`,
`platformdirs`, `safetensors`); `torch`, `transformers`, `peft`, `accelerate`
and `numpy` live in the `train` extra; `bitsandbytes` lives in the `cuda`
extra. There is no `distributed` extra to add later without a redesign.

## Consequences

Positive:

- Memory policy is a local decision made by one module
  (`src/miniverl/training/memory.py`) with a recorded reason string, instead of
  an emergent property of a scheduler.
- Reproducibility is tractable: one RNG lineage, captured and restored by
  `src/miniverl/utils/seeding.py`, and `tests/integration/test_resume_and_swap.py`
  asserts that an interrupted and an uninterrupted run agree exactly.
- Report and cache inspection work from a bare `pip install miniverl` on a
  laptop, because nothing in those paths imports torch.
- The failure modes a reader has to hold in their head are OOM and a bad
  config, not actor placement or object-store spill.

Negative:

- There is no path to a second GPU. Adding one is a redesign, not a flag.
- The teacher must fit next to the student or be swapped out; see ADR 0005.
- Throughput is bounded by single-sequence decode. On this machine a
  64-new-token probe measured 11.19 tok/s for the NF4 student and 12.84 tok/s
  for the bf16 LoRA student under determinism; a 14-token prefill cost 37.0 ms
  against 30.9 ms for a cached one-token step, so decoding is kernel-launch
  bound. Batched or continuous-batching rollout is not implemented.
- Any capability claim from this project is a small-model claim. Nothing here
  has been run above 1.7B parameters.

## Alternatives considered

**Build on verl.** Rejected as a different project, not a better version of
this one. verl already covers the cluster case, including on-policy
distillation and tool use, and its Ray dependency is unconditional, so a
consumer-GPU user inherits the cluster machinery whether or not they use it.

**Use Ray for single-node actor isolation.** Rejected: the only isolation
miniVERL needs is between the teacher-scoring phase and the update phase, and
that is a VRAM-residency question solved directly in `memory.py`.

**Ship an untested multi-GPU path behind a flag.** Rejected under the project's
evidence rule. Nothing in the codebase has been executed on more than one
device, so a `--multi-gpu` flag would be an unverifiable claim.

## Roadmap (not implemented)

Multi-GPU training, tensor or pipeline parallelism, and batched or
continuous-batching rollout are not implemented and are not planned for v0.1.
