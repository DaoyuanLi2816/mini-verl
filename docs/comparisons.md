# Choose the right distillation stack

miniVERL, verl, TRL GKD, KDFlow and research OPD harnesses serve different
workflows. Start from the execution environment and the artifact contract you
need, then choose the smallest stack that covers them.

The source snapshot behind this page was checked on 2026-07-29. Upstream
projects evolve, so follow the primary links in [references](references.md)
before making a long-lived infrastructure decision.

## Quick choice

| Your priority | Best starting point |
| --- | --- |
| Read, modify and audit OPD on one NVIDIA GPU | **miniVERL** |
| Scale training across accelerators and nodes | **verl** |
| Add generalized-JSD distillation to a Transformers training workflow | **TRL GKD** |
| Explore cross-tokenizer or multimodal KD | **KDFlow** |
| Reproduce one paper's OPD experiments | the paper's **research harness** |
| Train a fixed supervised dataset | **plain SFT** |

## Capability snapshot

| Dimension | miniVERL | verl | TRL GKD | KDFlow | OPSD | plain SFT |
| --- | --- | --- | --- | --- | --- | --- |
| **Design center** | Inspectable single-GPU OPD, plus local SFT/DPO/KD baselines | General RL and distillation post-training at scale | A distillation trainer inside the Transformers ecosystem | Distributed KD across policy, tokenizer and modality choices | Paper-oriented OPD experiments built on verl | Next-token learning on a fixed dataset |
| **Runtime shape** | One Python process; optional CUDA training stack | Ray with distributed model and rollout backends | Transformers Trainer + Accelerate | Ray + SGLang | verl-based | Framework-dependent |
| **On-policy path** | Strict current-policy rollout → teacher score → actor update | First-class distributed distillation and agent-loop paths | Configurable student-generated sequences | Available | Available | Fixed dataset |
| **Tool trajectories** | Calculator, JSON navigation, read-only SQLite and custom typed environments | Agent loop and tool parser | Chat-dataset training | Project-dependent | Tool-oriented experiments | Dataset-defined |
| **Teacher artifacts** | Exact or top-k targets, sampled-k1 signals and sharded safetensors cache | Distributed trainer state | Teacher forward pass in the trainer; server distillation supports top-k + tail controls | Chunked/distributed target handling | Chunked divergence path | Target tokens |
| **Primary output** | PEFT adapter, trajectories, cache, plan and portable provenance bundle | Distributed checkpoints and rollout/training artifacts | Transformers model or adapter | Project-defined model artifacts | Experiment artifacts | Model or adapter |
| **Best fit** | One-GPU experiments where semantic traceability matters | Throughput, scale and RL integration | Existing Transformers/TRL workflows | Broader KD research space | Reproducing its published setup | A known supervised target dataset |

The table summarizes each project's documented design center rather than
ranking quality or speed. Hardware numbers and algorithm outcomes are meaningful
only within their original model, data and runtime setup.

## miniVERL's design center

miniVERL concentrates on the parts of a local distillation run that benefit
from explicit evidence:

- a typed compiler that records how each source field affects local execution;
- strict policy-version binding between rollouts and teacher targets;
- token-span provenance created during generation;
- exact, top-k and sampled-k1 teacher signals with checksummed cache identity;
- transactional plans, checkpoints and export publication;
- standard PEFT, safetensors and Parquet interchange.

That design is especially useful when you have one personal NVIDIA GPU, want
to inspect or change the loss, and value semantic traceability over rollout
throughput.

## When scale is the main requirement

verl is the natural continuation when the workload needs multi-GPU execution,
high-throughput rollout engines, distributed checkpointing or RL objectives.
miniVERL's export path prepares a pinned bundle of local artifacts for that
workflow and reports artifact completeness, materialization and launchability
as separate states.

Use the [scale-out contract](verl-opd-scaleout.md) for that handoff. The full
list of miniVERL's algorithm, architecture and evidence boundaries lives in
[limitations](limitations.md), while [compatibility](compatibility.md) defines
the exact versioned profile contract.

## Evidence notes

The public miniVERL results cover one-GPU systems behavior and several scoped
task studies. They include negative outcomes and preregistered early stops.
Read the detailed reports before treating a method result as portable to a new
teacher, task, model pair or budget.

The external project names and descriptions above are attributed to their
primary repositories and papers. miniVERL is an independent project; the table
is an interoperability and workflow guide.
