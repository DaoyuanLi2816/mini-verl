# Choose your training workflow

miniVERL, verl, TRL GKD, KDFlow and research OPD harnesses serve different
workflows. Start with what you want to train and where you want to run it.

The source snapshot behind this page was checked on 2026-07-29. Upstream
projects evolve, so follow the primary links in [references](references.md)
before making a long-lived infrastructure decision.

## Quick choice

| Your priority | Best starting point |
| --- | --- |
| Prototype and inspect supported verl RL or OPD on one NVIDIA GPU | **miniVERL** |
| Scale training across accelerators and nodes | **verl** |
| Add generalized-JSD distillation to a Transformers training workflow | **TRL GKD** |
| Explore cross-tokenizer or multimodal KD | **KDFlow** |
| Reproduce one paper's OPD experiments | the paper's **research harness** |
| Train a fixed supervised dataset | **plain SFT** |

## Capability snapshot

| Dimension | miniVERL | verl | TRL GKD | KDFlow | OPSD | plain SFT |
| --- | --- | --- | --- | --- | --- | --- |
| **Design center** | Local RL and OPD on one consumer GPU, plus SFT/DPO/KD | General RL and distillation post-training at scale | A distillation trainer inside the Transformers ecosystem | Distributed KD across policy, tokenizer and modality choices | Paper-oriented OPD experiments built on verl | Next-token learning on a fixed dataset |
| **Runtime shape** | One Python process; optional CUDA training stack | Ray with distributed model and rollout backends | Transformers Trainer + Accelerate | Ray + SGLang | verl-based | Framework-dependent |
| **On-policy path** | Grouped rollout → reward/reference or teacher → actor update | First-class distributed RL, distillation and agent-loop paths | Configurable student-generated sequences | Available | Available | Fixed dataset |
| **Tool trajectories** | Calculator, JSON navigation, read-only SQLite and custom typed environments | Agent loop and tool parser | Chat-dataset training | Project-dependent | Tool-oriented experiments | Dataset-defined |
| **Teacher artifacts** | Exact or top-k targets, sampled-k1 signals and sharded safetensors cache | Distributed trainer state | Teacher forward pass in the trainer; server distillation supports top-k + tail controls | Chunked/distributed target handling | Chunked divergence path | Target tokens |
| **Primary output** | PEFT adapter, trajectories, cache, plan and portable provenance bundle | Distributed checkpoints and rollout/training artifacts | Transformers model or adapter | Project-defined model artifacts | Experiment artifacts | Model or adapter |
| **Best fit** | One-GPU experiments where semantic traceability matters | Throughput, scale and RL integration | Existing Transformers/TRL workflows | Broader KD research space | Reproducing its published setup | A known supervised target dataset |

## miniVERL's design center

miniVERL makes the local experiment easy to follow: reuse your verl config,
inspect generated responses and learning signals, resume interrupted training,
and export a model you can use elsewhere. PPO, grouped reward-driven methods
and OPD share these tools.

Start here when you have one NVIDIA GPU and want a short path from configuring
an experiment to understanding its behavior.

## When scale is the main requirement

verl is the natural continuation when the workload needs multi-GPU execution,
distributed checkpointing or the broader algorithm/runtime surface. miniVERL's
export path prepares a pinned bundle of local artifacts for that workflow and
reports artifact completeness, materialization and launchability as separate
states.

Use the [scale-out contract](verl-opd-scaleout.md) for that handoff. The full
list of miniVERL's algorithm, architecture and evidence boundaries lives in
[limitations](limitations.md), while [compatibility](compatibility.md) defines
the exact versioned profile contract.

## Scope and sources

The table describes workflows and documented features. Performance comparisons
need matched model, data and runtime settings; the dated upstream snapshot is
linked above.

The public miniVERL results cover one-GPU systems behavior and several scoped
task studies. They include negative outcomes and preregistered early stops.
Read the detailed reports before treating a method result as portable to a new
teacher, task, model pair or budget.

The external project names and descriptions above are attributed to their
primary repositories and papers. miniVERL is an independent project; the table
is an interoperability and workflow guide.
