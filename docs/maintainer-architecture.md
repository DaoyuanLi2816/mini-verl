# Maintainer architecture

This page maps the change boundaries maintainers use for review.

## Config to runtime

`bridge.profiles` selects a closed OPD profile, while `bridge.rl_v09` owns the
resolved verl v0.9 RL subset. Both classify source fields before publishing a
native `RunConfig`. `bridge.opd_runtime` and `bridge.opd_plan` add placement,
data and immutable-plan identity for OPD; the RL compiler emits its native
recipe and field report transactionally. `OPDTrainer.from_config` remains the
public runtime facade for native modes. No generic dynamic profile loader
exists.

The profile identity enters immutable plans and is copied into run manifests,
teacher-cache metadata, checkpoint identity and export/materialization reports.
Any reader that combines those artifacts compares the complete identity before
using their contents.

## Logical roles and physical placement

Actor, teacher, reference, reward and critic are distinct logical
identities. Rollout generates with the current actor policy. OPD teacher
scoring observes only visited positions; RL reward providers consume ordered
trajectory identities; reference KL evaluates a frozen adapter before the
actor is restored for update. `PPOPhaseRuntime` owns old-value scoring and
independent clipped critic updates. Physical placement may be resident phased
models, an allowed unquantized swap, or a shared backbone with separate
adapters. Quantized swap remains illegal.

`training.trainer.OPDTrainer` is the compatibility facade and state-machine
owner. Batching, memory planning, optimizer construction, checkpoint I/O and
offline-dataset persistence live in dedicated `training` modules; model and
tokenizer construction live under `models`; cache transactions live under
`cache`. Pure upstream-conformant advantage, policy, value and KL mathematics
live under `algorithms`; trusted reward protocols live under `rewards`. Public
methods remain on the facade so internal extraction does not change user
imports.

RL trajectories are schema v3 records with prompt-group, sample, seed,
generated-span and policy identity. Tool/environment content remains context;
only model-generated positions enter behavior log-probability and loss masks.
The trainer publishes complete groups before scoring or updating them.

## State and transaction boundaries

Construction reserves a run directory and either finishes a valid starting
manifest or removes/marks failed construction. `train`, `evaluate`, checkpoint
load/save and close acquire explicit lifecycle ownership. A transactional
checkpoint publishes a complete manifest only after all tensor and state files
are durable. Resume verifies identity and restores the policy/parameter
versions, cursor, optimizer and RNG state before another rollout. Attaching an
evaluation writes a derived artifact without rewriting original run
provenance.

## Import boundary

Base-install modules, CLI help, schemas, config compilation, plan inspection,
qualification validation and artifact verification must not import Torch.
Torch is allowed inside trainer/model/loss execution paths and GPU workload
drivers, and is imported lazily by commands that require `[train]`.

## Adding a profile

Do not modify an existing profile or digest. Add a new typed source model,
field rules, registry identity, native compiler version, loss-conformance
version and export version. Prove every accepted field has the claimed native
effect, add pinned upstream scalar/gradient/optimizer conformance, exercise
plan/cache/checkpoint/export identity mismatch failures, then complete the
[upstream lifecycle](upstream-support-policy.md) and GPU qualification.

For an RL compiler field, choose one of `exact`, `semantically_conformant`,
`locally_lowered`, `informational_only`, `distributed_only`,
`not_implemented` or `unsupported`. An accepted runnable source must have no
unknown fields and must validate as a complete `RunConfig` before atomic
publication. Keep local physical controls under `miniverl.*` where practical.

## Validation entry points

```bash
pytest -q -m "not gpu and not network"
pytest -q -m network
pytest -q -m gpu
python scripts/release_gate.py --qualification path/to/qualification.json
```

The first command is CPU CI. Network tests verify pinned remote resources. GPU
tests exercise local CUDA behavior. The qualification workflow additionally
uses a built wheel and the exact-SHA release contract.
