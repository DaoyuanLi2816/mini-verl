# Maintainer architecture

This page maps the change boundaries maintainers use for review.

## Config to runtime

`bridge.profiles` selects one closed profile and dispatches to its typed source
compiler. The compiler classifies every source field and produces a resolved
compatibility plan. `bridge.opd_runtime` derives legal one-device placement and
the native `RunConfig`; `bridge.opd_plan` binds data files, revisions, profile
identity and native config into an immutable plan. `OPDTrainer.from_config`
then constructs the local runtime. No generic dynamic profile loader exists.

The profile identity enters immutable plans and is copied into run manifests,
teacher-cache metadata, checkpoint identity and export/materialization reports.
Any reader that combines those artifacts compares the complete identity before
using their contents.

## Logical roles and physical placement

Actor, teacher and optional reference are distinct logical identities. Rollout
generates with the current actor policy; teacher scoring observes only the
visited positions; update consumes provenance-bound targets. Physical
placement may be resident phased models, an allowed unquantized swap, or a
shared backbone with separate adapters. Quantized swap remains illegal.

`training.trainer.OPDTrainer` is the compatibility facade and state-machine
owner. Batching, memory planning, optimizer construction, checkpoint I/O and
offline-dataset persistence live in dedicated `training` modules; model and
tokenizer construction live under `models`; cache transactions live under
`cache`. Public methods remain on the facade so internal extraction does not
change user imports.

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
