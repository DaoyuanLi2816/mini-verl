# Legacy environment/PPO bridge

This page documents the older `single-gpu-online-distillation-v1` artifact
bridge. It is retained for migration and historical reproducibility; it is not
miniVERL's current verl-style OPD runtime.

The legacy bridge converts validated environment data to Parquet and emits a
PPO/reward integration scaffold for official verl `v0.8.0` at `7aed6b23`.
Because teacher identity, reward semantics, base snapshots and user mappings
are not fully determined, a new legacy bundle stays fail-closed:

```text
reward_implementation_complete: false
launchable: false
distributed_execution_tested: false
algorithm_semantic_parity: false
```

`miniVERL-defined compatibility Level 3` refers only to bundle structure,
hashes and pinned parse/load smoke. It is not generic verl compatibility and
does not mean a Ray/FSDP/vLLM job ran. The reward scaffold is untrusted input;
doctor statically inspects it by default and executes it only after an explicit
trust opt-in.

Existing legacy commands and artifacts remain readable. New users should use
the [current local OPD runtime](verl-opd-runtime.md) and [current scale-out
contract](verl-opd-scaleout.md), which preserve pure OPD intent and do not
generate a reward scaffold.

For the complete historical security and artifact checks, see the archived
[combined bridge reference](verl-bridge.md).
