# PROJECT_STATE

Current maintainer handoff for **miniVERL** (`mini-verl` package, `miniverl`
CLI). `release-state.yaml` is the canonical version source; this page indexes
current product and evidence state rather than repeating release history.

Last updated: 2026-09-09.

Canonical release state: stable `v0.14.0` (`60afc63da1257e3e22aa6056607a63a90d4d90f1`), development `0.14.1.dev0`.

## Release state

- v0.14.0 productization is published: upstream configuration coverage,
  packaged PPO/GRPO walkthroughs, strict run inspection and system-level recovery.
- Upstream drift check: GitHub's latest stable release is still `v0.9.0`
  (published 2026-08-14); existing compiler targets remain appropriate.
- Implemented: v3 prompt-minibatch units and explicit physical lowerings;
  six upstream scripts/seven resolved cases with complete field accounting;
  installed PPO/GRPO templates and adaptation ledgers; strict run inspection;
  PPO value/behavior identity guards and checkpoint-bound log recovery.
- Exact-release qualification run `34318236880` passed on attempt 1, followed
  by non-publishing gate `34325442958` and OIDC release `34328973021`.
  PPO 4 actor/4 critic updates and GRPO 2 actor updates passed exact tensor
  replay and inspected handoff from the installed candidate wheel.
- Local regression: 2,656 non-GPU/non-network tests passed at 83% branch
  coverage; 10 RTX 4080 GPU and 16 network tests passed. Four pinned upstream
  conformance tests and four-viewport documentation browser checks passed.

- Stable release: `v0.14.0` at
  `60afc63da1257e3e22aa6056607a63a90d4d90f1`.
- Current development line: `0.14.1.dev0`.
- Stable docs: <https://daoyuanli2816.github.io/mini-verl/>.
- Development docs: <https://daoyuanli2816.github.io/mini-verl/dev/>.
- Historical build log: [v0.1-v0.9 archive](docs/history/project-state-v0.1-v0.9.md).

## Current product boundary

Development `0.14.1.dev0` has a closed typed profile compiler: it retains v1/v2 and adds
`verl-rl-v0.9-single-gpu-v3`, a typed resolved RL subset of official verl
`v0.9.0` at commit `483b8a009ba3a97563edee3a19887e4862b8094a`.
It compiles PPO/GAE into phased one-GPU actor, critic, reference and reward
roles. The critic is independently trainable and owns its optimizer, schedule,
checkpoint state and exact-resume identity. The same release line connects
actor KL, entropy regularization and a pinned Hugging Face sequence-classifier
reward provider.

The established `verl-opd-v0.8-single-gpu-v1` family remains the teacher-based
distillation path. It targets official verl `v0.8.0` at
`7aed6b230776f963fa09509c10d9c3a767d1102c` and provides direct GKD,
sampled-k1, grouped and rewarded profiles.

The stable v0.9.1 repair makes dtype, quantization, attention, student
adapter input, logical/physical batch limits and placement legality explicit.
Executable compatibility claims are mutation-tested and recorded in
`docs/generated/verl-opd-v0.8-field-effects.json`. Quantized roles cannot swap;
unknown-size quantized roles require proof instead of receiving an executable
plan.

Profile-scoped recipes, plans, caches, checkpoints and exports bind an
independent identity. The v0.9 compiler is Torch-free; the RL math and runtime
remain inside `[train]`. The new estimator, policy, value and KL primitives are
compared against the exact upstream source. The direct-GKD and sampled-k1
profiles retain their pinned conformance and measured RTX 4080 paths.
Portable hardware records preserve measured, estimated and unknown states;
schema-valid community submissions remain unreviewed until maintainer validation.
The release chain builds one hosted-runner candidate and binds the maintainer's
RTX 4080 qualification to that exact manifest and wheel. Release jobs may
publish only the same verified bytes and never rebuild them. Future GitHub
Releases use a canonical, explicit evidence layout with four principal JSON
records and one deterministic subordinate-evidence archive; historical v0.10.1
asset names remain immutable.

The v0.9 export path preserves actor, critic, Parquet, recipe and provenance
artifacts in a checksummed bundle. Its launch template stays inactive because
miniVERL does not translate local optimizer state into upstream distributed
checkpoints or run Ray/FSDP. Arbitrary unresolved verl YAML and unknown
algorithm-changing fields remain outside the versioned compiler. The legacy
v0.8 environment/PPO artifact bridge is migration-only.

## Current evidence

| Evidence | Status |
| --- | --- |
| Installed PPO product workflow | Qwen3-0.6B, 8 trajectories, 4 actor/4 critic updates, 2.7266 GiB peak reserved, 15.688 s reported train duration; exact tensor replay and handoff passed |
| Installed GRPO product workflow | Qwen3-0.6B, 8 trajectories, 2 actor updates, 1.502 GiB peak reserved, 8.923 s reported train duration; exact tensor replay and handoff passed |
| Qwen3-0.6B/1.7B developer workload | 32 prompts, 8 current-policy updates, 3.1914 GiB peak reserved on one RTX 4080; matched interruption/resume was byte-identical |
| Qwen3 sampled-k1 PG | 32 prompts, 8 updates, exact interruption/resume, 3.1914 GiB peak reserved; no quality comparison |
| SmolLM2-360M/1.7B direct GKD | 32 prompts, 8 updates, 1.4961 GiB peak reserved; exact resume, PEFT reload and materialized export passed |
| Ubuntu 26.04 WSL2 | plan, bounded probe, rollout, teacher scoring, one update and PEFT reload measured on the same RTX 4080 |
| external alignment v1 | preregistered early stop: 0 selected checkpoints, teachers, continuation arms or final-test accesses |
| distributed verl execution | not tested |

Frozen benchmark and task-level result JSON/JSONL remain immutable. The
calculator schema-v2 source stays at SHA-256
`53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`.

## Maintainer entry points

- [Installed PPO/GRPO workflow](docs/local-rl-workflow.md)
- [Single-GPU verl RL](docs/verl-rl-runtime.md)
- [Current scale-out contract](docs/verl-rl-runtime.md#scale-out-handoff)
- [Compatibility policy](docs/compatibility.md)
- [Measured workload](docs/verl-opd-reference-workload.md)
- [Limitations](docs/limitations.md)
- Active research roadmap: [issue #39](https://github.com/DaoyuanLi2816/mini-verl/issues/39).
  Additional compiler lowerings are indexed by the [upstream corpus](docs/verl-compatibility-corpus.md).
