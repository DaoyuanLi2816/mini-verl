# PROJECT_STATE

Current maintainer handoff for **miniVERL** (`mini-verl` package, `miniverl`
CLI). `release-state.yaml` is the canonical version source; this page indexes
current product and evidence state rather than repeating release history.

Last updated: 2026-08-14.

Canonical release state: stable `v0.9.1` (`6c0f3d818c10419e0bfba81f3ad1c5adf24eaf09`), development `0.10.0.dev0`.

## Release state

- Stable: `v0.9.1` at `6c0f3d818c10419e0bfba81f3ad1c5adf24eaf09`.
- Development: `0.10.0.dev0`.
- Stable docs: <https://daoyuanli2816.github.io/mini-verl/>.
- Development docs: <https://daoyuanli2816.github.io/mini-verl/dev/>.
- Historical build log: [v0.1-v0.9 archive](docs/history/project-state-v0.1-v0.9.md).

## Current product boundary

The primary product is `verl-opd-v0.8-single-gpu-v1`: a typed, resolved subset
of official verl `v0.8.0` at commit
`7aed6b230776f963fa09509c10d9c3a767d1102c`. It runs one actor, one teacher,
`n=1`, reward-free direct GKD `forward_kl_topk` and token-mean reduction in one
local process on one NVIDIA CUDA GPU.

The stable v0.9.1 repair makes dtype, quantization, attention, student
adapter input, logical/physical batch limits and placement legality explicit.
Executable compatibility claims are mutation-tested and recorded in
`docs/generated/verl-opd-v0.8-field-effects.json`. Quantized roles cannot swap;
unknown-size quantized roles require proof instead of receiving an executable
plan.

Development `0.10.0.dev0` has a closed typed profile registry and torch-free
compatibility introspection. Profile-scoped plans, caches, checkpoints and
exports bind an independent identity. The direct-GKD and sampled-k1 vanilla
policy-loss profiles both have pinned conformance and measured RTX 4080 paths.

Arbitrary verl YAML, other policy-gradient modes, rewards, PPO/GRPO, Ray, FSDP,
Megatron, multi-GPU and distributed execution remain unsupported. The legacy
environment/PPO artifact bridge is migration-only.

## Current evidence

| Evidence | Status |
| --- | --- |
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

- [Current local runtime](docs/verl-opd-runtime.md)
- [Current scale-out contract](docs/verl-opd-scaleout.md)
- [Compatibility policy](docs/compatibility.md)
- [Measured workload](docs/verl-opd-reference-workload.md)
- [Limitations](docs/limitations.md)
- Active roadmap: [issue #39](https://github.com/DaoyuanLi2816/mini-verl/issues/39)
  and [issue #64](https://github.com/DaoyuanLi2816/mini-verl/issues/64)
