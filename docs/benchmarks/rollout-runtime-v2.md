# Rollout Runtime v2: frozen HF reference baseline

The pre-v0.11 Hugging Face path is now frozen as the compatibility and
performance baseline for Rollout Runtime v2. All 24 preregistered cells
completed on one WSL2 RTX 4080 without an OOM or discarded cell.

This result measures actor rollout only. It is not a task-quality experiment,
does not include teacher scoring or an optimizer update, and does not yet
exercise typed grouped-sample semantics.

## Frozen setup

| Item | Exact value |
| --- | --- |
| Source | `0d6e0070ae73ef35f718aec3624ee5263ac96e3a` |
| Candidate wheel | `0256bd9e63ca6ed52999a5073a3577a008581a8d4418062314750d86f21cd5fe` |
| Actor | `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca` |
| Weight payload digest | `b29cd98b83f9bddc7ec8943be5f142243e956f448f13456847849e5b8615b413` |
| Runtime | Python 3.12.13, Torch 2.13.0+cu130, Transformers 5.14.1, NF4/BF16 |
| GPU / driver | NVIDIA GeForce RTX 4080 16 GiB / 596.49 |
| Workload | 4 logical prompts; prompt 128/512; response 64/256/512; `n=1/4`; greedy and seeded stochastic |

## Measured rollout throughput

Every row generated the full configured response bound. Rates use actual
generated tokens and median wall time over three measured repetitions after
one warmup.

| Prompt | Response | Samples/prompt | Greedy tokens/s | Seeded stochastic tokens/s | Peak reserved GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 64 | 1 | 95.42 | 29.93 | 1.053 |
| 128 | 64 | 4 | 97.07 | 29.95 | 1.053 |
| 128 | 256 | 1 | 95.87 | 29.95 | 1.084 |
| 128 | 256 | 4 | 92.35 | 30.07 | 1.084 |
| 128 | 512 | 1 | 93.16 | 30.23 | 1.086 |
| 128 | 512 | 4 | 92.91 | 30.30 | 1.086 |
| 512 | 64 | 1 | 92.68 | 30.28 | 1.086 |
| 512 | 64 | 4 | 83.81 | 30.21 | 1.086 |
| 512 | 256 | 1 | 80.10 | 30.31 | 1.086 |
| 512 | 256 | 4 | 77.24 | 30.06 | 1.086 |
| 512 | 512 | 1 | 66.28 | 29.92 | 1.086 |
| 512 | 512 | 4 | 66.26 | 30.31 | 1.086 |

The two sampling paths have different pre-v0.11 implementations. Greedy uses
one padded batch but recomputes the full prefix at each step; seeded stochastic
preserves per-request RNG by running the established sequential KV-cache path.
These measurements are therefore separate baselines, not an algorithm ranking.

## What remains unmeasured

The legacy public backend does not expose clean prefill, decode or time-to-first
token boundaries, so those fields are `not_measured`, not zero. Policy sync is
`not_applicable` for an actor already loaded in the same process. Teacher
scoring, actor update and full-cycle time are also `not_measured` in this
rollout-only baseline and remain required before v0.11 release.

`n=4` here is four independently repeated requests per prompt. Transactional
group identity, grouped resume and reward composition do not exist in this
baseline and cannot be inferred from it.

## Artifact integrity

| Artifact | SHA-256 |
| --- | --- |
| Preregistration | `8cc3ba738c69b59ed19c22c1de874fd00249404198a3e05983477dc8899bb7e5` |
| Raw measurement | `2e303eabb559b843d25377a7c72e0aeb0219eec7eb8bc109c0419839b3251170` |
| Final identity-bound result | `b25daee7ee726b7a7be18d7dbe26590fb61325bf212cfd9e7110b69f5fa8889c` |
| Frozen calculator benchmark | `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc` |

See the [method](rollout-runtime-v2-method.md) for timing and invalidation
rules. The next backend must use this workload unchanged and pass the
preregistered correctness, memory and 256/512-token speedup gates.
