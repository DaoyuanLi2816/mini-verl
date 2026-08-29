# Rollout Runtime v2: baseline and selected-backend evidence

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

## First development candidate (preserved negative result)

The exact `0.11.0.dev0` wheel at source `690db9b079bfecfec14dc3bedc3aa0308cbacf60`
ran the same 24 cells with `hf_cached` and managed vLLM 0.28.0. Both runs used
wheel SHA-256 `266fcd59bb3e85b02974a92b26c9a4e59c7c9b9205853d17baadcfffdb898e27`
and the baseline software stack: Python 3.12.13, Torch 2.13.0+cu130,
Transformers 5.14.1, PEFT 0.20.0 and bitsandbytes 0.50.2.

| Response | Sampling | `hf_cached` / reference | vLLM / `hf_cached` | vLLM tokens/s |
| ---: | --- | ---: | ---: | ---: |
| 64 | greedy | 0.65–0.76× | 1.73–1.80× | 112.7–114.9 |
| 64 | seeded stochastic | 1.83–1.87× | 1.94–2.06× | 109.3–114.2 |
| 256 | greedy | 0.71–0.85× | 1.71–1.77× | 111.8–117.4 |
| 256 | seeded stochastic | 0.84–1.85× | 2.04–4.55× | 113.1–114.5 |
| 512 | greedy | 0.67–1.05× | 1.65–1.87× | 114.8–117.7 |
| 512 | seeded stochastic | 1.84–1.90× | 1.90–2.05× | 108.5–114.4 |

Ranges contain both prompt lengths and both `n` values. One `hf_cached`
p512/r256/n4 stochastic cell had measured runs of 118.807, 267.930 and
162.641 seconds; the median and resulting 0.84× speedup remain in the result.

### Gate outcome

- `hf_cached >= 2× hf_reference` at response 256 and 512: **failed**. The
  conservative preregistration interpretation requires every paired cell to
  reach the threshold.
- vLLM `>= 1.2× hf_cached` at response 256 and 512: **passed** in every cell;
  the observed range was 1.65–4.55×.
- Peak total GPU memory below 14.5 GiB: **passed**. `hf_cached` peaked at
  4,598 MiB and vLLM at 11,820 MiB.
- Eight vLLM refreshes: **passed**. Every sync and identity was unique, with no
  monotonic memory growth; the localhost port closed at teardown.
- vLLM PG-k1 log-probability conformance: **failed**. Greedy token agreement
  was 32/32, while maximum absolute log-probability difference was 0.0194
  against the 0.01 NF4 threshold. Direct GKD remains the supported scope.

This candidate failed the local-backend speed gate and triggered the profiling
work that produced the qualified candidate below. Its bytes and failed outcome
remain part of the record.

## Candidate artifact integrity

| Artifact | SHA-256 |
| --- | --- |
| `hf_cached` raw | `a9f8f0b0275d940f30497a0d88d76da0e112ffbe2bc1b37a42c6ed313b852242` |
| vLLM raw | `abf14d73a7289f9619515943b4c8d7dafe7cd187fefe928655739ac7ed1ab16c` |
| Aggregate result | `32be86856263ccdf787986c4ec54570d323af8c238dc1664e00a9ce41ca393c4` |
| Frozen calculator | `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc` |

The [aggregate JSON](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/rollout-runtime-v2-rtx4080.json)
contains every cell, exact raw hashes, gate calculation and backend-selection
state. `scripts/publish_rollout_runtime_v2_evidence.py --check` reproduces it.

## Qualified v0.11 candidate

The exact `0.11.0.dev0` wheel at source
`3cfa09b0b7aa2ee63f5702a5766b73d9a32fa8b9` reran the same 24 cells. Both
backends used wheel SHA-256
`33c30834dcd01001a5c2ae619a1c982ceb2ad5fcc457998baa9f960a87267eba`.
The local backend keeps the NF4 actor as the training source of truth, builds a
BF16 inference mirror, and synchronizes the exact live LoRA tensors before each
policy version. The managed vLLM backend uses CUDA Graph execution.

| Response | `hf_cached` / reference | vLLM / `hf_cached` | `hf_cached` tokens/s | vLLM tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 2.39–4.30× | 2.99–5.71× | 124.2–242.5 | 626.6–806.0 |
| 256 | 2.38–4.30× | 3.19–5.97× | 125.9–240.8 | 685.6–836.4 |
| 512 | 2.54–4.23× | 3.08–5.88× | 124.3–248.7 | 693.4–821.2 |

Ranges include both prompt lengths, both `n` values and both sampling modes.
Each cell used one warmup and three measured repetitions. Compilation and
engine warmup happen before measured repetitions and remain visible in the raw
artifact.

### Gate outcome

- `hf_cached >= 2× hf_reference` at response 256 and 512: **passed in every
  cell**, with conservative minima of 2.375× and 2.541×.
- vLLM `>= 1.2× hf_cached` at response 256 and 512: **passed in every cell**;
  the full observed range was 3.076–5.966×.
- Peak total GPU memory below 14.5 GiB: **passed**. `hf_cached` peaked at
  6,504 MiB and vLLM at 11,931 MiB.
- Exact policy refresh and teardown: **passed** for both backends across eight
  unique policy identities, without monotonic memory growth.
- Local NF4/mirror conformance: **passed** with exact greedy tokens and maximum
  sampled-token log-probability difference 0.002483 against the 0.01 limit.
- vLLM PG-k1 log-probability conformance: **failed closed**. Greedy tokens were
  exact, but the maximum difference was 0.019646. vLLM is therefore selected
  for direct GKD only; PG-k1 stays on `hf_cached`.

The runtime performance gate is complete. Release publication still requires
the exact-wheel qualification and release workflow.

## Qualified-candidate artifact integrity

| Artifact | SHA-256 |
| --- | --- |
| `hf_cached` raw | `dd29402642ab68f7854579bd68accb4b449ca0a2ec4cb8b90d834ee2e6f03757` |
| vLLM CUDA Graph raw | `00c55a25deb9ecf7786d7f54ccba4c5c5b0bcce4bf1e46fe8ac6b48d0596194c` |
| Aggregate result | `0fb2066a8b567adbb011113141b2b4e3a4fa281cf3f3d13f6eecd1d7b0499a5d` |
| Frozen calculator | `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc` |

The [qualified aggregate JSON](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/rollout-runtime-v2-v0.11.0-candidate-rtx4080.json)
binds every cell to the exact wheel, raw files and preregistration.
`scripts/publish_rollout_runtime_v2_candidate.py --check` reproduces it.
