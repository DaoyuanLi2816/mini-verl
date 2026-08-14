# SmolLM2 direct-GKD developer workload

This is a maintainer-measured systems recipe, not a model-quality benchmark.
It uses the direct `forward_kl_topk` profile with pinned Apache-2.0
SmolLM2-360M-Instruct actor and SmolLM2-1.7B-Instruct teacher snapshots.

| Contract | Measured value |
| --- | --- |
| GPU | 1× RTX 4080, 15.992 GiB |
| Data | 64 distinct prompts available; 32 consumed |
| Bounds | 128 prompt tokens; 64 response tokens |
| Schedule | 8 current-policy updates; interrupt after update 4 |
| Objective | direct GKD `forward_kl_topk`, k=32, token mean |
| Peak allocated / reserved | 1.4254 / 1.4961 GiB |
| First update | 23.1082 s including 8.4132 s construction |
| Median rollout / score / update | 16.3067 / 0.1941 / 1.1270 s |
| Median throughput | 13.7173 rollout tok/s; 1190.38 scored positions/s; 186.9733 update positions/s |

The interrupted execution resumed in 3.1039 seconds. Its trajectories,
adapter and optimizer tensors were byte-identical to the uninterrupted run;
all training-state fields matched except the intentionally run-specific
resolved-config digest. The standard PEFT adapter reloaded successfully.

The export was then materialized with both exact base snapshots. The pinned
upstream config parse, Parquet check, PEFT load, tokenizer identity and
sequential CPU model-load/forward smoke passed, producing `launchable: true`.
That status describes a complete local artifact bundle; distributed verl
execution remains **not tested**.

The machine-readable record is
[`rtx4080-smollm2-opd-developer-v1.json`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/rtx4080-smollm2-opd-developer-v1.json),
and the reproducible driver is
[`run_smollm2_opd_reference_workload.py`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/scripts/run_smollm2_opd_reference_workload.py).
The earlier one-update compatibility smoke remains immutable and is no longer
the strongest evidence for this pair.

## WSL2 check

The same physical RTX 4080 also completed a separate Ubuntu 26.04 WSL2 path:
plan, measured zero-update probe, one 64-token-bounded rollout, teacher
scoring, one update and standard PEFT export/reload. Peak reserved VRAM was
1.4043 GiB and time to first update was 9.5106 seconds. See the checksummed
[`wsl2-rtx4080-smollm2-opd-v1.json`](evidence/wsl2-rtx4080-smollm2-opd-v1.json).

Neither record evaluates task quality, alignment, preference or safety, and
neither demonstrates distributed verl execution.
