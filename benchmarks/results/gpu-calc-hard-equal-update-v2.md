# Benchmark `gpu-calc-hard-equal-update-v2`

Cold start, then equal-optimizer-update continuation under supervised fine-tuning and on-policy distillation from raw, privileged-answer and protocol-SFT Qwen3-1.7B teachers on the chained calculator split.

- schema v2 | miniVERL 0.2.0 | git `3dafd5fb3c8029e762139d27e9c311ae6eaa6340`
- created 2026-07-28T06:27:58+00:00
- budget axis: `optimizer_steps`
- hardware: NVIDIA GeForce RTX 4080 (15.992 GiB), driver 596.49
- seeds: [1234, 20260727]

## Results

| arm | objective | steps | success | selected positions | teacher queried | train s | eval s | peak allocated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cold-start-only | sft_cross_entropy | 0 | 75.0% | 0 | n/a | 0.1 | 96.2 | 1093597696 |
| sft-continued | sft_cross_entropy | 12 | 100.0% | 4032 | n/a | 80.5 | 93.1 | 1465412096 |
| opd-raw-teacher | on_policy_distillation | 12 | 0.0% | 3047 | 3047 | 468.6 | 32.1 | 4909056000 |
| opd-privileged-context | on_policy_distillation | 12 | 0.0% | 3149 | 3149 | 563.2 | 81.8 | 4912181760 |
| opd-protocol-sft-teacher | on_policy_distillation | 12 | 100.0% | 3922 | 3922 | 523.7 | 91.2 | 4987733504 |
| cold-start-only | sft_cross_entropy | 0 | 75.0% | 0 | n/a | 0.1 | 105.9 | 1335433216 |
| sft-continued | sft_cross_entropy | 12 | 100.0% | 3950 | n/a | 92.4 | 108.9 | 1485084160 |
| opd-raw-teacher | on_policy_distillation | 12 | 0.0% | 2987 | 2987 | 419.3 | 39.0 | 4985019904 |
| opd-privileged-context | on_policy_distillation | 12 | 0.0% | 2900 | 2900 | 499.8 | 42.2 | 4986920448 |
| opd-protocol-sft-teacher | on_policy_distillation | 12 | 100.0% | 4050 | 4050 | 523.9 | 107.9 | 5061393920 |

## Resolved controls

The complete common declared configuration and separate scientific, runtime-resolution and harness-only diffs are stored in the JSON artifact. Undeclared scientific differences are rejected before any model is loaded.

## Notes

RTX 4080 16 GB, driver 596.49; prespecified v0.2 five-arm two-seed run; protocol teacher candidate A passed the 50% gate at 100% strict success. One earlier incomplete attempt was preserved as an aborted harness-bug run before this complete rerun.
