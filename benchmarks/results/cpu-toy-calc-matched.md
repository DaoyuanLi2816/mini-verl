# Benchmark `calc-matched-budget`

Cold start, then equal-step continuation under SFT, offline KD, bucketed OPD and exact-full-vocabulary OPD on the deterministic calculator environment. This is a PARITY check, not a ranking: the toy models can only solve the easy split, where supervised fine-tuning saturates, so the accuracy differences between arms are within noise. Its job is to show that all seven arms run to completion under identical budgets. Capability numbers come from benchmarks/configs/gpu_calc_hard.yaml.

- miniVERL 0.1.0 | git `f2f89daeb99e938feba32dca7efcc085d7683e90`
- created 2026-07-27T11:13:35+00:00
- hardware: NVIDIA GeForce RTX 4080 (15.992 GiB), driver None
- seeds: [1234, 20260727]

## Results

| arm | mode | loss mode | steps | success | avg turns | invalid calls | gen tok/task | selected tokens | cache | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cold-start-only | sft | bucketed_topk_tail | 0 | 66.7% | 2.00 | 0.0% | 42.9 | 0 | - | 5.8 |
| sft-continued | sft | bucketed_topk_tail | 60 | 83.3% | 2.00 | 0.0% | 42.9 | 341 | - | 14.5 |
| offline-kd | offline_kd | bucketed_topk_tail | 60 | 62.5% | 2.04 | 11.5% | 44.7 | 344 | 0.06 MiB / 2x | 112.7 |
| opd-bucketed-k16 | opd | bucketed_topk_tail | 60 | 79.2% | 2.00 | 0.0% | 42.9 | 341 | 0.12 MiB / 2x | 107.0 |
| opd-exact | opd | exact_full_vocab | 60 | 79.2% | 2.00 | 0.0% | 42.8 | 341 | 0.00 MiB / 0x | 106.3 |
| opd-bucketed-forward-kl | opd | bucketed_topk_tail | 60 | 79.2% | 2.00 | 0.0% | 42.8 | 341 | 0.12 MiB / 2x | 107.0 |
| opd-tool-and-final | opd | bucketed_topk_tail | 60 | 79.2% | 2.00 | 0.0% | 42.8 | 344 | 0.12 MiB / 2x | 113.4 |
| cold-start-only | sft | bucketed_topk_tail | 0 | 8.3% | 2.04 | 11.1% | 42.8 | 0 | - | 3.0 |
| sft-continued | sft | bucketed_topk_tail | 60 | 8.3% | 2.04 | 11.1% | 42.9 | 342 | - | 8.8 |
| offline-kd | offline_kd | bucketed_topk_tail | 60 | 4.2% | 2.04 | 12.0% | 45.2 | 343 | 0.06 MiB / 2x | 85.3 |
| opd-bucketed-k16 | opd | bucketed_topk_tail | 60 | 8.3% | 2.00 | 4.0% | 43.0 | 343 | 0.12 MiB / 2x | 116.9 |
| opd-exact | opd | exact_full_vocab | 60 | 4.2% | 2.00 | 0.0% | 42.8 | 341 | 0.00 MiB / 0x | 117.5 |
| opd-bucketed-forward-kl | opd | bucketed_topk_tail | 60 | 8.3% | 2.04 | 11.1% | 43.2 | 343 | 0.12 MiB / 2x | 113.3 |
| opd-tool-and-final | opd | bucketed_topk_tail | 60 | 4.2% | 2.04 | 11.1% | 43.0 | 342 | 0.12 MiB / 2x | 134.2 |

## Aggregate

| arm | seeds | mean success | min | max |
| --- | --- | --- | --- | --- |
| cold-start-only | 2 | 37.5% | 8.3% | 66.7% |
| sft-continued | 2 | 45.8% | 8.3% | 83.3% |
| offline-kd | 2 | 33.3% | 4.2% | 62.5% |
| opd-bucketed-k16 | 2 | 43.8% | 8.3% | 79.2% |
| opd-exact | 2 | 41.7% | 4.2% | 79.2% |
| opd-bucketed-forward-kl | 2 | 43.8% | 8.3% | 79.2% |
| opd-tool-and-final | 2 | 41.7% | 4.2% | 79.2% |

## What was held constant

```json
{
  "arms_differ_only_in": {
    "cold-start-only": {
      "run": {
        "mode": "sft"
      },
      "train": {
        "cycles": 0,
        "sft_warmup_cycles": 0
      }
    },
    "offline-kd": {
      "cache": {
        "reuse_across_policy_versions": true,
        "strict_policy_version": false
      },
      "run": {
        "mode": "offline_kd"
      },
      "train": {
        "cycles": 60,
        "learning_rate": 0.0005,
        "lr_schedule": "constant",
        "sft_warmup_cycles": 0
      }
    },
    "opd-bucketed-forward-kl": {
      "loss": {
        "divergence": "forward_kl",
        "mode": "bucketed_topk_tail",
        "top_k": 16
      },
      "run": {
        "mode": "opd"
      },
      "train": {
        "cycles": 60,
        "learning_rate": 0.0005,
        "lr_schedule": "constant",
        "sft_warmup_cycles": 0
      }
    },
    "opd-bucketed-k16": {
      "loss": {
        "divergence": "reverse_kl",
        "mode": "bucketed_topk_tail",
        "top_k": 16
      },
      "run": {
        "mode": "opd"
      },
      "train": {
        "cycles": 60,
        "learning_rate": 0.0005,
        "lr_schedule": "constant",
        "sft_warmup_cycles": 0
      }
    },
    "opd-exact": {
      "loss": {
        "divergence": "reverse_kl",
        "mode": "exact_full_vocab",
        "top_k": 1
      },
      "run": {
        "mode": "opd"
      },
      "train": {
        "cycles": 60,
        "learning_rate": 0.0005,
        "lr_schedule": "constant",
        "sft_warmup_cycles": 0
      }
    },
    "opd-tool-and-final": {
      "loss": {
        "divergence": "reverse_kl",
        "mode": "bucketed_topk_tail",
        "top_k": 16
      },
      "run": {
        "mode": "opd"
      },
      "selection": {
        "selector": "tool_and_final"
      },
      "train": {
        "cycles": 60,
        "learning_rate": 0.0005,
        "lr_schedule": "constant",
        "sft_warmup_cycles": 0
      }
    },
    "sft-continued": {
      "run": {
        "mode": "sft"
      },
      "train": {
        "cycles": 60,
        "learning_rate": 0.0005,
        "lr_schedule": "constant",
        "sft_warmup_cycles": 0
      }
    }
  },
  "cold_start_cycles": 450,
  "cold_start_mode": "sft",
  "difficulty": "easy",
  "effective_batch_trajectories": 8,
  "environment": "calculator",
  "eval_seed": 0,
  "eval_split": "test",
  "eval_tasks": 24,
  "eval_temperature": 0.0,
  "learning_rate": 0.003,
  "lr_schedule": "cosine",
  "max_trajectory_tokens": 512,
  "max_turns": 3,
  "optimizer": "adamw",
  "rollouts_per_cycle": 8,
  "seeds": [
    1234,
    20260727
  ],
  "shared_initial_checkpoint": true,
  "split_seed": 7,
  "test_tasks": 24,
  "train_tasks": 256
}
```

Arms differ **only** in the override keys listed above. Student generated tokens,
selected training tokens, teacher query ratio and wall clock cannot be matched by
construction, so they are measured and reported per arm rather than equalized.

## Notes

CPU toy-backend PARITY check, two seeds. The toy models can only solve the easy calculator split, where supervised fine-tuning saturates, so the accuracy differences between arms are within noise and must not be read as a ranking. Wall-clock figures were taken while a GPU benchmark shared the machine.
