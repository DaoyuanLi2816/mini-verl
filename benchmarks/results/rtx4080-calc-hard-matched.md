# Benchmark `gpu-calc-hard-matched`

Cold start, then equal-step continuation under supervised fine-tuning and on-policy distillation, on the chained calculator split with Qwen3-0.6B distilled from Qwen3-1.7B on one 16 GB consumer GPU.

- miniVERL 0.1.0 | git `d40942eb41c67929db52c1a037fd28890f12b281`
- created 2026-07-27T10:55:53+00:00
- hardware: NVIDIA GeForce RTX 4080 (15.992 GiB), driver None
- seeds: [1234]  **single seed -- no significance claimed**

## Results

| arm | mode | loss mode | steps | success | avg turns | invalid calls | gen tok/task | selected tokens | cache | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cold-start-only | sft | bucketed_topk_tail | 0 | 62.5% | 2.00 | 0.0% | 42.0 | 0 | - | 89.4 |
| sft-continued | sft | bucketed_topk_tail | 12 | 100.0% | 2.38 | 0.0% | 50.2 | 355 | - | 450.4 |
| opd-bucketed-k64 | opd | bucketed_topk_tail | 12 | 0.0% | 2.00 | 83.3% | 38.5 | 204 | 0.19 MiB / 684x | 684.8 |
| opd-privileged-context | opd | bucketed_topk_tail | 12 | 0.0% | 1.00 | 0.0% | 20.8 | 106 | 0.15 MiB / 666x | 546.4 |

## Aggregate

| arm | seeds | mean success | min | max |
| --- | --- | --- | --- | --- |
| cold-start-only | 1 | 62.5% | 62.5% | 62.5% |
| sft-continued | 1 | 100.0% | 100.0% | 100.0% |
| opd-bucketed-k64 | 1 | 0.0% | 0.0% | 0.0% |
| opd-privileged-context | 1 | 0.0% | 0.0% | 0.0% |

## What was held constant

```json
{
  "arms_differ_only_in": {
    "cold-start-only": {
      "environment": {
        "difficulty": "hard",
        "test_tasks": 24
      },
      "eval": {
        "enabled": false,
        "split": "test",
        "tasks": 24
      },
      "run": {
        "mode": "sft"
      },
      "train": {
        "cycles": 0,
        "sft_warmup_cycles": 0
      }
    },
    "opd-bucketed-k64": {
      "environment": {
        "difficulty": "hard",
        "test_tasks": 24
      },
      "eval": {
        "enabled": false,
        "split": "test",
        "tasks": 24
      },
      "loss": {
        "divergence": "reverse_kl",
        "mode": "bucketed_topk_tail",
        "top_k": 64
      },
      "models": {
        "teacher": {
          "mode": "standard"
        }
      },
      "run": {
        "mode": "opd"
      },
      "train": {
        "cycles": 12,
        "learning_rate": 5e-05,
        "lr_schedule": "constant",
        "sft_warmup_cycles": 0
      }
    },
    "opd-privileged-context": {
      "environment": {
        "difficulty": "hard",
        "test_tasks": 24
      },
      "eval": {
        "enabled": false,
        "split": "test",
        "tasks": 24
      },
      "loss": {
        "divergence": "reverse_kl",
        "mode": "bucketed_topk_tail",
        "top_k": 64
      },
      "models": {
        "teacher": {
          "mode": "privileged_context"
        }
      },
      "run": {
        "mode": "opd"
      },
      "train": {
        "cycles": 12,
        "learning_rate": 5e-05,
        "lr_schedule": "constant",
        "sft_warmup_cycles": 0
      }
    },
    "sft-continued": {
      "environment": {
        "difficulty": "hard",
        "test_tasks": 24
      },
      "eval": {
        "enabled": false,
        "split": "test",
        "tasks": 24
      },
      "run": {
        "mode": "sft"
      },
      "train": {
        "cycles": 12,
        "learning_rate": 5e-05,
        "lr_schedule": "constant",
        "sft_warmup_cycles": 0
      }
    }
  },
  "cold_start_cycles": 12,
  "cold_start_mode": "sft",
  "difficulty": "medium",
  "effective_batch_trajectories": 6,
  "environment": "calculator",
  "eval_seed": 0,
  "eval_split": "test",
  "eval_tasks": 48,
  "eval_temperature": 0.0,
  "learning_rate": 0.0001,
  "lr_schedule": "cosine",
  "max_trajectory_tokens": 704,
  "max_turns": 3,
  "optimizer": "adamw8bit",
  "rollouts_per_cycle": 6,
  "seeds": [
    1234
  ],
  "shared_initial_checkpoint": true,
  "split_seed": 7,
  "test_tasks": 48,
  "train_tasks": 256
}
```

Arms differ **only** in the override keys listed above. Student generated tokens,
selected training tokens, teacher query ratio and wall clock cannot be matched by
construction, so they are measured and reported per arm rather than equalized.

## Notes

Single seed on one RTX 4080 (16 GB). No statistical significance is claimed. opd-bucketed-k64 distils toward the raw instruct teacher, which has never seen the tool protocol; opd-privileged-context gives that same teacher the verified answer.
