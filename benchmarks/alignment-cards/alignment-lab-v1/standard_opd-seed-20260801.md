# Alignment Card

Method: `standard_opd`

Seed: `20260801`

Starting SFT checkpoint: `7304922281268a687dd1c75ba918e26c64c8207b5701db78c368afd20d80ae89`

Policy: `miniverl-tool-policy@v1`

## Measured endpoints

```json
{
  "alignment_score": 0.9583333333333334,
  "appropriate_refusal_rate": 1.0,
  "benign_compliance_rate": 0.9375,
  "decision_distribution_shift_jsd": null,
  "general_utility_retention": 0.9375,
  "harmful_compliance_rate": 0.0,
  "instruction_retention": 1.0,
  "over_refusal_rate": 0.0,
  "preference_win_rate": 0.9583333333333334,
  "tasks": 48,
  "teacher_queried_positions": 351,
  "teacher_query_ratio": 1.0,
  "tool_utility_retention": 0.9166666666666666
}
```

## Cost

```json
{
  "gpu_seconds": 43.571,
  "optimizer_updates": 4,
  "peak_vram_bytes": 1010267648,
  "wall_seconds": 172.661803
}
```

DPO cost includes the pinned external TRL training job when applicable. Evaluation time is not included in `gpu_seconds`.

## Limitations

- One model family, one deterministic sandbox policy suite and one measured GPU.
- The common SFT checkpoint already saturated every measured policy and utility endpoint.
- Three seeds describe observed variation and do not establish a population claim.
- External IFEval, XSTest, HarmBench and RewardBench endpoints were not measured in this artifact.

Source result SHA-256: `584752dccb91654109c357b8ebb12681a12a9c1476a9ba539dd35e4d860a22ef`

Card content SHA-256: `881f8e21848e82e810385247d7647a5d7b07ee1c4791a48b338bc01797df81c7`
