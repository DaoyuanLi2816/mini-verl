# Alignment Card

Method: `continued_sft`

Seed: `20260801`

Starting SFT checkpoint: `7304922281268a687dd1c75ba918e26c64c8207b5701db78c368afd20d80ae89`

Policy: `miniverl-tool-policy@v1`

## Measured endpoints

```json
{
  "alignment_score": 0.8333333333333334,
  "appropriate_refusal_rate": 1.0,
  "benign_compliance_rate": 0.75,
  "decision_distribution_shift_jsd": null,
  "general_utility_retention": 0.75,
  "harmful_compliance_rate": 0.0,
  "instruction_retention": 1.0,
  "over_refusal_rate": 0.0,
  "preference_win_rate": 0.8333333333333334,
  "tasks": 48,
  "teacher_queried_positions": null,
  "teacher_query_ratio": null,
  "tool_utility_retention": 0.6666666666666666
}
```

## Cost

```json
{
  "gpu_seconds": 2.452,
  "optimizer_updates": 4,
  "peak_vram_bytes": 1031345152,
  "wall_seconds": 132.714321
}
```

DPO cost includes the pinned external TRL training job when applicable. Evaluation time is not included in `gpu_seconds`.

## Limitations

- One model family, one deterministic sandbox policy suite and one measured GPU.
- The common SFT checkpoint already saturated every measured policy and utility endpoint.
- Three seeds describe observed variation and do not establish a population claim.
- External IFEval, XSTest, HarmBench and RewardBench endpoints were not measured in this artifact.

Source result SHA-256: `584752dccb91654109c357b8ebb12681a12a9c1476a9ba539dd35e4d860a22ef`

Card content SHA-256: `18753c21ffb6e0a135ea3c324591cef8577f50bd7e7b646d20fded6959534e22`
