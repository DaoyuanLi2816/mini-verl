# Alignment Card

Method: `verifier_gated_opd`

Seed: `20260801`

Starting SFT checkpoint: `7304922281268a687dd1c75ba918e26c64c8207b5701db78c368afd20d80ae89`

Policy: `miniverl-tool-policy@v1`

## Measured endpoints

```json
{
  "alignment_score": 1.0,
  "appropriate_refusal_rate": 1.0,
  "benign_compliance_rate": 1.0,
  "decision_distribution_shift_jsd": null,
  "general_utility_retention": 1.0,
  "harmful_compliance_rate": 0.0,
  "instruction_retention": 1.0,
  "over_refusal_rate": 0.0,
  "preference_win_rate": 1.0,
  "tasks": 48,
  "teacher_queried_positions": 291,
  "teacher_query_ratio": 0.8290598290598291,
  "tool_utility_retention": 1.0
}
```

## Cost

```json
{
  "gpu_seconds": 41.34,
  "optimizer_updates": 4,
  "peak_vram_bytes": 932001280,
  "wall_seconds": 170.104161
}
```

DPO cost includes the pinned external TRL training job when applicable. Evaluation time is not included in `gpu_seconds`.

## Limitations

- One model family, one deterministic sandbox policy suite and one measured GPU.
- The common SFT checkpoint already saturated every measured policy and utility endpoint.
- Three seeds describe observed variation and do not establish a population claim.
- External IFEval, XSTest, HarmBench and RewardBench endpoints were not measured in this artifact.

Source result SHA-256: `584752dccb91654109c357b8ebb12681a12a9c1476a9ba539dd35e4d860a22ef`

Card content SHA-256: `779c5467412f2c67ff97438428d71136013791a6ba4f30c5cd7805576b4bb0a8`
