# Current OPD scale-out contract

A completed direct-GKD or sampled-k1 PG profile run can export portable PEFT,
Parquet, config and provenance artifacts for the exact pinned upstream profile.
This is an artifact handoff, not proof that a distributed job ran or that
miniVERL and verl have algorithmic parity.

```bash
miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout
miniverl bridge materialize scaleout --download --offline
miniverl bridge doctor scaleout --require-verl
```

The export preserves source-run student and teacher identities, model
revisions, prompt/response bounds, learning rate, schedule, Parquet bytes and
the selected profile's exact `forward_kl_topk` or sampled-k1 policy-loss
overrides. The PG bundle has no top-k requirement. Any inserted placeholder is listed; missing
validation data remains missing rather than being replaced with training data.

## Readiness states stay separate

| State | What it proves |
| --- | --- |
| artifact bundle complete | required portable files and hashes exist |
| upstream config parse passed | the exact pinned source parsed the bounded overrides |
| model/data load smoke passed | local load checks completed for materialized inputs |
| launchable | all required local inputs exist and the fail-closed checks passed |
| distributed execution tested | always false in the current evidence |
| algorithm semantic parity | not claimed |

Fresh identity-only exports remain `launchable: false`. Materialization resolves
the exact base snapshots, validates tokenizer/model/adapter/Parquet closure and
publishes a checksummed `launch.sh` transactionally. Even `launchable: true`
does not say Ray, FSDP, vLLM or a distributed optimizer step executed.

The current pure-OPD export has no reward scaffold. The older
`single-gpu-online-distillation-v1` environment/PPO scaffold is retained only
as a [legacy bridge](legacy-verl-bridge.md).

See [materialization details](scaleout-materialization.md) and [compatibility
policy](compatibility.md).
