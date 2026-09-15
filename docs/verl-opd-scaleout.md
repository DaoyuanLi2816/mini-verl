# Take your local OPD run to verl

Export your trained adapter, data and configuration, then use bridge doctor
to see what the upstream setup still needs. Direct-GKD and sampled-k1 PG runs
preserve their model identities and objective settings in the bundle.

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

## Read the readiness report

| State | What it proves |
| --- | --- |
| artifact bundle complete | required portable files and hashes exist |
| upstream config parse passed | the exact pinned source parsed the bounded overrides |
| model/data load smoke passed | local load checks completed for materialized inputs |
| launchable | all required local inputs exist and the fail-closed checks passed |
| distributed execution tested | whether the recorded evidence includes the upstream distributed job |
| algorithm semantic parity | whether a separately declared parity study exists |

Fresh identity-only exports begin before model snapshots are materialized.
Materialization resolves
the exact base snapshots, validates tokenizer/model/adapter/Parquet closure and
publishes a checksummed `launch.sh` transactionally. The later execution and
semantic-parity fields keep those evidence stages distinct from launchability.

The current pure-OPD export has no reward scaffold. The older
`single-gpu-online-distillation-v1` environment/PPO scaffold is retained only
as a [legacy bridge](legacy-verl-bridge.md).

See [materialization details](scaleout-materialization.md) and [compatibility
policy](compatibility.md).
