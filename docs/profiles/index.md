# Compatibility profiles

A miniVERL compatibility profile is a closed, versioned contract that binds an
accepted schema and its field rules to one upstream repository, tag and commit.
Native-compiler, loss-conformance and export versions travel with that identity.

```bash
miniverl profiles list
miniverl profiles show verl-opd-v0.8-single-gpu-v1
miniverl profiles schema verl-opd-v0.8-single-gpu-v1 --json
```

`profiles show` includes a copyable resolved YAML example and override command.
The packaged registry is data-only, so profile inspection stays deterministic.

## Check a resolved profile

```bash
miniverl compat explain \
  --profile verl-opd-v0.8-single-gpu-v1 \
  actor_rollout_ref.actor.ppo_mini_batch_size

miniverl compat check \
  --profile verl-opd-v0.8-single-gpu-v1 \
  --config verl-opd.yaml \
  --accept-local-reinterpretations --json
```

The report distinguishes the selected algorithm, effective fields, local
reinterpretations, informational fields and out-of-profile values. Each
effective field is also covered by mutation-based field-effect evidence.

## Independent artifact identity

New immutable plans, native run manifests, teacher caches, checkpoints and
scale-out reports carry the complete profile identity. Changing any version
axis produces a new digest, and a later algorithm path receives its own profile
name.

## Which profile should I use?

| Profile | Objective | Teacher target | Trade-off | Status |
| --- | --- | --- | --- | --- |
| `verl-opd-v0.8-single-gpu-v1` | direct GKD `forward_kl_topk` | top-k token IDs and log-probabilities | fuller distributional signal; larger target artifact | measured |
| `verl-opd-v0.8-single-gpu-pg-k1-v1` | sampled `k1` + vanilla policy loss | sampled-token teacher log-probability | sampled-token signal; smaller target artifact | measured |
| `verl-opd-v0.8-single-gpu-grouped-v1` | direct GKD over independent grouped samples | top-k token IDs and log-probabilities | configurable `n>1`; no group baseline | conformance only |
| `verl-opd-v0.8-single-gpu-pg-k1-grouped-v1` | sampled `k1` over independent grouped samples | sampled-token teacher log-probability | configurable `n>1`; no GRPO estimator | conformance only |

All four are reward-free, strict current-policy, one-actor/one-teacher profiles
on one CUDA GPU. The PG profiles use a detached distillation-derived advantage
with the pinned vanilla policy-loss form. The measured records compare runtime
and semantic conformance; task-quality questions belong to a declared benchmark.
The two grouped profiles add transactional Parquet prompt groups while leaving
the published `n=1` identities unchanged; each sample is optimized independently.
See [For verl users](../for-verl-users.md), the
[PG-k1 ADR](../adr/0010-verl-v0.8-pg-k1-contract.md), and the runtime evidence
for the measured boundary.
