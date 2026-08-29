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
| `verl-opd-v0.8-single-gpu-pg-k1-rewarded-v1` | task reward + sampled `k1` through vanilla policy loss | sampled-token teacher log-probability + deterministic exact-answer reward | explicit group transform; no critic or value model | conformance only |

The first four profiles are reward-free. The rewarded profile is a separate,
strict current-policy contract with a closed deterministic provider and
versioned task/distillation advantage composition. All five use one actor and
one teacher on one CUDA GPU. The PG profiles retain the pinned vanilla
policy-loss form. The measured records compare runtime and semantic
conformance; the rewarded profile has no task-quality result yet.
Library callers may inject an object implementing `RewardProvider` with
`reward.provider: python_api`; YAML and run artifacts never name or load Python
modules. The built-in verl-shaped profile remains fixed to `exact_answer`.
The group-capable profiles use transactional Parquet prompt groups while
leaving the published `n=1` identities unchanged.
See [For verl users](../for-verl-users.md), the
[PG-k1 ADR](../adr/0010-verl-v0.8-pg-k1-contract.md), and the runtime evidence
for the measured boundary.
