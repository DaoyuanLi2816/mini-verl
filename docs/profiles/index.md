# Compatibility profiles

A miniVERL compatibility profile is a closed, versioned contract—not a claim
that arbitrary verl YAML works locally. Each built-in profile binds its accepted
schema and field rules to the upstream repository, tag and commit, plus explicit
native-compiler, loss-conformance and export versions.

```bash
miniverl profiles list
miniverl profiles show verl-opd-v0.8-single-gpu-v1
miniverl profiles schema verl-opd-v0.8-single-gpu-v1 --json
```

`profiles show` includes a copyable resolved YAML example and override command.
No entry points or third-party profile code are loaded.

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

The report distinguishes the supported algorithm, accepted and effective
fields, local reinterpretations, informational fields, unsupported fields and
profiles that do not apply. A field being parsed is not evidence that it changes
the native runtime.

## Independent artifact identity

New immutable plans, native run manifests, teacher caches, checkpoints and
scale-out reports carry the complete profile identity. Changing any version
axis produces a new digest. Existing profile names cannot silently acquire new
semantics; a later algorithm path is registered under a different name.

## Which profile should I use?

| Profile | Objective | Teacher target | Trade-off | Status |
| --- | --- | --- | --- | --- |
| `verl-opd-v0.8-single-gpu-v1` | direct GKD `forward_kl_topk` | top-k token IDs and log-probabilities | fuller distributional signal; larger target artifact | measured |
| `verl-opd-v0.8-single-gpu-pg-k1-v1` | sampled `k1` + vanilla policy loss | sampled-token teacher log-probability | sampled-token signal; smaller target artifact | measured |

Both are reward-free, strict current-policy, one-actor/one-teacher profiles on
one CUDA GPU. The PG profile is not PPO: it uses a policy-loss form with a
detached distillation-derived advantage and has no task reward, critic,
reference KL or replay. Neither profile is claimed to be more accurate or
better aligned. See [For verl users](../for-verl-users.md), the
[PG-k1 ADR](../adr/0010-verl-v0.8-pg-k1-contract.md), and the runtime evidence
for the measured boundary.
