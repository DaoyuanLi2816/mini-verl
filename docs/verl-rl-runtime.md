# Single-GPU verl RL

The v1 and v2 compilers translate a resolved, documented verl v0.9 RL subset
into a native miniVERL recipe. v1 covers critic-free estimators; v2 adds
PPO/GAE, an independent critic, actor KL and entropy. Distributed placement
becomes an explicit temporal schedule for one process and one CUDA GPU.

## Quickstart

Create a reward-bearing Parquet file and compile the shipped PPO example:

```bash
miniverl data sample --task-rewards --rows 8 --out data/rl-prompts.parquet
miniverl import-verl --profile verl-rl-v0.9-single-gpu-v2 \
  --config examples/verl-rl-v0.9-single-gpu-ppo.yaml --out local-ppo.yaml
miniverl validate local-ppo.yaml --json
miniverl train local-ppo.yaml --dry-run
```

The importer publishes two files:

- `local-ppo.yaml`: a validated native recipe;
- `local-ppo.import-report.json`: every source field, source value, local
  target, classification, reason and lowering decision.

Run without `--dry-run` after reviewing both artifacts. A source with missing
reward or parameterization choices receives a non-executable
`*.template.yaml`; unknown fields, unresolved interpolation and objective
changes are rejected.

## Supported algorithms

| verl estimator | Local algorithm | Group requirement | Main statistic |
| --- | --- | --- | --- |
| `gae` | `ppo` | any complete batch | discounted GAE from an independently trained value model |
| `grpo` + standard normalization | `grpo` | `n > 1` | group-centered, sample-std normalized outcome |
| `grpo` + `norm_adv_by_std_in_grpo: false` | `dr_grpo` | `n > 1` | group-centered outcome without std division |
| `rloo` | `rloo` | `n > 1` | leave-one-out group baseline |
| `reinforce_plus_plus` | `reinforce_plus_plus` | any complete batch | discounted token returns with masked whitening |

All five use the pinned verl v0.9 vanilla dual-clipped actor objective and
token-mean aggregation. PPO also uses the pinned clipped value objective.
Conformance tests compare values, masks, diagnostics, gradients and optimizer
steps against official source at tag `v0.9.0`, commit
`483b8a009ba3a97563edee3a19887e4862b8094a`.

## Behavior policy and grouped rollouts

Each trajectory records the actor policy version, prompt-group ID, sample ID,
sample seed, generated token span and sampled-token behavior log-probability.
Top-p and top-k choose tokens; the recorded old log-probability follows
upstream's temperature-scaled actor distribution. Tool results and environment
observations remain context and are never selected as model-generated targets.

`data.train_batch_size × actor_rollout_ref.rollout.n` is the logical rollout
batch. `actor.ppo_mini_batch_size` determines the logical actor mini-batch and
lowers to gradient accumulation over the configured physical trajectory batch.
A full logical mini-batch performs one strict fresh-policy update. A smaller
mini-batch is labelled `replay`, matching multiple actor updates from one
upstream rollout rather than pretending every update was freshly sampled.

## Reward and reference roles

The portable profile supports `miniverl.reward.provider: exact_answer` and the
deterministic `target_length` formatting reward, both driven by explicit
Parquet metadata. The Python API can inject a trusted weighted batch provider
or an environment verifier without importing code from YAML.
Provider identity, ordered inputs, component outputs, failures and composition
are bound into run and checkpoint provenance.

`hf_sequence_classifier` adds a trained reward-model role with exact model and
tokenizer revisions. It scores prompt/response batches deterministically and
can return to CPU between phases so actor and reward inference do not compete
for device residency.

The exact-wheel release gate runs PPO with a Qwen3-0.6B actor and independent
critic, verifies parameter-changing actor/value updates and tensor-exact resume,
then exercises the pinned trained reward model as a separate inference role.

Set `algorithm.use_kl_in_reward: true`, a fixed positive `kl_coef`, and an
explicit frozen `miniverl.reference.adapter_path` to enable reference-policy
reward KL. The scheduler switches adapters on a shared compatible backbone,
then restores the actor before the update. Supported verl penalties are `kl`,
`abs`, `mse` and `low_var_kl`.

The v2 profile also maps `actor.use_kl_loss`, `kl_loss_coef`, `kl_loss_type`
and `entropy_coeff` into the actor objective. Reference log-probabilities remain
bound to the same sampled positions as behavior log-probabilities.

## Field classifications

The compiler uses seven dispositions:

| Classification | Meaning |
| --- | --- |
| `exact` | value and unit feed the same experiment meaning |
| `semantically_conformant` | local representation differs, tested meaning is the same |
| `locally_lowered` | physical implementation changes while the objective is preserved |
| `informational_only` | retained in the report but inactive for the selected algorithm |
| `distributed_only` | placement machinery has no one-GPU execution counterpart |
| `not_implemented` | meaningful on one GPU but absent from this runtime |
| `unsupported` | no safe versioned rule exists for the supplied field/value |

The committed [v1 report](generated/verl-rl-v0.9-compatibility.json) and
[v2 PPO report](generated/verl-rl-v0.9-ppo-compatibility.json) are regenerated
from their shipped YAML examples in CI.

## Resume and artifacts

Checkpoints store algorithm and reward identities, policy version, dataset
cursor, group/sample counters, actor and critic weights/optimizers/schedules,
and RNG state. Resume verifies those identities before generating another
group. The integration suite compares uninterrupted and interrupted PPO runs
tensor-for-tensor for both trainable roles.

Run artifacts include schema-v3 trajectories, JSONL reward and metric records,
transactional checkpoints, the resolved recipe and final PEFT adapter. Use
`miniverl inspect` for trajectories and metrics.

## Scale-out handoff

`miniverl export-verl --target-verl v0.9.0` packages the actor adapter, critic
weights, Parquet data, resolved semantics and checksums. Its launch template
remains inactive until the exact base snapshot, upstream reward implementation
and distributed settings have been reviewed. Actor/critic optimizer state is
exactly resumable in miniVERL but is not translated into an upstream sharded
checkpoint. See [compatibility](compatibility.md) and
[limitations](limitations.md) for the complete boundary.
