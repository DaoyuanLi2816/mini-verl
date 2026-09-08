# Single-GPU verl RL

The `verl-rl-v0.9-single-gpu-v1` compiler translates a resolved, documented
verl v0.9 RL subset into a native miniVERL recipe. Algorithm semantics stay
recognizable while distributed placement becomes an explicit temporal schedule
for one process and one CUDA GPU.

## Quickstart

Create a reward-bearing Parquet file and compile the shipped GRPO example:

```bash
miniverl data sample --task-rewards --rows 8 --out data/rl-prompts.parquet
miniverl import-verl --profile verl-rl-v0.9-single-gpu-v1 \
  --config examples/verl-rl-v0.9-single-gpu.yaml --out local-grpo.yaml
miniverl validate local-grpo.yaml --json
miniverl train local-grpo.yaml --dry-run
```

The importer publishes two files:

- `local-grpo.yaml`: a validated native recipe;
- `local-grpo.import-report.json`: every source field, source value, local
  target, classification, reason and lowering decision.

Run without `--dry-run` after reviewing both artifacts. A source with missing
reward or parameterization choices receives a non-executable
`*.template.yaml`; unknown fields, unresolved interpolation and objective
changes are rejected.

## Supported algorithms

| verl estimator | Local algorithm | Group requirement | Main statistic |
| --- | --- | --- | --- |
| `grpo` + standard normalization | `grpo` | `n > 1` | group-centered, sample-std normalized outcome |
| `grpo` + `norm_adv_by_std_in_grpo: false` | `dr_grpo` | `n > 1` | group-centered outcome without std division |
| `rloo` | `rloo` | `n > 1` | leave-one-out group baseline |
| `reinforce_plus_plus` | `reinforce_plus_plus` | any complete batch | discounted token returns with masked whitening |

All four use the pinned verl v0.9 vanilla dual-clipped actor objective and
token-mean aggregation. Conformance tests compare values, masks, diagnostics,
gradients and optimizer steps against official source at tag `v0.9.0`, commit
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

The exact-wheel RTX 4080 release qualification uses `target_length` only to
prove nonconstant grouped rewards, advantages and parameter-changing GRPO
updates. It is a systems workload, not evidence of task-quality improvement.

Set `algorithm.use_kl_in_reward: true`, a fixed positive `kl_coef`, and an
explicit frozen `miniverl.reference.adapter_path` to enable reference-policy
reward KL. The scheduler switches adapters on a shared compatible backbone,
then restores the actor before the update. Supported verl penalties are `kl`,
`abs`, `mse` and `low_var_kl`.

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

The committed [machine-readable example report](generated/verl-rl-v0.9-compatibility.json)
is regenerated from the shipped YAML in CI.

## Resume and artifacts

Checkpoints store algorithm identity, reward identity, policy version, dataset
cursor, group/sample counters, optimizer state and RNG state. Resume verifies
those identities before generating another group. The CPU integration suite
compares uninterrupted and interrupted GRPO runs tensor-for-tensor and rejects
a changed reward provider.

Run artifacts include schema-v3 trajectories, JSONL reward and metric records,
transactional checkpoints, the resolved recipe and final PEFT adapter. Use
`miniverl inspect` for trajectories and metrics.

## Current boundary

PPO/GAE value and clipped-value primitives are upstream-conformance tested, but
the executable runtime does not yet have an independently trainable critic,
critic optimizer/checkpoint state or actor/reference/critic phase lifecycle.
Ray, multi-node scheduling, FSDP/FSDP2, Megatron and parallel degrees above one
are distributed placement systems and remain outside the local runtime. See
[compatibility](compatibility.md) and [limitations](limitations.md) for the
complete boundary.
