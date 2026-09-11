# Run a verl config directly

Bring a verl `v0.9.0` config tree, name and launch overrides to one CUDA GPU.
Install matching CUDA PyTorch, then `pip install "miniverl[train,hydra]"`:

```bash
miniverl run --verl-config-path /path/to/verl/trainer/config \
  --verl-config-name ppo_trainer --dry-run --json \
  algorithm.adv_estimator=grpo actor_rollout_ref.rollout.n=8 trainer.n_gpus_per_node=8
```

Omit `--verl-config-path` to use the exact upstream tree packaged in the wheel.
Add runtime `--bind` options and remove `--dry-run` to execute. No intermediate
resolved YAML or native recipe is required. The existing resolved-file path also works:

```bash
miniverl run resolved-verl.yaml --dry-run
miniverl run resolved-verl.yaml \
  --bind data.train_files=/local/train.parquet \
  --bind data.val_files=/local/val.parquet \
  --bind reward.function=/local/reward.py:compute_score \
  --trust-reward-code YOUR_REVIEWED_FILE_SHA256 \
  --output runs --run-id prototype
```

Inputs stay unchanged. Native Hydra selects `verl-rl-v0.9-single-gpu-v5`;
resolved YAML retains `verl-rl-v0.9-single-gpu-v4`. Both classify every leaf, construct a validated
internal `RunConfig`, then executes local phases. The pin is official verl
`v0.9.0`, commit `483b8a009ba3a97563edee3a19887e4862b8094a`.

Hydra 1.3.2 composes defaults, dotted scalar/list/dict overrides, quoted strings,
booleans, null, `+`, `++`, `~`, ordered precedence, references and `oc.select` in
a fresh process. `--print-resolved` prints the exact compiler input;
`--dry-run --json` prints the field report. Runtime paths use the current working
directory. Existing native recipes and historical `import-verl` profiles remain available.

## Start without a checkout

After installing the matching CUDA PyTorch build and `miniverl[train,hydra]`:

```bash
miniverl data sample --reward-profile target-length --rows 8 --out data/rl-prompts.parquet
miniverl run --example hydra-ppo --bind reward.provider=target_length --dry-run
miniverl run --example hydra-ppo --bind reward.provider=target_length --run-id direct-ppo
miniverl inspect runs/direct-ppo
miniverl run --example hydra-ppo --bind reward.provider=target_length \
  --resume-from runs/direct-ppo/checkpoints/step-000002
miniverl export-adapter --run runs/direct-ppo --out runs/direct-ppo/model
miniverl export-verl --run runs/direct-ppo --target-verl v0.9.0 --out ppo-handoff
miniverl bridge doctor ppo-handoff
```

Replace `hydra-ppo` with `hydra-grpo` and use a new run/output name for the grouped workflow.
Both packaged inputs use upstream field names, an explicit LoRA actor and two
rollout iterations with two actor epochs. PPO has its own LoRA critic. They
exercise validation, prompt filtering and exact checkpoint replay; GRPO also
exercises sequence-mean/token-mean reduction. The length reward is an explicit
systems exercise, not an external model-quality evaluation.

## Read the three statuses

`composition_status: resolved` records successful Hydra composition. A malformed
override, unresolved reference or wrong config tree instead returns `failed`,
with semantics and execution `not_assessed`. After composition:

| Semantic status | Execution status | Meaning / next action |
| --- | --- | --- |
| `accepted` | `requires_bindings` | Supply listed local Parquet or trusted reward inputs. |
| `accepted` | `preflight_required` | Static compilation passed; execution will resolve snapshots and check capacity. |
| `accepted` | `hardware_blocked` | The exact requested roles exceed the estimated device envelope, or CUDA is unavailable. |
| `accepted` | `executable_with_local_lowering` | Preflight passed with physical microbatch one and temporal offload. |
| `accepted` | `completed` | The local training run completed. |
| `rejected` | `unsupported_semantics` | The report identifies experiment behavior the selected profile cannot preserve. |

`--dry-run --json` is a static, model-download-free semantic report. Actual
execution resolves immutable model revisions and parameter metadata, loads the
tokenizer to filter prompts, and derives epochs from retained drop-last batches.
The GPU estimate includes role state and an activation reserve; it is not a
guarantee of fit and does not measure available host RAM. Large temporal roles
also need sufficient system memory.

## Runtime bindings

| Binding | Purpose |
| --- | --- |
| `data.train_files`, `data.val_files` | A local path or YAML list of Parquet paths. |
| `actor.model`, `actor.revision`, `actor.tokenizer_revision` | Explicit actor snapshot rebinding; a same-base critic follows a model rebind unless separately bound. |
| `critic.model`, `critic.revision` | An independent PPO critic snapshot. |
| `reward.model`, `reward.revision`, `reward.tokenizer_revision` | Trained classifier identity. |
| `reward.function` | Reviewed `file.py:function` implementing the upstream reward ABI. |
| `reward.provider` | Explicit `exact_answer` or `target_length` built-in exercise. |
| `rollout.backend` | `hf_cached` or `hf_reference` local generation. |
| `output` | Parent run directory; `--output` is the CLI alternative. |
| `sampling.max_attempted_groups` | v5-only positive per-cycle execution guard; exhaustion fails before committing a partial optimizer batch. |

A reward function receives `data_source`, `solution_str`, `ground_truth`, and
`extra_info`; it returns a finite scalar or a dictionary containing `score`.
Approve the SHA-256 of the reviewed file with `--trust-reward-code`. The code
runs with your process permissions. Approval is for that file, not a sandbox or
a claim about its imported dependencies. Use process-environment credentials
and the normal model cache; never put access tokens in config or bindings.

A smaller `actor.model` is an explicit experiment change, recorded as such.
Cluster node count, GPU topology and offload hints are physical lowerings.
Logical prompt count, group size, minibatch size and epoch count are retained.

## Semantics and provenance

- Token mean/sum, sequence mean of token means/sums, and normalized sequence
  sums have upstream scalar, gradient and optimizer conformance tests. Zero-token
  sequences do not enter the mean denominator.
- Actor and critic minibatches retain upstream prompt units multiplied by
  `rollout.n`. Physical microbatches do not change the logical reduction.
- Repeated actor epochs keep old-policy probabilities fixed; PPO also keeps
  old values fixed across its independently scheduled critic epochs.
- AdamW betas and learning-rate schedules retain source values. Upstream
  scheduler steps occur once per rollout iteration, not per local microbatch.
- Overlong prompts are filtered before sampling. Dataset bytes, filtered counts,
  shuffle seed and drop-last traversal bind the schedule and resume identity.
- Reference KL uses an independently frozen initial base. The ordinary upstream
  classifier RM path uses complete chat rendering and raw last-class logits,
  corresponding to `classify(use_activation=False)` rather than a softmax score.

Each run stores `verl-source.yaml`, `verl-direct-report.json`,
`config.resolved.yaml`, the manifest, trajectories, rewards, metrics and
checkpoints. The direct report includes field classification, bindings, the
hardware plan and IR digest. Resume requires the same source bytes and bindings;
resolved revisions are reused instead of following a moving Hub branch.

Hydra runs also store `verl-composition.json`: exact input path/name, defaults,
ordered overrides, per-file/tree hashes and resolved SHA-256. Resume verifies
that composition identity as well as source bytes and bindings.

## Shuffling, filtering and refill

Actor and critic use independent upstream-seeded torch DataLoaders. A generator
is created per role/update call and advances across epochs, including DataLoader's
base-seed consumption. Shuffling applies to trajectories, while physical
microbatching leaves their logical order and old log-probabilities/values intact.

`algorithm.filter_groups.enable=true` requires a named `metric` in reward extra
info (`score` uses raw reward). Complete groups with multiple samples and exactly
zero NumPy standard deviation are dropped; singleton groups are retained. Reward
normalization and PPO/GAE see only the final retained batch. Numeric dictionary
components from approved reward code are retained for filtering.

Pinned V1 synchronous sampling forces one prompt per dispatch. Filtered groups
add two replacement credits; fully failed, empty groups add one when
`trainer.v1.sampler.sync_refill_failed_groups=true`. Dispatch waves are bounded
by `max_inflight_gen_batches`; surplus is drained and discarded. miniVERL uses
recorded dispatch order to break equal-age ties where upstream set order is
unspecified. Every attempt/decision and the final selected IDs appear in
`rl_group_selection` / `rl_sampling` metrics. Recovery replays the whole cycle
from its committed boundary, including collection, rewards and both update roles.

The pinned V1 implementation ignores legacy `max_num_gen_batches`; the declared
`rollout.over_sample_rate` also has no execution consumer. Both are reported as
informational, not advertised as working retry controls. Use the explicit local
attempt guard when an all-filtered stream needs a finite stopping bound.

## Current boundaries

The [corpus](verl-compatibility-corpus.md) separates semantic acceptance from
exact-model execution. Unknown active objectives, partial failed-group masks,
multi-modal input and custom chat preprocessing need a separately
implemented contract. Model architecture/tokenizer compatibility is checked at
construction; semantic acceptance alone does not qualify arbitrary models.

The portable handoff currently carries PEFT actor artifacts and PPO critic
weights. A dense actor can train locally but needs a separate full-model export
path. Handoff scripts remain templates until base snapshots, reference roles and
reward mappings are materialized. Distributed verl execution is not tested.
The [limitations](limitations.md) page collects broader scientific boundaries.
