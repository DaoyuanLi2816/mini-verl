# Run a verl config directly

Bring a resolved verl `v0.9.0` PPO/GRPO-style config to one CUDA GPU:

```bash
miniverl run resolved-verl.yaml --dry-run
miniverl run resolved-verl.yaml \
  --bind data.train_files=/local/train.parquet \
  --bind data.val_files=/local/val.parquet \
  --bind reward.function=/local/reward.py:compute_score \
  --trust-reward-code YOUR_REVIEWED_FILE_SHA256 \
  --output runs --run-id prototype
```

The original file stays unchanged. miniVERL selects
`verl-rl-v0.9-single-gpu-v4`, classifies every leaf, constructs a validated
internal `RunConfig`, then executes local phases. The pin is official verl
`v0.9.0`, commit `483b8a009ba3a97563edee3a19887e4862b8094a`.

Resolve Hydra defaults and `${...}` in your trusted upstream environment first.
The direct command reads data; it does not execute YAML imports or shell scripts.
Paths are relative to the working directory. Scientific-notation learning rates
such as `1e-5` are supported. Native recipes and the historical v1/v2/v3
`import-verl` profiles remain available for advanced workflows.

## Start without a checkout

After installing the matching CUDA PyTorch build and `miniverl[train]`:

```bash
miniverl data sample --reward-profile target-length --rows 8 --out data/rl-prompts.parquet
miniverl run --example ppo --bind reward.provider=target_length --dry-run
miniverl run --example ppo --bind reward.provider=target_length --run-id direct-ppo
miniverl inspect runs/direct-ppo
miniverl run --example ppo --bind reward.provider=target_length \
  --resume-from runs/direct-ppo/checkpoints/step-000002
miniverl export-adapter --run runs/direct-ppo --out runs/direct-ppo/model
miniverl export-verl --run runs/direct-ppo --target-verl v0.9.0 --out ppo-handoff
miniverl bridge doctor ppo-handoff
```

Replace `ppo` with `grpo` and use a new run/output name for the grouped workflow.
Both packaged inputs use upstream field names, an explicit LoRA actor and two
rollout iterations with two actor epochs. PPO has its own LoRA critic. They
exercise validation, prompt filtering and exact checkpoint replay; GRPO also
exercises sequence-mean/token-mean reduction. The length reward is an explicit
systems exercise, not an external model-quality evaluation.

## Read the two statuses

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

## Current boundaries

The [corpus](verl-compatibility-corpus.md) separates semantic acceptance from
exact-model execution. Group selection, rollout oversampling/replacement and
shuffled PPO minibatch passes are explicit rejections, as are unknown active
objectives. Multi-modal input and custom chat preprocessing need a separately
implemented contract. Model architecture/tokenizer compatibility is checked at
construction; semantic acceptance alone does not qualify arbitrary models.

The portable handoff currently carries PEFT actor artifacts and PPO critic
weights. A dense actor can train locally but needs a separate full-model export
path. Handoff scripts remain templates until base snapshots, reference roles and
reward mappings are materialized. Distributed verl execution is not tested.
The [limitations](limitations.md) page collects broader scientific boundaries.
