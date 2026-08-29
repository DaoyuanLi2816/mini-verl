# Config overrides

miniVERL accepts one resolved YAML mapping plus familiar dotted `key=value`
tokens. Resolve Hydra/OmegaConf composition inside the trusted verl workflow,
then pass either the YAML, an argv JSON array, or a plain override file to the
local compiler.

```bash
miniverl plan \
  --config verl-opd.yaml \
  --overrides-file site.overrides \
  --set actor_rollout_ref.actor.optim.lr=1e-5 \
  -- \
  'data.train_files=["data/train.parquet"]' \
  distillation.distillation_loss.topk=64
```

Precedence is deterministic:

1. base YAML;
2. each `--overrides-file`, in command order;
3. each `--set`, in command order;
4. tokens after `--`, in command order.

Every occurrence remains in the compiled report with its source, order,
previous value, previous source, final value and whether it is the effective
occurrence. Duplicate fields are therefore visible rather than silently
collapsed.

## Safe input forms

A plain file contains one expression per line. Empty lines and lines beginning
with `#` are ignored:

```text
data.train_batch_size=8
actor_rollout_ref.rollout.name=vllm
```

A `.json` file must contain an array of strings, suitable for argv captured by
a trusted launcher:

```json
[
  "data.train_batch_size=8",
  "actor_rollout_ref.rollout.name=vllm"
]
```

Values use safe YAML scalar/list/dict parsing. `${...}`, non-finite numbers,
unknown fields, YAML object constructors, Hydra `+`/`~` operators and shell
scripts fail closed. No input string is evaluated as code.

## High-risk local reinterpretations

`miniverl plan` always lists fields whose upstream meaning is replaced by a
materially different one-GPU meaning—for example a PPO mini-batch becoming a
logical direct-GKD update batch, or `vllm` selecting the sequential local HF
runtime. The packaged profile carries a reviewed approval manifest. Running an
external YAML requires:

```bash
miniverl run --config verl-opd.yaml --accept-local-reinterpretations
```

This flag accepts only the high-risk mappings printed in that compiled report.
It never enables an unsupported algorithm, multi-GPU field, FSDP behavior or
policy-gradient objective.

For a reviewed run, bind the acceptance and exact inputs once with
`miniverl plan --out plan.json`, then use `miniverl run --plan plan.json`; see
[Immutable execution plans](immutable-plans.md).

## Official-example field coverage

The generated [coverage report](generated/verl-opd-v08-official-fields.json)
classifies every leaf in a reduced field-surface fixture derived from verl
v0.8.0's Apache-2.0 Qwen3 FSDP OPD example. It intentionally retains fields
that are rejected, so coverage means “explained deterministically,” not
“supported.”
