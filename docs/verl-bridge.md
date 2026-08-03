# Verified verl bridge

miniVERL implements compatibility Level 3 for one named profile:
`single-gpu-online-distillation-v1`. It exchanges standard artifacts with the
official [`verl v0.8.0`](https://github.com/verl-project/verl/releases/tag/v0.8.0)
source at commit `7aed6b230776f963fa09509c10d9c3a767d1102c`. The tested compatibility
environment is Python 3.12; the package built from that tag reports version
`0.8.0.dev0`.

![miniVERL to verl architecture](verl-bridge-architecture.svg)

This is a bridge to a documented subset, not generic verl YAML support and not
distributed-runtime parity. miniVERL is independent from the verl project; no
endorsement or upstream compatibility guarantee is implied.

## Compatibility levels

| Level | Contract | v0.6 status |
| --- | --- | --- |
| 0 | prompt/data → rollout → scoring → target/advantage → update → evaluation | documented |
| 1 | Hugging Face, PEFT, safetensors, tokenizer, Parquet and provenance artifacts | validated |
| 2 | named config-field whitelist for a pinned source | validated, fail-closed |
| 3 | generated bundle, exact pin, config/Parquet/adapter/scaffold/hash smoke | validated |

The Level-3 claim means the bundle is structurally loadable and bound to the
pin. It does not mean a distributed training job ran.

## Import the narrow config profile

```bash
miniverl import-verl path/to/verl.yaml \
  --profile single-gpu-online-distillation-v1 \
  --target-verl v0.8.0 \
  --out recipes/imported.yaml
```

`import-report.json` records the source digest, every mapped field,
informational ignores, unsupported fields, conflicts, inserted defaults and
the generated recipe digest. A rejected import still writes the report but
never writes a partial recipe.

| Accepted verl field | miniVERL disposition |
| --- | --- |
| `data.train_files`, `data.val_files`, `data.prompt_key` | retained as bridge metadata in the report; use `convert-dataset` for data |
| `data.max_prompt_length` | contributes to `rollout.max_total_tokens` |
| `data.max_response_length` | `rollout.max_new_tokens_per_turn` and total-token bound |
| `data.seed` | `run.seed` and deterministic split seed |
| `actor_rollout_ref.model.path` | same-base student and policy-conditioned teacher scaffold |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | student gradient checkpointing |
| `actor_rollout_ref.actor.optim.lr` | `train.learning_rate` |
| `trainer.save_freq`, `trainer.test_freq` | checkpoint/evaluation cycle frequencies |
| `trainer.project_name`, `trainer.experiment_name` | portable run name |
| `trainer.total_epochs` | `train.cycles` |

Critic, advantage-estimator, PPO clipping, GRPO grouping, Ray resources,
FSDP/Megatron placement, tensor/pipeline parallelism, vLLM/SGLang placement,
async rollout and multi-node fields fail by default. Unknown fields fail too.

## Convert prompt datasets

```bash
python -m pip install "miniverl[bridge]"
miniverl convert-dataset --from verl-parquet input.parquet --out miniverl.parquet
miniverl convert-dataset --to verl-parquet miniverl.parquet --out export.parquet
```

Both directions validate `data_source`, structured chat `prompt`, `ability`,
`reward_model.ground_truth` and `extra_info`; report accepted/rejected rows,
truncation risk and input/output digests; and never truncate silently.
miniVERL token provenance and teacher targets live only in
`extra_info.miniverl` or its checksummed sidecar. They are distillation targets,
never relabeled as PPO reference log-probabilities.

## Export and inspect a bundle

```bash
miniverl export-verl --run runs/my-alignment \
  --target-verl v0.8.0 \
  --out exports/my-alignment-verl
miniverl bridge doctor exports/my-alignment-verl --json
```

The source run must contain a standard adapter under `model/` and official
prompt-schema `data/train.parquet` plus `data/val.parquet`. The exporter writes:

```text
model/       adapter_config.json, adapter_model.safetensors, tokenizer metadata,
             base-model.json (exact identity; base weights are not bundled)
data/        train.parquet, val.parquet
recipe/      verl-overrides.yaml, launch.sh, REQUIRED_VERL.txt
reward/      reward_or_verifier_scaffold.py
provenance/  source manifests, compatibility report, SHA256SUMS
README.md
```

`bridge doctor` checks the exact target, PEFT config and safetensors structure,
tokenizer structural digest, both Parquet schemas, OmegaConf-compatible recipe
shape, side-effect-free reward import, unsupported semantics, privacy and every
artifact hash. Add `--require-verl` to require a VCS installation whose
`direct_url.json` resolves to the pinned commit.

The generated override points verl at `model/base` and the adapter at `model/`.
Before launch, materialize the exact model id and 40-character revision from
`model/base-model.json`; `launch.sh` fails closed and prints the corresponding
`hf download` command if `model/base/config.json` is absent. It also refuses to
run while the reward scaffold still contains its generated fail-closed body.

## Recorded smoke

The release candidate installed the official commit without its distributed
dependency stack, using Python 3.12. The first Windows build needed
`PYTHONUTF8=1` because the upstream setup reads its UTF-8 README with the local
code page; the same exact commit then built successfully. OmegaConf parsed the
official generated PPO config, found all 14 import-whitelist fields plus the six
export handoff fields, and structurally merged the exported overrides. A
standard `LoraConfig`, safetensors header, both Parquet splits and the
fail-closed reward scaffold loaded; privacy and 14 artifact hashes passed. The checksummed record is
[`generated/verl-bridge-smoke.json`](generated/verl-bridge-smoke.json).

The tiny CPU dry run is intentionally artifact-only. Installing and launching
Ray, FSDP/Megatron and vLLM/SGLang was outside this test, so distributed
execution remains **not tested**.

## Unsupported semantic conversions

The bridge does not convert optimizer state, distributed RNG, FSDP or Megatron
native checkpoints, Ray runtime state, or teacher cache semantics. It does not
claim that a miniVERL teacher cache is a PPO reference cache. Review and test
the generated reward scaffold before any scale-out launch.
