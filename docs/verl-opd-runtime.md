# Current verl-style OPD runtime

This is miniVERL's current executable path: one documented, resolved subset of
official verl `v0.8.0`, pinned at `7aed6b23`, compiled into local phases on one
NVIDIA CUDA GPU. It is not a distributed verl runtime and does not accept
arbitrary Hydra YAML.

## Start from the pinned profile

Install the matching CUDA PyTorch build for the machine first, then install the
training and bitsandbytes dependencies:

```bash
python -m pip install "miniverl[train,cuda]"
miniverl data sample --format verl-parquet --out prompts.parquet
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 \
  --config builtin:qwen3-0.6b-1.7b-opd \
  --set 'data.train_files=["prompts.parquet"]' --out plan.json
miniverl run --profile verl-opd-v0.8-single-gpu-v1 --plan plan.json --dry-run
```

Remove `--dry-run` only after inspecting the plan and confirming the model pair
fits. External profiles must explicitly accept the reported high-risk local
reinterpretations; the built-in profile has a value-bound approval manifest.

## Runtime contract

| Source intent | Local effect |
| --- | --- |
| actor dtype, quantization, attention | exact `models.student` runtime settings |
| teacher inference dtype plus miniVERL quantization/attention | exact `models.teacher` settings |
| `data.train_batch_size` | logical trajectories in one strict current-policy update |
| rollout batch/token fields | physical padded rollout limits |
| update trajectory/token fields | physical actor forward limits; never extra optimizer steps |
| `lora_adapter_path` plus pinned metadata | validated trainable student initialization |

`forward_kl_topk` consumes teacher top-k token IDs and log-probabilities and
reports top-k mass/overlap diagnostics. It does not create the explicit K+1
tail bucket used by miniVERL's separate native `bucketed_topk_tail` objective.

## Placement is fail-closed

- NF4/int8 roles use resident local phases; bitsandbytes parameters cannot swap.
- Swap is available only for movable unquantized LoRA roles.
- Shared backbone requires compatible same-base roles.
- Unknown-size quantized roles report `requires_probe` and are not executable
  until feasibility is proven.

Normal planning is weight-free. `plan --probe` is a bounded, cached CUDA
calibration that performs zero optimizer updates. An executable plan must not
select a placement the native runtime rejects for a known static reason.

The generated [compatibility matrix](generated/verl-opd-v0.8-compatibility.json)
and [field-effect evidence](generated/verl-opd-v0.8-field-effects.json) are
compiler-bound and byte-compared in CI.

Next: [quickstart](opd-quickstart.md), [hardware planning](hardware-planning.md),
or [scale-out](verl-opd-scaleout.md).
