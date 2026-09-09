# Hardware planning and bounded probes

## Evidence before a fit claim

The maintained qualification device is one RTX 4080. For the new PPO/GRPO
workflow, `train local.yaml --dry-run --json` reports logical prompts,
trajectories, actor updates and critic updates before model loading.
It is a schedule/configuration plan, not a measured VRAM prediction.

| VRAM class | PPO/GRPO planning status | Useful evidence |
| --- | --- | --- |
| 8 GB | unknown until measured for the exact recipe | `doctor`, token bounds, actual peak/retries |
| 12 GB | unknown until measured for the exact recipe | same recipe and stack identity |
| 16 GB | maintainer RTX 4080 qualification for named workloads | exact-wheel release record and `inspect` |
| 24 GB | unknown on other devices until measured | community submissions stay unreviewed |

Capacity alone does not establish kernel, dtype or model compatibility.
Measured numbers bind the model/revision, algorithm, role strategy, token
bounds, package versions and GPU. Estimates must identify their method;
absence stays unknown. Quantized roles cannot swap, and PPO cannot share a
trainable critic adapter as though it were a frozen teacher. Illegal placement
is rejected before training.

## OPD calibration

Normal planning is CPU-safe and weight-free. It labels memory as estimated and
time as unknown:

```bash
miniverl plan --config verl-opd.yaml --out plan.json --offline
```

After reviewing the config, an explicit probe can calibrate the exact pinned
models on one visible CUDA GPU:

```bash
miniverl plan --config verl-opd.yaml \
  --accept-local-reinterpretations --out probed-plan.json \
  --probe --offline
```

The bounded probe loads roles sequentially. It measures actor static memory,
greedy 2-token rollout candidates, one selected-position backward, teacher
static memory, and one top-k teacher score. It creates no optimizer, performs
zero parameter updates, and publishes no checkpoint. Role objects and temporary
tensors are destroyed between phases; failure to return allocation near the
starting CUDA baseline invalidates the probe.

Results separate measured phase values, recommendations and failed OOM
candidates. The cache key binds GPU UUID/name/capability/memory, driver, CUDA
runtime, Torch and miniVERL versions, plan digest, model/tokenizer revisions,
quantization, LoRA, token bounds and top-k. A mismatched or modified cache is
never reused. Use `--force-probe` to remeasure deliberately.

The probe is calibration, not training or a throughput benchmark. Its tiny
inputs do not prove that the full logical workload fits. Retain configured
headroom and treat the recommended batches as conservative starting points;
the runtime still fails closed rather than changing model, teacher, context,
top-k or loss semantics after an OOM.
