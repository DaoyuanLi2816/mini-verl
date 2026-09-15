# Plan a workload for your GPU

Start by inspecting the schedule, then calibrate OPD memory with a short probe.
Model size, sequence lengths and the roles resident at each phase determine
how much VRAM a run needs. The [single-GPU guide](single-gpu-guide.md) explains
the settings you can adjust.

## Inspect the PPO/GRPO schedule

The maintained qualification device is one RTX 4080. For the new PPO/GRPO
workflow, `train local.yaml --dry-run --json` reports logical prompts,
trajectories, actor updates and critic updates before model loading.
The output counts planned work; actual peak VRAM comes from executing the recipe.

## Hardware coverage

| VRAM class | PPO/GRPO planning status | Useful evidence |
| --- | --- | --- |
| 8 GB | unknown until measured for the exact recipe | `doctor`, token bounds, actual peak/retries |
| 12 GB | unknown until measured for the exact recipe | same recipe and stack identity |
| 16 GB | maintainer RTX 4080 qualification for named workloads | exact-wheel release record and `inspect` |
| 24 GB | unknown on other devices until measured | community submissions stay unreviewed |

Measurements include the model/revision, algorithm, role strategy, token bounds,
package versions and GPU so you can compare them to your setup. Check kernel
and dtype support as well as VRAM capacity. Placement rules keep quantized roles
resident and PPO's trainable critic separate from a frozen teacher.

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

Use the recommended batches as starting points and retain the configured
headroom: the probe's tiny inputs calibrate phases, while a complete run measures
the full workload. An OOM stops execution with the model, teacher, context,
top-k and loss settings preserved. See [limitations](limitations.md) for probe scope.
