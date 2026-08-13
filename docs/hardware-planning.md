# Hardware planning and bounded probes

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
