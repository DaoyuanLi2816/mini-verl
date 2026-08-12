# Run verl-style OPD locally

miniVERL v0.8 supports one pinned, fail-closed subset of verl v0.8: one actor,
one teacher, one generation per prompt, reward-free direct GKD with
`forward_kl_topk`, token-mean aggregation and a LoRA/QLoRA student. It is local
single-GPU execution, not Ray/FSDP or distributed verl execution.

Install a CUDA build of PyTorch that matches your machine first, then:

```bash
python -m pip install "miniverl[train,cuda,bridge]"
miniverl data sample --format verl-parquet --out data/opd-smoke.parquet
miniverl plan --profile verl-opd-v0.8-single-gpu-v1 \
  --config builtin:qwen3-0.6b-1.7b-opd
miniverl run --profile verl-opd-v0.8-single-gpu-v1 \
  --config builtin:qwen3-0.6b-1.7b-opd
```

The built-in recipe downloads pinned Qwen3-0.6B and Qwen3-1.7B snapshots when
they are not cached. Allow roughly 6 GiB of download/cache space and 0.25 GiB
for run artifacts. The plan command itself is CPU-only and weight-free; use
`--offline` for a zero-network compiler smoke.

## Measured reference

| GPU | student / teacher | strategy | limits / top-k | peak reserved | first update | status |
| --- | --- | --- | --- | ---: | ---: | --- |
| RTX 4080 16 GiB | Qwen3-0.6B / Qwen3-1.7B, both NF4 | dual resident | 128 + 16 tokens / 32 | 3.176 GiB | 12.02 s | measured |
| 12 GiB CUDA GPU | same built-in recipe | planner-selected | same | — | — | not measured |
| 24 GiB CUDA GPU | same built-in recipe | planner-selected | same | — | — | not measured |

The measured run completed one current-policy rollout/scoring/update cycle,
exported a loadable PEFT adapter, and used one RTX 4080. It demonstrates
runtime and artifact correctness only; it did not evaluate alignment quality.
The checksummed record is
[`rtx4080-verl-opd-runtime-v1.json`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/rtx4080-verl-opd-runtime-v1.json).

## What `plan` means

`plan` reports estimates separately from measurements. Auto placement uses
model metadata plus the configured VRAM headroom; it never branches on a GPU
product name. Unknown model sizes conservatively select swap. `--probe` is
reserved but fails closed in v0.8.0 rather than loading weights unexpectedly.

Unsupported settings—including policy-gradient OPD, task rewards, reference
KL, multiple teachers, multiple generations, multimodal inputs and every
distributed dimension—are rejected rather than reinterpreted.

## Import and export the same bounded profile

```bash
miniverl import-verl --profile verl-opd-v0.8-single-gpu-v1 \
  --config verl-opd.yaml --out local-opd.yaml
miniverl run --profile verl-opd-v0.8-single-gpu-v1 \
  --config local-opd.yaml --output runs
miniverl export-verl --run runs/<run-id> --target-verl v0.8.0 --out scaleout
miniverl bridge doctor scaleout --require-verl
```

A compatible export contains the standard student PEFT adapter, tokenizer
metadata, exact student/teacher identities, original Parquet bytes, the source
config, compiled plan and pure OPD overrides. No reward scaffold is generated.
Validation data is exported only when the source declared it; an empty
`data.val_files` remains empty rather than duplicating training rows.

The bundle stays `launchable: false` until the exact student and teacher base
snapshots are materialized. A local teacher adapter adds an explicit merge
requirement because the pinned upstream profile does not consume that adapter
path directly. The bridge never claims a distributed verl job ran.
