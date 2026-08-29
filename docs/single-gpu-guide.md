# Bring your own GPU

miniVERL is a **single-GPU CUDA LLM post-training** stack. One process uses one
CUDA device, and the model pair, sequence budget and runtime strategy determine
how the recipe fits.

The shipped Qwen3 recipe follows a device-name-agnostic CUDA path and has a
measured RTX 4080 reference. Its portable defaults are:

- `models.device: auto` selects CUDA when PyTorch can use it;
- `dtype: auto` selects bfloat16 when the device supports it and float16
  otherwise;
- NF4 student weights and paged 8-bit Adam reduce the trainable-model footprint;
- `train.trajectory_batch_size: 1` is the conservative compatibility default;
  `2` or `4` can improve update throughput when measured headroom permits;
- `memory.strategy: auto` resolves the supported resident/swap policy;
- an out-of-memory retry reduces the mathematically neutral vocabulary-loss
  chunk size while keeping the objective, model, rollout length and optimizer
  update fixed.

## Start here

First use the [PyTorch install selector](https://pytorch.org/get-started/locally/)
to install the wheel matching your CUDA driver. For a flexible install:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install "miniverl[train,cuda]"
miniverl doctor
miniverl validate recipes/qwen_consumer_gpu_calc.yaml
miniverl train recipes/qwen_consumer_gpu_calc.yaml --dry-run
```

The `train,cuda` extra adds the training and quantization dependencies; it does
not select a CUDA-enabled PyTorch build on its own.

For the exact stack measured by the maintainer on one RTX 4080, use the
machine-readable
[`environments/known-good-rtx4080-cu130.json`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/environments/known-good-rtx4080-cu130.json)
and its constraints:

```bash
python -m pip install "torch==2.13.0+cu130" \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install "miniverl[train,cuda]" \
  --constraint environments/known-good-rtx4080-cu130.txt
python scripts/check_known_good_environment.py
```

The ordinary dependency ranges remain the library contract. In the manifest,
Python and package pins are reproducibility inputs; GPU name, VRAM, driver and
CUDA runtime record the measured machine. In the schema, the driver and CUDA
runtime are observed audit fields; other GPUs are unmeasured by the maintained
reference and can be recorded with the same schema.

Then watch free memory in another terminal before the real run:

```bash
nvidia-smi
miniverl train recipes/qwen_consumer_gpu_calc.yaml
```

`doctor` reports the device, CUDA availability and optional dependencies.
`--dry-run` validates model identity, tokenizer compatibility, adapter
provenance and the resolved configuration. `plan --probe` adds a bounded
measurement before the full run establishes end-to-end fit.

The default calculator recipe owns separate 0.6B student and 1.7B teacher
models, so it uses `models.runtime: dual_model`. When student and teacher use
the same base revision, `models.runtime: shared_backbone` can instead keep one
base with separate adapters. The shipped
[`qwen_consumer_gpu_shared.yaml`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/recipes/qwen_consumer_gpu_shared.yaml)
demonstrates the wiring and batch-4 update path. See the
[measured runtime report](consumer-runtime-v1.md) for its systems evidence and
teacher provenance before adapting it.

## What different cards change

Use these as starting points and record the resulting plan:

| Your card | Sensible first move | Evidence status |
| --- | --- | --- |
| 8–12 GiB, such as an RTX 3070 or Titan V | Try the shipped pair, then shorten token budgets or select a smaller teacher if the probe requires it. Keep `dtype: auto`; Titan V-class hardware resolves to fp16. | Portable CUDA path; contribute a measured record |
| 16–24 GiB | Start with the shipped recipe unchanged. Increase budgets only after recording a successful baseline. | The exact default pair is measured on one RTX 4080 16 GiB |
| 24–32+ GiB, such as RTX 3090/4090/5090-class cards | Use the same recipe first; extra headroom can support longer contexts, larger models, or less quantization. Change one variable at a time. | Portable CUDA path; contribute a measured record |

The default measured run peaked below 5 GiB of CUDA allocated/reserved memory.
Allocator behavior, kernels, driver versions and generation length all affect
the headroom on another machine, which is why the plan records them.

## If the recipe does not fit

Work down this list and preserve the resulting YAML with the run:

1. Close other GPU processes and re-run `nvidia-smi`.
2. Reduce `loss.chunk_size`; the automatic OOM retry can do this down to
   `memory.min_chunk_size`.
3. Reduce `train.trajectory_batch_size` to `2` or `1`; this changes physical
   execution without changing the optimizer-group objective.
4. Reduce rollout token limits or the number of rollouts accumulated in a
   cycle. This changes the experiment budget, so label the run accordingly.
5. Choose a smaller teacher/student pair with byte-identical tokenizers and
   compatible output vocabularies.
6. For an unquantized student, evaluate `memory.strategy: swap`. Quantized
   students remain resident because moving bitsandbytes modules between
   devices is not a supported lifecycle.

Use the strict held-out verifier and terminal manifest alongside the loss. The
published negative controls show why task behavior and teacher protocol
competence belong in the same review.

## Share a hardware result

Hardware reports are useful when they include the full configuration and
provenance rather than only a card name:

```bash
miniverl export-benchmark runs/<run-id> --notes "GPU, VRAM, driver, torch and CUDA versions"
```

Validate the exported JSON and open a pull request following
[`benchmarks/README.md`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/README.md). Results from cards other than
the measured RTX 4080 are welcome and extend the public hardware matrix.
