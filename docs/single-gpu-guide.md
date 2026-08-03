# Bring your own GPU

miniVERL is a **single-GPU CUDA LLM post-training** stack. It has no GPU model
allowlist and no multi-GPU launcher: one process uses one CUDA device, while
the model pair and sequence budget determine whether a recipe fits.

The shipped Qwen3 recipe is measured on an RTX 4080, but it is not coded for an
RTX 4080. Its portable defaults are:

- `models.device: auto` selects CUDA when PyTorch can use it;
- `dtype: auto` selects bfloat16 when the device supports it and float16
  otherwise;
- NF4 student weights and paged 8-bit Adam reduce the trainable-model footprint;
- `train.trajectory_batch_size: 1` is the conservative compatibility default;
  `2` or `4` can improve update throughput when measured headroom permits;
- `memory.strategy: auto` resolves the supported resident/swap policy;
- an out-of-memory error may reduce only the vocabulary-loss chunk size and
  retry the gradient phase. It never silently changes the objective, model,
  rollout length, or optimizer update.

## Start here

First use the [PyTorch install selector](https://pytorch.org/get-started/locally/)
to install the wheel matching your CUDA driver. The measured stack used the
following channel; choose a different supported channel when your system needs
one:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install "miniverl[train,cuda]"
miniverl doctor
miniverl validate recipes/qwen_consumer_gpu_calc.yaml
miniverl train recipes/qwen_consumer_gpu_calc.yaml --dry-run
```

The `train,cuda` extra adds the training and quantization dependencies; it does
not select a CUDA-enabled PyTorch build on its own.

Then watch free memory in another terminal before the real run:

```bash
nvidia-smi
miniverl train recipes/qwen_consumer_gpu_calc.yaml
```

`doctor` reports the device, CUDA availability and optional dependencies.
`--dry-run` validates model identity, tokenizer compatibility, adapter
provenance and the resolved configuration. It does not prove that every phase
will fit; only an actual run can do that.

The default calculator recipe owns separate 0.6B student and 1.7B teacher
models, so it uses `models.runtime: dual_model`. When student and teacher use
the same base revision, `models.runtime: shared_backbone` can instead keep one
base with separate adapters. The shipped
[`qwen_consumer_gpu_shared.yaml`](../recipes/qwen_consumer_gpu_shared.yaml)
demonstrates the wiring and batch-4 update path; its frozen teacher is a systems
artifact, not a newly quality-qualified recipe. See the
[measured runtime report](consumer-runtime-v1.md) before adapting it.

## What different cards change

These are starting points, not benchmark claims:

| Your card | Sensible first move | Evidence status |
| --- | --- | --- |
| 8–12 GiB, such as an RTX 3070 or Titan V | Try the shipped pair, but be ready to shorten token budgets or select a smaller teacher. Keep `dtype: auto`; Titan V-class hardware resolves to fp16 because it has no bf16 path. | Supported by the device-name-agnostic CUDA path; not measured in this repository |
| 16–24 GiB | Start with the shipped recipe unchanged. Increase budgets only after recording a successful baseline. | The exact default pair is measured on one RTX 4080 16 GiB |
| 24–32+ GiB, such as RTX 3090/4090/5090-class cards | Use the same recipe first; extra headroom can support longer contexts, larger models, or less quantization. Change one variable at a time. | Expected from the same single-device path; not measured here |

The default measured run peaked below 5 GiB of CUDA allocated/reserved memory,
but allocator behavior, kernels, driver versions and generation length vary.
That number is evidence from one machine, **not** a promise that every 8 GiB
card or software stack will complete.

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

Do not infer correctness from a falling loss alone. Run the strict held-out
verifier and inspect the terminal manifest. The published negative controls
are included precisely because an incompatible teacher can optimize normally
while producing a 0% tool policy.

## Share a hardware result

Hardware reports are useful when they include the full configuration and
provenance rather than only a card name:

```bash
miniverl export-benchmark runs/<run-id> --notes "GPU, VRAM, driver, torch and CUDA versions"
```

Validate the exported JSON and open a pull request following
[`benchmarks/README.md`](../benchmarks/README.md). Results from cards other than
the measured RTX 4080 are welcome; until they exist, the README labels those
cards as portable code paths rather than measured performance.
