# Consumer Runtime v1

> A low-memory one-GPU runtime for actor rollout, teacher/reference scoring and
> online policy update.

v0.4 adds padded multi-trajectory update batches and a shared-backbone adapter
runtime while keeping the existing objective and rollout semantics. The default
remains `dual_model` with `train.trajectory_batch_size: 1`.

## Measured result

The preregistered systems benchmark used one RTX 4080, Qwen3-0.6B, NF4 weights
with FP32 compute, eight deterministic SQLite RecoveryBench oracle trajectories
and one strict-OPD reverse-KL update. Values are medians of three measured
updates after one warmup update.

![Throughput versus reserved VRAM](consumer-runtime-v1-pareto.svg)

| runtime | physical batch | physical forwards | padding | trajectories/s | peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| dual model | 1 | 8 | 0 | 2.369 | 2.551 GiB |
| dual model | 2 | 4 | 270 | 3.523 | 2.893 GiB |
| dual model | 4 | 2 | 616 | **3.866** | 3.035 GiB |
| dual model | auto (8) | 1 | 1,636 | 3.075 | 4.568 GiB |
| shared backbone | 1 | 8 | 0 | 2.251 | 2.223 GiB |
| shared backbone | 2 | 4 | 270 | 3.271 | 2.225 GiB |
| shared backbone | 4 | 2 | 616 | **3.475** | **2.227 GiB** |
| shared backbone | auto (8) | 1 | 1,636 | 2.904 | 3.834 GiB |

Batch-4 improved end-to-end throughput by 1.63× in the dual runtime and 1.54×
in the shared runtime. Sharing saved 26.6% of peak reserved memory at batch-4,
but was 10.1% slower than the dual runtime in that cell. `auto` was not the
fastest choice because padding all eight variable-length trajectories added
1,636 tokens. These are workload- and machine-specific measurements, not GPU
or model-family guarantees.

The shared-auto profiler is retained as a checksummed operator summary. Matrix
multiplication (`aten::mm`, `aten::bmm` and underlying GEMM kernels) dominates
CUDA self time. The measured adapter-role switch was about 23.7 ms; it is real
overhead, not subtracted from the end-to-end result.

## Equivalence gate

Every cell reused the same ordered tasks, unpadded sequences, selected
positions, token weights, teacher targets, student initialization, optimizer
settings and one-update objective. The trajectory and target digests were
identical across all eight cells.

| comparison across all cells | maximum observed | preregistered tolerance |
| --- | ---: | ---: |
| loss absolute difference | 1.248e-6 | 1e-5 |
| gradient maximum absolute difference | 7.227e-6 | 3e-4 |
| gradient maximum relative to reference maximum | 2.803e-5 | 3e-3 |
| updated-logit maximum absolute difference | 1.299e-4 | 3e-3 |
| updated-logit maximum relative to reference maximum | 3.953e-6 | 3e-3 |

For each physical batch, dual and shared execution matched exactly in loss,
the complete trainable gradient and post-update probe logits. All 12 declared
comparisons passed. Attention masks also have direct isolation tests, padding
never enters the loss, and the update never constructs `[B, T, V]` logits.

Revision 1.1 of the preregistration replaced nondeterministic CUDA SDPA with
eager attention after a quick diagnostic. Revision 1.2 retained NF4 weights but
changed compute from BF16 to FP32 after the deterministic diagnostic found
batch-shape-dependent BF16 gradient drift. Both amendments were public before
this sole headline measurement; the diagnostic files are not headline data.

## Runtime model

The high-level controller now exposes a small local role graph without copying
verl's Ray, DataProto, FSDP or placement APIs.

| miniVERL component | local verl-style role |
| --- | --- |
| student backend | `ActorPolicy` |
| rollout runner | `RolloutRuntime` |
| teacher scorer | `TeacherPolicy` extension |
| optional fixed scorer | `ReferencePolicy` |
| environment verifier | `RewardOrVerifier` |
| alignment plus teacher-target construction | `TargetBuilder` |
| trainer update path | `UpdateRuntime` |
| evaluator | `EvaluationRuntime` |
| run/checkpoint/report persistence | `ArtifactBridge` |
| trainer controller and memory plan | local single-device controller and placement plan |

`models.runtime: shared_backbone` owns one physical Hugging Face base with a
trainable student adapter, a frozen teacher adapter and an optional frozen
reference adapter. The optimizer sees only student parameters. Role switches
restore the previous adapter after success or failure, and nested use of the
already-active role is a no-op. Teacher and reference roles remain semantically
distinct even when they happen to reference the same frozen artifact.

```yaml
models:
  backend: hf
  runtime: shared_backbone
  device: cuda
  student:
    model_id: your/base
    revision: immutable-commit
    quantization: nf4
    lora: {enabled: true, r: 16, alpha: 32, dropout: 0.0}
  teacher:
    model_id: your/base
    revision: immutable-commit
    quantization: nf4
    adapter:
      source: hub
      path: your/frozen-teacher-adapter
      revision: immutable-adapter-commit
train:
  trajectory_batch_size: 4
```

All shared roles must use the same base, tokenizer, precision, quantization and
attention settings. The teacher adapter is required and must be revision-pinned
for a reproducible run. An optional `models.reference` uses the same base with
its own pinned adapter. Quantized shared backbones are resident; `swap` is
rejected rather than pretending that bitsandbytes parameters can move safely.
Checkpoints export a standard student PEFT adapter, and manifests record the
logical roles plus immutable adapter provenance without exposing local paths.

## Batch semantics

`train.gradient_accumulation_steps` remains the number of trajectories in one
optimizer group. `train.trajectory_batch_size` controls how many of those
trajectories share a padded backbone forward:

- `1` preserves sequential execution and is the compatibility default;
- an integer groups deterministic length buckets of that size;
- `auto` pads the whole optimizer group and can be slower when lengths differ.

Each trajectory retains its own causal attention boundary and objective
normalization. Selected hidden states are flattened only after mask-aware
backbone execution, then projected in `[selected positions, vocabulary]`
chunks. SFT, offline KD and strict OPD use the same batching contract for exact
and top-k-plus-tail objectives. Rollout decoding itself is still one sequence
at a time.

## Frozen artifacts and scope

| artifact | SHA-256 |
| --- | --- |
| [`consumer-runtime-v1.json`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/consumer-runtime-v1.json) | `a302da31af99f1d29f1efd4e6b3dbeb6ea4ac956bba102ca8a1bee8dff0319eb` |
| [`consumer-runtime-v1-profiler.json`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/consumer-runtime-v1-profiler.json) | `66111cd7fc876cf1befea3297a1a51bcd99252c0bf8989c029381e1dc155a98b` |
| [`consumer-runtime-v1-pareto.svg`](consumer-runtime-v1-pareto.svg) | `98645a668a7832423d28b621262292619615917f037adf7219ff1bf071fb2fea` |

The systems-only teacher adapter is pinned at
`DaoyuanLi/mini-verl-qwen3-0.6b-consumer-runtime-teacher@e277b92d8c1fdb76cd133f872f0ddd2c47a4ab8c`.
It exists to compare runtime ownership, not as a newly qualified teacher.
No compatible preregistered 4B or 7B teacher adapter was available, so those
diagnostics are retained as `not_run` rather than replaced with model-size
claims. The immutable calculator benchmark remains byte-identical at SHA-256
`53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`.
