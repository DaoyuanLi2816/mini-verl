# Consumer Runtime

Choose a dual-model setup or share one resident backbone between PEFT roles
to make room for training on a single GPU.
Padded trajectory updates improve update throughput without changing the
effective optimizer batch or the strict-OPD freshness contract.

<picture>
  <source media="(max-width: 900px)" srcset="../consumer-runtime-v1-pareto-mobile.svg">
  <img src="../consumer-runtime-v1-pareto.svg" alt="Measured throughput and peak-memory tradeoffs for shared-backbone and dual-model training across trajectory batch sizes.">
</picture>

The figure measures throughput and memory for one RTX 4080 workload. Read the
[full data-bound Consumer Runtime v1 report](../consumer-runtime-v1.md) for the
matrix, profiler evidence, equivalence gate and hardware limits.

Next: configure a matching PyTorch build and recipe with the
[single-GPU guide](../single-gpu-guide.md), or inspect failure recovery in
[RecoveryBench](../recoverybench/recoverybench-v1.md).
