# Consumer Runtime

The single-GPU runtime supports a conservative dual-model path and a
shared-backbone path that switches standard PEFT roles on one resident base.
Padded trajectory updates improve update throughput without changing the
effective optimizer batch or the strict-OPD freshness contract.

![Measured continuation-time and peak-memory Pareto view for the frozen Consumer Runtime matrix](../consumer-runtime-v1-pareto.svg)

The figure is a systems result for one RTX 4080 workload, not a cross-GPU speed
forecast or a new quality experiment. Read the
[full data-bound Consumer Runtime v1 report](../consumer-runtime-v1.md) for the
matrix, profiler evidence, equivalence gate and hardware limits.

Next: configure a matching PyTorch build and recipe with the
[single-GPU guide](../single-gpu-guide.md), or inspect failure recovery in
[RecoveryBench](../recoverybench/recoverybench-v1.md).
