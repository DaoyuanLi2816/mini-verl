# Consumer Runtime

The single-GPU runtime supports a conservative dual-model path and a
shared-backbone path that switches standard PEFT roles on one resident base.
Padded trajectory updates improve update throughput without changing the
effective optimizer batch or the strict-OPD freshness contract.

<picture>
  <source media="(max-width: 900px)" srcset="../consumer-runtime-v1-pareto-mobile.svg">
  <img src="../consumer-runtime-v1-pareto.svg" alt="Measured continuation-time and peak-memory Pareto view for the frozen Consumer Runtime matrix. Batch 4 is the knee for both runtimes: shared backbone reaches 3.48 trajectories per second at 2.23 GiB and dual model reaches 3.87 at 3.04 GiB.">
</picture>

The figure is a systems result for one RTX 4080 workload, not a cross-GPU speed
forecast or a new quality experiment. Read the
[full data-bound Consumer Runtime v1 report](../consumer-runtime-v1.md) for the
matrix, profiler evidence, equivalence gate and hardware limits.

Next: configure a matching PyTorch build and recipe with the
[single-GPU guide](../single-gpu-guide.md), or inspect failure recovery in
[RecoveryBench](../recoverybench/recoverybench-v1.md).
