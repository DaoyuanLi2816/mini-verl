# miniVERL

Single-GPU prototyping for a documented subset of verl-style online
post-training. Develop, diagnose and validate an alignment or distillation
recipe locally, then export standard model, dataset, recipe and provenance
artifacts to a pinned verl release for scale-out.

miniVERL is an independent project. Its verified bridge targets only
`single-gpu-online-distillation-v1` on `verl v0.8.0`; distributed execution is
not tested.

## Start here

- [Why OPD after SFT?](alignment-lab/when-opd-should-follow-sft.md)
- [Choose SFT vs DPO vs OPD](alignment-lab/alignment-lab-v1.md)
- [One-GPU alignment quickstart](single-gpu-guide.md)
- [Shared-backbone dual-adapter runtime](consumer-runtime-v1.md)
- [Batched runtime](benchmarking.md)
- [RecoveryBench](recoverybench/recoverybench-v1.md)
- [Verified verl bridge](verl-bridge.md)
- [90-second verified-bridge demo](verl-bridge-demo.md)
- [Artifact and schema reference](trajectory-schema.md)
- [Community recipes and submissions](community-benchmarks.md)
- [v0.6 launch story and card copy](verl-bridge-launch.md)
