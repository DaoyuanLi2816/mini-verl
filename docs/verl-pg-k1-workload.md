# RTX 4080 sampled-k1 PG systems workload

This is runtime and semantic-conformance evidence for
`verl-opd-v0.8-single-gpu-pg-k1-v1`, not a task-quality or algorithm
comparison result. The source profile pins verl `v0.8.0` at
`7aed6b230776f963fa09509c10d9c3a767d1102c`.

| Item | Measured value |
| --- | --- |
| Hardware | one NVIDIA RTX 4080, 15.992 GiB |
| Models | Qwen3-0.6B actor / Qwen3-1.7B teacher, pinned NF4 snapshots |
| Workload | 32 distinct prompts, response bound 64, 8 strict updates |
| Target | sampled-token teacher log-probability; `k1` + vanilla policy loss |
| Peak CUDA | 2.3032 GiB allocated / 3.1914 GiB reserved |
| Time to first update | 30.0902 s |
| Steady rollout | 13.4693 generated tokens/s |
| Steady teacher scoring | 893.86 sampled positions/s |
| Steady actor update | 221.0899 positions/s |

The matched run stopped after update 4, reconstructed the trainer from its
checkpoint, and finished at update 8 with byte-identical trajectories, adapter
and optimizer tensors and identical training-state fields (apart from the
deliberately run-specific resolved-config digest). Every update recorded
`rollout policy version == parameter version before update`; each successful
update incremented the parameter version once.

The standard PEFT adapter exported with tokenizer metadata. A separate
checksummed scale-out bundle preserved the sampled-k1 fields without a top-k
requirement; exact student and teacher snapshots were materialized
transactionally, the pinned upstream config merge and sequential CPU
model/data load smoke passed, and the local bundle became `launchable: true`.
No distributed verl job ran, and algorithm-wide parity is not claimed.

The complete machine-readable record is
[`evidence/rtx4080-verl-pg-k1-v1.json`](evidence/rtx4080-verl-pg-k1-v1.json)
(SHA-256 `b549905b2b62b5e51d721ba2932befb7b910284ff1b4016095d341ec232a725f`).
The estimator/scalar/gradient contract is documented in
[ADR 0010](adr/0010-verl-v0.8-pg-k1-contract.md).
