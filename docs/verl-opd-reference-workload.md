# RTX 4080 OPD developer workload

This measured workload asks a systems question: can the documented
`verl-opd-v0.8-single-gpu-v1` path sustain several useful-size rollout,
teacher-scoring and update cycles, survive an interruption, and stay well
inside a 16 GiB consumer-GPU envelope? It is **not** a task-quality or alignment
benchmark.

<picture>
  <source media="(max-width: 600px)" srcset="../verl-opd-reference-workload-mobile.svg">
  <img src="../verl-opd-reference-workload.svg" alt="Three aligned panels show median phase time, labelled throughput and peak reserved VRAM for the RTX 4080 developer workload. The run consumed 32 distinct prompts, completed eight updates at 3.1914 GiB peak reserved VRAM, and had no OOM downshifts.">
</picture>

## Measured recipe

| Field | Exact value |
| --- | --- |
| GPU | 1× NVIDIA GeForce RTX 4080, 15.992 GiB |
| Student | Qwen3-0.6B, pinned commit, NF4 + LoRA r8/alpha16 |
| Teacher | Qwen3-1.7B, pinned commit, NF4 |
| Data | 64 distinct structured prompts; first 32 consumed |
| Bounds | 128 prompt tokens, 64 response tokens |
| Logical / rollout / update batch | 4 / 4 / 1 |
| Objective | reward-free `forward_kl_topk`, top-k 32, token mean |
| Schedule | 8 current-policy rollout/scoring/update cycles |

The uninterrupted run reached its first rollout in 19.1611 seconds and its
first update in 22.2473 seconds, including 9.3293 seconds of cold construction.
Across cycles 2–8, median phase times were 9.8682 seconds for rollout, 0.4864
seconds for teacher scoring and 2.3391 seconds for the actor update. Median
rates were 25.94 rollout tokens/s, 526.31 teacher-scored positions/s and 109.44
update positions/s.

Peak allocated/reserved VRAM was 2.3033/3.1914 GiB. The runtime retained
physical rollout batch 4 throughout, with zero generation downshifts and zero
projection-chunk OOM retries. The final teacher cache was 90,727 bytes, the
checkpoint 27,649,609 bytes, the exported PEFT adapter 9,213,148 bytes and the
complete uninterrupted run 65,065,202 bytes.

## Interruption and resume

A matched execution stopped after update 4, wrote a transactional checkpoint,
constructed a fresh trainer, loaded in 8.3470 seconds and completed updates
5–8. The resumed run consumed the same ordered 32 prompts. Its trajectories,
adapter tensors and optimizer tensors were byte-identical to the uninterrupted
run, and every training-state field matched. Only `resolved_config_digest`
differs by design because it binds the distinct run id.

The complete paired workload took 243.85 seconds—0.068 GPU hours—and stayed
within the preregistered 14.5 GiB and 4 GPU-hour limits. The machine-readable
record is [`rtx4080-verl-opd-developer-v1.json`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/benchmarks/results/rtx4080-verl-opd-developer-v1.json),
and [`run_verl_opd_reference_workload.py`](https://github.com/DaoyuanLi2816/mini-verl/blob/main/scripts/run_verl_opd_reference_workload.py)
reconstructs the dataset, plan, uninterrupted run and resumed run.

## Scope

No reward, task correctness, alignment, preference, safety or method-comparison
endpoint was evaluated. This record does not show that OPD beats SFT, DPO or
KD, and it does not show that a distributed verl job ran. It validates one
Qwen3, one-GPU, forward-top-k compatibility profile.
