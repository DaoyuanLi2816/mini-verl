# From one GPU to a verified scale-out handoff

miniVERL v0.6 adds a narrow, tested bridge to official verl `v0.8.0`. The goal
is not to imitate a distributed runtime on a laptop. It is to make the boundary
between local scientific diagnosis and later scale-out explicit, standard and
reviewable.

## Flagship article

Online post-training has two distinct engineering regimes. On one GPU, a small
team can inspect every rollout, token mask, teacher query, update and evaluation
decision. At scale, orchestration, sharding and high-throughput generation are
the hard parts. Treating those regimes as interchangeable produces either a
local system that is impossible to audit or a “compatible” exporter that
quietly changes algorithm semantics.

miniVERL keeps the local regime small: actor → rollout → teacher/reference or
reward → update → evaluation, in one process on one CUDA device. v0.6 then adds
four explicit compatibility levels. Level 1 exchanges standard Hugging Face,
PEFT, safetensors, tokenizer and Parquet artifacts. Level 2 imports exactly 14
documented fields from one fail-closed profile. Level 3 generates a checksummed
bundle for official verl `v0.8.0` at commit
`7aed6b230776f963fa09509c10d9c3a767d1102c`.

The importer rejects critic, PPO/GRPO, Ray resources, FSDP/Megatron placement,
vLLM/SGLang placement, asynchronous rollout and unknown fields. The dataset
converter preserves chat messages, ground truth, extension data, rejection
reasons and hashes without silently truncating. The exporter writes standard
adapter, tokenizer and Parquet files, a pinned override recipe, a safe reward
scaffold, source manifests and `SHA256SUMS`. `miniverl bridge doctor` checks the
whole handoff before the user chooses to launch anything.

The release smoke installed the exact official source under Python 3.12. It
parsed the official and exported OmegaConf shapes; loaded a standard PEFT LoRA
config, safetensors header and both Parquet splits; imported the reward
scaffold; and verified privacy plus every artifact hash. It did not install or
run Ray, FSDP/Megatron or vLLM/SGLang. Distributed execution is therefore
recorded as **not tested**, not implied by the Level-3 label.

This bridge also preserves miniVERL's negative results. RecoveryBench did not
show a general fresh-state advantage, and the Alignment Lab began from a
saturated SFT policy whose best decision was to avoid unnecessary OPD cost.
Scale-out cannot repair an unjustified experimental question; it should follow
the diagnosis, not replace it.

## Release announcement

miniVERL v0.6.0 is a single-GPU runtime for a documented subset of verl-style
online post-training. It adds a verified bridge to pinned official verl
`v0.8.0`: a fail-closed config importer, bidirectional Parquet conversion,
standard PEFT/safetensors export, `bridge doctor`, exact-source compatibility
smoke, five evidence-bound recipe records and a community hardware submission
format.

Scope matters: this is one named profile, not generic verl YAML support; the
smoke validates the handoff but does not run a distributed job. Existing
Alignment Lab and RecoveryBench negative results remain first-class evidence.

## Short post

miniVERL v0.6: develop and diagnose online post-training on one GPU, then
export a checksummed standard-artifact bundle to a pinned subset of verl
v0.8.0. Exact config whitelist, Parquet round trip, PEFT/safetensors checks and
`bridge doctor`; distributed execution explicitly not tested.

## Hugging Face card addenda

Use these blocks on the corresponding adapter cards without changing immutable
historical revisions. A card update creates a new revision and must name it.

### Common Qwen3-0.6B SFT start

> Standard PEFT adapter used as the common starting policy in miniVERL
> Alignment Lab v1. miniVERL v0.6 can include this adapter in a checksummed
> `single-gpu-online-distillation-v1` bundle for pinned verl v0.8.0. The bridge
> validates artifact structure; no distributed training run is claimed.

### Qwen3-1.7B protocol teacher

> Frozen protocol-qualified teacher adapter for the historical calculator
> study. Its public revision remains immutable. miniVERL v0.6 exports standard
> PEFT/safetensors artifacts but does not reinterpret teacher logits as PPO
> reference log-probabilities.

### Qwen3-0.6B consumer-runtime teacher

> Frozen adapter used in the v0.4 one-GPU shared-backbone systems study.
> miniVERL v0.6 can validate the standard adapter surface for a pinned verl
> handoff; the recorded v0.4 throughput and VRAM numbers remain scoped to the
> original RTX 4080 experiment.

## Social metadata

- Title: `miniVERL — one GPU to a verified verl handoff`
- Description: `Diagnose SFT, DPO, KD and OPD locally; export standard artifacts to one pinned verl v0.8.0 profile.`
- Image: `social-preview-v0.6.svg`

Do not add adoption, production-readiness, distributed-parity or generic verl
compatibility claims without new public evidence.
