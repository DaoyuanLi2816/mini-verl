# Compatibility policy

miniVERL versions experiment compilers separately from artifact schemas. A
compiler profile names an exact upstream tag and commit, a closed field-rule
set, the native objective implementation and its export contract. That identity
travels through recipes, plans, trajectories, caches, checkpoints and reports.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| exact | the same value and unit drive the same experiment meaning |
| semantically conformant | the representation differs, while tested algorithm semantics agree |
| locally lowered | physical execution changes without changing the objective |
| informational only | retained for provenance and inactive for this selection |
| distributed only | cluster placement or communication has no one-GPU counterpart |
| not implemented | meaningful on one GPU but absent from the current runtime |
| unsupported | a concrete field/value has no safe versioned rule |

These states deliberately separate algorithm support from physical scale. A
source config may request eight GPUs and two nodes while its GRPO experiment
still lowers to one device; those resource values are reported as
`distributed_only`, not treated as an unknown algorithm.

## verl v0.9 RL compiler

Native Hydra `verl-rl-v0.9-single-gpu-v5`, resolved-file v4 and preserved v1/v2/v3 profiles target
official verl `v0.9.0` at
`483b8a009ba3a97563edee3a19887e4862b8094a`. v1 retains the released
critic-free contract; v2 adds PPO and its value role. v3 binds upstream prompt
minibatch units. v4 adds direct execution, common loss reductions, repeated
actor epochs, prompt filtering, dataset-derived schedules and explicit bindings.
v5 adds pinned config-tree composition, independent shuffled actor/critic epochs,
zero-variance group filtering and synchronous failed-group refill.
Start with the [direct PPO/GRPO workflow](direct-verl-config.md); the
[upstream corpus](verl-compatibility-corpus.md) accounts for complete real configs.

| Surface | Status | Local contract |
| --- | --- | --- |
| PPO/GAE | semantically conformant | independent causal-LM critic plus scalar head, GAE and clipped value updates |
| GRPO | semantically conformant | sample-std group normalization, epsilon placement and masking match v0.9 |
| Dr.GRPO advantage estimator | semantically conformant | GRPO centering without std normalization; v4 preserves the sequence-normalized loss |
| RLOO | semantically conformant | leave-one-out prompt-group baseline |
| REINFORCE++ | semantically conformant | discounted token returns plus masked whitening |
| vanilla dual-clipped policy objective | semantically conformant | token and sequence reductions, clipping and loss gradients match v0.9 |
| grouped `rollout.n` | exact | complete prompt groups and stable sample identities |
| behavior log-probability | semantically conformant | current rollout actor, temperature-scaled, recomputed current policy at update |
| Parquet data and token bounds | exact | source files, key, shuffle, seed and limits drive local loading |
| exact-answer / target-length rewards | exact | reward-bearing Parquet metadata and deterministic scorers |
| HF sequence-classifier reward model | supported | pinned model/tokenizer revisions, deterministic batches and phased offload |
| environment/Python reward providers | supported | local API, plus checksum-approved upstream Python bindings in v4 |
| fixed reference reward KL | locally lowered | v4 uses an independent frozen initial base; older profiles retain explicit adapter roles |
| actor logical mini-batch | semantically conformant | one logical mini-batch, physically microbatched on the GPU |
| TP/PP/DP, nodes, resource pools | distributed only | original values retained; execution uses one process/device |
| actor-loss KL / entropy | semantically conformant | sampled-token reference KL and entropy enter the actor objective |

The native command composes Hydra inputs; v4 still consumes resolved YAML.
Composition success is independent of semantic acceptance, missing
local bindings and exact-model hardware capacity; static acceptance does not
claim execution. Historical importers produce templates for missing choices.
Unknown active semantics and unresolvable interpolation are reported before
execution. Every executable IR is validated with `RunConfig`.

The [machine-readable v1 report](generated/verl-rl-v0.9-compatibility.json) and
[v2 PPO report](generated/verl-rl-v0.9-ppo-compatibility.json) are generated
from shipped examples and checked byte-for-byte in CI. The
[RL runtime guide](verl-rl-runtime.md) covers execution and resume.

## verl v0.8 OPD profiles

The OPD family targets official verl `v0.8.0` at
`7aed6b230776f963fa09509c10d9c3a767d1102c`:

| Profile family | Objective | Status |
| --- | --- | --- |
| direct GKD | `forward_kl_topk`, token mean | measured local runtime |
| sampled-k1 | sampled `k1` plus vanilla policy loss | measured local runtime |
| grouped variants | independent `n > 1` samples | conformance only |
| rewarded sampled-k1 | exact-answer advantage plus teacher advantage | conformance only |

The [generated OPD matrix](generated/verl-opd-v0.8-compatibility.json) and
[field-effect record](generated/verl-opd-v0.8-field-effects.json) bind compiler
rules to observed native plan effects. Existing profile identities and
published evidence retain their historical meaning.

## Artifact and scale-out bridge

OPD export produces standard PEFT, safetensors, tokenizer, Parquet, config and
provenance artifacts. The report keeps these states independent:

- artifact bundle complete;
- upstream config parse passed;
- model/data load smoke passed;
- reward implementation complete;
- launchable;
- distributed execution tested;
- algorithm semantic parity.

Where the legacy label remains, **miniVERL-defined compatibility Level 3**
means the generated bundle and exact-source smoke, not a distributed training
claim. Read the [scale-out contract](verl-opd-scaleout.md) and
[legacy bridge](legacy-verl-bridge.md).

## Stable surfaces

- Package `miniverl`, CLI `miniverl`, documented command names, `RunConfig` and
  the `OPDTrainer` compatibility facade.
- Existing run artifact names and historical trajectory/cache/checkpoint schema
  readers, including the constrained legacy cache reader.
- Historical protocol-v1 and verifier-v1 behavior; new protocol versions do
  not rewrite frozen evidence.
- Existing compiler profile identities and upstream pins.

Minor releases may add a new versioned profile or tighten unsafe input
validation. Deprecations warn for at least one minor release. For exact resume,
retain the package version and complete run directory; see
[reproducibility](reproducibility.md).
