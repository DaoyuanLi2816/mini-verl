# miniVERL v0.5.0 launch copy

This archived launch copy is tied to the v0.5.0 evidence bundle. Update its
release URL only when republishing against the exact tag.

## GitHub Release summary

miniVERL v0.5.0 adds a one-GPU Alignment Lab for comparing continued SFT, DPO,
offline distillation, standard OPD and verifier-gated OPD from the same SFT
checkpoint. It includes `miniverl align`, an uncertainty-aware `miniverl
pilot`, four runnable alignment recipes, versioned deterministic policy tests,
privacy-safe Alignment Cards and a three-seed Qwen3-0.6B result.

The result is deliberately negative: the SFT checkpoint already achieved 100%
alignment and retained tool utility, and no continuation method improved it.
The pilot therefore recommends turning online teacher queries off for this
recipe. Completed regressions and all scientific caveats remain visible.

## Short post

miniVERL v0.5.0: online alignment and distillation on one GPU.

Six methods, one SFT start, three preregistered seeds—and a useful negative
result. The start was already at 100% alignment + utility, so the pilot says:
do not pay for OPD here.

Includes `align`, `pilot`, Alignment Cards, verifier-gated OPD, full task-level
evidence, report and reproducible demo. One model/suite/GPU; no broad safety or
OPD-superiority claim.

## Hacker News / forum title

miniVERL v0.5: a one-GPU alignment lab whose pilot can tell you not to run OPD

## Repository social metadata

- Title: `miniVERL — one-GPU alignment and distillation lab`
- Description: `Compare SFT, DPO, offline KD and OPD from one checkpoint; retain policy quality, utility, query and cost evidence.`
- Image: `docs/alignment-lab/social-preview.svg`

No adoption, user-count, star-count, production-readiness or universal safety
claim should be added without new public evidence.
