# v1 readiness contract

Version 1 is a stability promise, not a feature-count milestone. miniVERL may
be proposed as a v1 candidate only when every required row below is `passed` on
the exact candidate commit. The current development line remains Beta.

## Compatibility commitments

- Stable CLI commands retain their names, option meanings, exit-code class and
  documented JSON shape. A rename or semantic change requires deprecation for
  at least one minor release.
- `miniverl.config.RunConfig`, `miniverl.trainer.OPDTrainer` and documented
  adapter/cache/checkpoint readers remain import-compatible across the v1
  line. Internal coordinators are not public APIs.
- Published compatibility profile identities never drift. A new upstream verl
  version, estimator or compiler contract receives a new profile identity.
- Plan, trajectory, teacher-cache, checkpoint, run-manifest, hardware-record,
  qualification and export-bundle schemas are versioned. Readers either accept
  an older version exactly or reject it with a migration path; writers never
  silently reuse a version number for new semantics.
- Security fixes may tighten acceptance of hostile or ambiguous input without
  a deprecation window. They must not reinterpret an accepted training
  objective.

Exact resume means the documented tensors, progress cursor, policy version,
optimizer/RNG state and trajectory order match for an eligible same-platform,
same-profile run. It does not promise equivalence across different CUDA,
PyTorch, kernels, model revisions, objectives or unsupported distributed
runtimes.

## Candidate gate

| requirement | required state |
| --- | --- |
| stable docs and PyPI quickstart agree with `release-state.yaml` | passed |
| base wheel install remains torch-free | passed |
| known-good CUDA environment is machine-readable and installable | passed |
| release smoke is successful on one RTX 4080 for the exact candidate SHA | passed |
| full qualification covers both profiles, SmolLM2 and exact resume | passed for a v1 candidate |
| public schemas and profile lifecycle policy are documented | passed |
| pre-release gate, build, Twine, sdist, link, visual and privacy checks | passed |
| deprecation and supported-version policy are current | passed |
| working tree, tag target, distributions and attestations share provenance | passed |

The following are explicitly outside v1: arbitrary verl YAML, PPO, GRPO,
critics, general reward pipelines, Ray, FSDP, Megatron, vLLM/SGLang execution,
multi-GPU, multi-node, distributed launch, multimodal models, multiple teachers
and broad cross-hardware qualification. Scale-out artifacts do not prove
distributed execution.

At present miniVERL is **not a v1 candidate**: the exact-SHA qualification
workflow is defined, but its private runner registration and first successful
artifact are an external pending step. No version change follows from this
document alone.

