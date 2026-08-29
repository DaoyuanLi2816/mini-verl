# v1 readiness contract

Version 1 marks a reviewed stability promise across the public surface. Release
qualification supplies the binary evidence; the contracts below define the
additional API, artifact and maintenance commitments.

## Stability contract

- Stable CLI commands retain their names, option meanings, exit-code class and
  documented JSON shape. A rename or semantic change requires deprecation for
  at least one minor release.
- `miniverl.config.RunConfig`, `miniverl.trainer.OPDTrainer` and documented
  adapter, cache and checkpoint readers remain import-compatible across the v1
  line. Internal coordinators are not public APIs.
- Published compatibility profile identities never drift. A new upstream verl
  version, estimator or compiler contract receives a new profile identity.
- Plan, trajectory, teacher-cache, checkpoint, run-manifest, hardware-record,
  qualification and export-bundle schemas are versioned. Readers accept an
  older version exactly or reject it with a migration path; writers never
  silently reuse a version number for new semantics.
- Security fixes may tighten acceptance of hostile or ambiguous input without
  a deprecation window, but must not reinterpret an accepted objective.

Exact resume means the documented tensors, progress cursor, policy version,
optimizer/RNG state and trajectory order match for an eligible same-platform,
same-profile run. It does not promise equivalence across different CUDA,
PyTorch, kernels, model revisions, objectives or unsupported distributed
runtimes.

## Evidence achieved in v0.10.1

The following evidence **passed in v0.10.1** on release commit
[`2364e9b8e8b550f44d1b66a77fc2d407b76b05b5`](https://github.com/DaoyuanLi2816/mini-verl/commit/2364e9b8e8b550f44d1b66a77fc2d407b76b05b5):

| evidence | verified record |
| --- | --- |
| hosted-built candidate wheel and full qualification from the same first-attempt run on one RTX 4080 | [GPU qualification run 31932226695](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31932226695) |
| direct GKD, sampled-k1, SmolLM2, PEFT reload and exact interruption/resume | the full qualification record from run 31932226695 |
| non-publishing reuse of those exact bytes without rebuilding | [release dry-run 31933844796](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31933844796) |
| OIDC publication, public file hashes and attestations, clean install and GitHub Release creation | [tag release run 31934196365](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31934196365) |
| immutable release identity | [`v0.10.1` tag](https://github.com/DaoyuanLi2816/mini-verl/tree/v0.10.1), [GitHub Release](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.10.1), [PyPI](https://pypi.org/project/miniverl/0.10.1/) |
| base wheel remains torch-free; stable docs and PyPI quickstart agree | release gate and clean public-install jobs in run 31934196365 |

The published wheel SHA-256 is
`fa5cdef1b6d0602ead7e90f6f51a024aa67c2f10d3f6c9d0507c3bbc3f1c82e8`;
the sdist SHA-256 is
`743cf17f365710bf3ddbe8c5599c6a8daf0afa82779344b449f7f1e93cd2d4f92`.
This is a manual maintainer qualification, not continuous GPU CI. It establishes
runtime correctness only on the measured stack, not task quality or broad
consumer-GPU coverage.

## Remaining v1 gates

| gate | current state | reviewable completion condition |
| --- | --- | --- |
| v1 public API baseline | **not yet satisfied** | Freeze the intended CLI, Python imports, JSON outputs and artifact schemas in a machine-readable baseline; gate compatibility in CI while excluding internal coordinators. |
| backward-compatibility evidence | **not yet satisfied** | Across at least one later stable release, execute the listed old plan, cache, checkpoint, adapter/export and other promised reader paths; unsupported distributed formats remain out of scope. |
| independent reproduction or explicit v1 hardware scope | **not yet satisfied** | Reproduce on a machine independent of the current workstation before a broad consumer-GPU claim, or approve a dedicated v1 scope decision limited to the maintainer-qualified stack. Untested GPUs are not evidence. |
| dedicated v1 candidate decision | **not yet satisfied** | Review a dedicated PR that updates classifiers, versioning, support policy and migration notes after all chosen gates pass. This documentation repair cannot trigger it. |

miniVERL is therefore **not a v1 candidate**. PPO, GRPO, critics, general reward
pipelines, arbitrary verl YAML, Ray, FSDP, Megatron, vLLM/SGLang execution,
multi-GPU, multi-node and distributed launch remain outside the current product
boundary. Scale-out artifacts do not prove distributed execution.
