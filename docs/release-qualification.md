# RTX 4080 release qualification

miniVERL has a version-controlled qualification path for the one machine on
which CUDA behavior is maintainer-measured. Candidate construction runs on a
GitHub-hosted runner; only CUDA qualification runs on the private runner. The
workflow is manual, not continuous GPU CI and not a pull-request required check.

## Two levels

v0.15 adds direct-config PPO/GRPO workflows to the existing full qualification.
The installed candidate must auto-select v4, retain the upstream source bytes,
use explicit runtime bindings and complete run → inspect → resume → handoff
without editing native YAML. The evidence validates both repeated actor passes,
PPO critic counts, exact tensor replay, the IR/rules digests and launch status.

| level | cadence | executed scope |
| --- | --- | --- |
| release smoke | diagnostic use | install the hosted-runner candidate wheel, verify import/CLI origin, run CLI doctor/plan/dry-run, pinned Qwen actor and teacher, one rollout/score/update, PEFT export/reload and CUDA teardown |
| full qualification | every formal release | release smoke, canonical resume workloads, rollout-backend matrix, GRPO/PPO/RM evidence and installed PPO/GRPO product workflows |

The smoke budget is intentionally small and is never substituted for a frozen
full workload or a scientific benchmark. Both levels record runtime
correctness only. Other hardware is unmeasured; distributed verl execution is
`not_tested`.
`gpu-release-smoke` is diagnostic and cannot authorize a formal release. The
release gate requires `candidate-distributions` and `gpu-full-qualification`
from the same successful run and first attempt.

## Repository contract

`.github/workflows/gpu.yml` runs only through `workflow_dispatch`. Its
GitHub-hosted `build-candidate` job creates exactly one wheel and one sdist with
the pinned build toolchain, validates them with Twine, and uploads
`candidate-distributions` with canonical checksums and a strict manifest. The
dependent `[self-hosted, cuda, rtx4080, wsl2]` job downloads that same-run artifact,
validates it before installation, then installs the candidate wheel in a fresh
virtual environment using the [known-good stack](single-gpu-guide.md). It does
not build a distribution. `qualification.json` binds the candidate manifest,
full source SHA, exact wheel hash, workflow run, profile identity, model
revisions, inputs, measured environment and output hashes. It also proves that
both the imported package and `miniverl` executable came from the qualification
environment rather than a checkout or user site.
For a v0.11 full run, the workflow additionally performs three real one-update
profiles, all 24 preregistered cells on each rollout backend, eight policy
refreshes per backend, conformance and teardown. Promotion verifies the exact
source, wheel and environment bindings; speed and memory thresholds; grouped
trajectory identity; reward/advantage completion; and vLLM's
policy-gradient fail-closed boundary. Loose result files cannot claim the
higher level.

For v0.12 and later, promotion also requires a compiled
`verl-rl-v0.9-single-gpu-v1` plan and two Qwen3-0.6B NF4 LoRA GRPO cycles on
the exact candidate wheel. The deterministic target-length formatting reward
must vary within at least one prompt group, produce nonzero advantages, commit
two parameter-changing updates, remain below 14.5 GiB peak reserved memory and
retain the runtime-only scientific scope.

For v0.13 and later, promotion additionally requires the v2 PPO compiler,
two Qwen3-0.6B actor updates, two independent critic updates, nontrivial GAE
advantages and returns, complete actor/critic checkpoints, and tensor-exact
interruption/resume for both roles. The same record exercises a pinned
DistilBERT sequence-classifier reward role with deterministic nonconstant
scores and post-phase CPU offload, plus a locally diagnosed v0.9 handoff bundle.

For v0.14 and later, promotion also requires the installed-package
[PPO/GRPO walkthrough](local-rl-workflow.md), executed from a fresh working
directory outside the checkout. Both cases import packaged upstream-derived
inputs, validate/plan, train, inspect/report, replay from a checkpoint, export
and diagnose the handoff. The gate verifies the candidate wheel hash, compiler
identity, all 25 CLI invocations, exact actor/critic/optimizer tensor replay,
unduplicated records and measured VRAM. The release evidence archive contains
`v014/product-workflows.json` with per-case rewards, losses, time, peak memory,
model revisions and transcript. A dirty-tree rehearsal cannot satisfy this gate.

Every dispatch is single-use. Both jobs reject `GITHUB_RUN_ATTEMPT` values
other than `1`; if infrastructure fails, start a new `workflow_dispatch`
instead of using GitHub's rerun button. This prevents a hosted candidate from
attempt 1 being combined with GPU evidence created by attempt 2.

The release workflow queries GitHub Actions for one successful manual
`gpu.yml` run whose repository, workflow identity and `head_sha` equal the
release target. It accepts only the unexpired candidate and qualification from
that same run, safely extracts both, and verifies the candidate manifest, API
artifact digest when provided, wheel byte hash and qualification bindings. It
also checks each declared evidence file's regular-file type, byte count and
SHA-256. Cross-origin artifact redirects retain ordinary API headers but strip
authorization, proxy authorization and cookies. It publishes the accepted
wheel and sdist without rebuilding them. Future release runs retain the full
qualification record, four principal workload JSON files and a deterministic
subordinate-evidence archive. For v0.11, that archive also contains the three
profile/backend qualification records; v0.12 adds the critic-free RL record and
v0.13 adds the PPO/RM/handoff record.
A committed
JSON file, manual upload, fork run, different workflow or cross-run artifact
pair cannot satisfy this gate.

## Canonical future Release assets

The release-asset builder uses an explicit evidence-role mapping. It never
derives public filenames from internal paths or appends a guessed extension.
The top level is exactly:

```text
dist/<wheel and sdist>
SHA256SUMS
candidate-manifest.json
release-verification.json
qualification.json
qualification-SHA256SUMS
qualification-release-smoke.json
qualification-direct-gkd.json
qualification-pg-k1.json
qualification-smollm2.json
qualification-evidence.tar.gz
qualification-evidence-manifest.json
```

The four principal workload records remain directly inspectable. Adapter
configuration and safetensors, its miniVERL manifest, input prompts, run
summary and version-specific qualification records live once in the
deterministic archive; its manifest binds every member to its semantic role,
byte count and SHA-256. `SHA256SUMS` covers only the wheel
and sdist. `qualification-SHA256SUMS` covers the nine provenance and evidence
assets in canonical order. Hashes prove byte integrity, not code signing or
third-party endorsement.

The historical v0.10.1 assets are immutable and retain the names emitted by the
original flattening step, including duplicated suffixes on some subordinate
files. They are not rewritten or re-uploaded. The canonical layout applies to
future releases only and does not prove distributed verl execution or hardware
beyond the one measured RTX 4080.

Local validation is torch-free:

```bash
python scripts/validate_release_chain.py \
  --candidate-dir path/to/candidate \
  --candidate-manifest path/to/candidate/candidate-manifest.json \
  --qualification path/to/qualification/qualification.json \
  --commit <full-release-sha> \
  --known-good-sha256 <known-good-manifest-sha256> \
  --required-gpu-name "NVIDIA GeForce RTX 4080"
```

## Runner setup and safety

Runner registration is an external maintainer action. Use a dedicated local
account and working directory, apply the labels exactly, disable unattended
access by other repository users, and allow only maintainer-dispatched jobs.
The runner needs repository read access and Actions artifact download/upload;
it does not need PyPI credentials or a publishing environment. Keep it offline
when not qualifying a reviewed commit.
Do not add `pull_request` or `pull_request_target`: model downloads and training
execute repository code on the workstation. Keep the runner application and
GPU driver patched, keep credentials out of the service environment, and
review the exact SHA before dispatch.
Prefer an ephemeral runner for one new dispatch. A rerun is not a recovery
mechanism: let the runner leave, fix the cause, register a fresh runner and
create a fresh dispatch.

The job deletes and recreates its qualification virtual environment, clears
`PYTHONPATH` and `PYTHONHOME`, never uses an editable package, uploads only
portable bounded artifacts, and checks cleanup targets remain under
`GITHUB_WORKSPACE`. Model caches remain runner-local and are not uploaded.
Rotate the runner token after suspected exposure.

The current release runner is Ubuntu under WSL2 with Python 3.12.13 and the
checked `known-good-rtx4080-wsl2-cu130` stack. The older Windows qualification
record remains historical and cannot authorize a current release.

## Measured release state

The first exact-commit full run completed for v0.10.1 on attempt 1:
[GPU qualification 31932226695](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31932226695),
[dry-run 31933844796](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31933844796)
and [tag publication 31934196365](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31934196365).
Future release commits still require their own new exact-SHA, same-run, attempt-1
full qualification; the v0.10.1 record cannot authorize them. This remains a
manual maintainer process, not continuous GPU CI.
