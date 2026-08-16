# RTX 4080 release qualification

miniVERL has a version-controlled qualification path for the one machine on
which CUDA behavior is maintainer-measured. Candidate construction runs on a
GitHub-hosted runner; only CUDA qualification runs on the private runner. The
workflow is manual, not continuous GPU CI and not a pull-request required check.

## Two levels

| level | cadence | executed scope |
| --- | --- | --- |
| release smoke | diagnostic use | install the hosted-runner candidate wheel, verify import/CLI origin, run CLI doctor/plan/dry-run, pinned Qwen actor and teacher, one rollout/score/update, PEFT export/reload and CUDA teardown |
| full qualification | every formal release | release smoke plus the unchanged direct-GKD, sampled-k1 and SmolLM2 canonical workloads, including interruption/resume and their existing export/materialization checks |

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
dependent `[self-hosted, cuda, rtx4080]` job downloads that same-run artifact,
validates it before installation, then installs the candidate wheel in a fresh
virtual environment using the [known-good stack](single-gpu-guide.md). It does
not build a distribution. `qualification.json` binds the candidate manifest,
full source SHA, exact wheel hash, workflow run, profile identity, model
revisions, inputs, measured environment and output hashes. It also proves that
both the imported package and `miniverl` executable came from the qualification
environment rather than a checkout or user site.
For a full run, the workflow additionally validates each canonical result's
exact resume, resource, PEFT and scale-out fields, then promotes that same
record to `full_qualification`; three loose result files cannot claim the
higher level.

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
subordinate-evidence archive in the canonical layout below. A committed
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
configuration and safetensors, its miniVERL manifest, input prompts and the run
summary live once in the deterministic archive; its manifest binds every member
to its semantic role, byte count and SHA-256. `SHA256SUMS` covers only the wheel
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

## Measured release state

The first exact-commit full run completed for v0.10.1 on attempt 1:
[GPU qualification 31932226695](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31932226695),
[dry-run 31933844796](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31933844796)
and [tag publication 31934196365](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31934196365).
Future release commits still require their own new exact-SHA, same-run, attempt-1
full qualification; the v0.10.1 record cannot authorize them. This remains a
manual maintainer process, not continuous GPU CI.
