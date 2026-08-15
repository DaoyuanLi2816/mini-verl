# RTX 4080 release qualification

miniVERL has a version-controlled qualification path for the one machine on
which CUDA behavior is maintainer-measured. Candidate construction runs on a
GitHub-hosted runner; only CUDA qualification runs on the private runner. The
workflow is manual, not continuous GPU CI and not a pull-request required check.

## Two levels

| level | cadence | executed scope |
| --- | --- | --- |
| release smoke | every release candidate | install the hosted-runner candidate wheel, verify import/CLI origin, run CLI doctor/plan/dry-run, pinned Qwen actor and teacher, one rollout/score/update, PEFT export/reload and CUDA teardown |
| full qualification | important minor or v1 candidate | release smoke plus the unchanged direct-GKD, sampled-k1 and SmolLM2 canonical workloads, including interruption/resume and their existing export/materialization checks |

The smoke budget is intentionally small and is never substituted for a frozen
full workload or a scientific benchmark. Both levels record runtime
correctness only. Other hardware is unmeasured; distributed verl execution is
`not_tested`.

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

The release workflow queries GitHub Actions for one successful manual
`gpu.yml` run whose repository, workflow identity and `head_sha` equal the
release target. It accepts only the unexpired candidate and qualification from
that same run, safely extracts both, and verifies the candidate manifest, API
artifact digest when provided, wheel byte hash and qualification bindings. It
publishes the accepted wheel and sdist without rebuilding them. A committed
JSON file, manual upload, fork run, different workflow or cross-run artifact
pair cannot satisfy this gate.

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

The job deletes and recreates its qualification virtual environment, clears
`PYTHONPATH` and `PYTHONHOME`, never uses an editable package, uploads only
portable bounded artifacts, and checks cleanup targets remain under
`GITHUB_WORKSPACE`. Model caches remain runner-local and are not uploaded.
Rotate the runner token after suspected exposure.

## Activation state

Repository code, schema, validator and workflows can be complete while runner
registration remains incomplete. Until the first successful exact-commit run
exists, report `blocked_external_setup`; do not write “exact-commit
qualification passed” or “continuous GPU CI”.
