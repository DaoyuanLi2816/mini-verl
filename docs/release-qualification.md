# RTX 4080 release qualification

miniVERL has a version-controlled qualification path for the one machine on
which CUDA behavior is maintainer-measured. It is deliberately a manual
self-hosted workflow, not hosted GPU CI and not a pull-request required check.

## Two levels

| level | cadence | executed scope |
| --- | --- | --- |
| release smoke | every release candidate | build wheel, clean known-good install, CLI doctor/plan/dry-run, pinned Qwen actor and teacher, one rollout/score/update, PEFT export/reload, CUDA teardown |
| full qualification | important minor or v1 candidate | release smoke plus the unchanged direct-GKD, sampled-k1 and SmolLM2 canonical workloads, including interruption/resume and their existing export/materialization checks |

The smoke budget is intentionally small and is never substituted for a frozen
full workload or a scientific benchmark. Both levels record runtime
correctness only. Other hardware is unmeasured; distributed verl execution is
`not_tested`.

## Repository contract

`.github/workflows/gpu.yml` runs only through `workflow_dispatch` on labels
`[self-hosted, cuda, rtx4080]`. It builds the exact checkout into a wheel,
installs that wheel in a clean virtual environment using the
[known-good stack](single-gpu-guide.md), and uploads `gpu-release-smoke`.
`qualification.json` binds the full source SHA, wheel hash, profile identity,
model revisions, input and plan hashes, measured environment, output hashes
and explicit executed/skipped/not-applicable states.
For a full run, the workflow additionally validates each canonical result's
exact resume, resource, PEFT and scale-out fields, then promotes that same
record to `full_qualification`; three loose result files cannot claim the
higher level.

The release workflow queries GitHub Actions for a successful manual `gpu.yml`
run whose `head_sha` equals the release SHA. It downloads and validates that
artifact, then rebuilds the distributions and rejects a wheel whose hash is not
the qualified wheel hash. A committed JSON file cannot satisfy this gate.

Local validation is torch-free:

```bash
python scripts/validate_gpu_qualification.py \
  path/to/qualification.json \
  --commit <full-release-sha> \
  --required-gpu-name "NVIDIA GeForce RTX 4080"
```

## Runner setup and safety

Runner registration is an external maintainer action. Use a dedicated local
account and working directory, apply the labels exactly, disable unattended
access by other repository users, and allow only maintainer-dispatched jobs.
Do not add `pull_request` or `pull_request_target`: model downloads and training
execute repository code on the workstation. Keep the runner application and
GPU driver patched, keep credentials out of the service environment, and
review the exact SHA before dispatch.

The job installs into a fresh workspace virtual environment, never uses an
editable package, uploads only portable bounded artifacts, and checks cleanup
targets remain under `GITHUB_WORKSPACE`. Model caches remain runner-local and
are not uploaded. Rotate the runner token after suspected exposure.

## Activation state

Repository code, schema, validator and workflows can be complete while runner
registration remains incomplete. Until the first successful exact-commit run
exists, report `blocked_external_setup`; do not write “exact-commit
qualification passed” or “continuous GPU CI”.
