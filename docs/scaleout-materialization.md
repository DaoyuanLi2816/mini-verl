# Materialize a pinned verl handoff

`export-verl` intentionally starts with identity-only student and teacher base
models. Its PEFT adapter, Parquet data, pure-OPD override and provenance are
complete, but `launchable` remains `false` until the exact model snapshots and
the pinned upstream validation are present.

## Offline-first workflow

The simplest offline path copies the two exact commits from the Hugging Face
cache into a regular-file staging tree:

```bash
miniverl export-verl --run runs/my-opd --target-verl v0.8.0 --out scaleout
miniverl bridge materialize scaleout --download --offline
```

Hugging Face cache snapshots commonly contain symlinks. The materializer will
not follow them from an explicitly supplied directory; download mode resolves
cached bytes into regular staging files without network access.

For a separately copied regular-file snapshot, pass both directories. Each
path must either end in `snapshots/<40-character-commit>` or carry a
`miniverl-snapshot.json` binding the recorded model id, revision and file
hashes:

```bash
miniverl bridge materialize scaleout \
  --student-snapshot /snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --teacher-snapshot /snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --offline
```

Remove `--offline` only when downloading the two exact recorded commits is
intended. A branch name such as `main` or a tag is never accepted as the bundle
identity.

The command also requires official verl `v0.8.0` installed from pinned commit
`7aed6b230776f963fa09509c10d9c3a767d1102c`. It merges the exported override
into that exact configuration, validates Parquet and PEFT payloads, loads both
local model/tokenizer snapshots sequentially, runs a tiny CPU forward for each
role and verifies the top-k/tokenizer contract. It launches no Ray worker and
performs no distributed training.

## What is published

Materialization stages a complete copy beside the bundle. Snapshot files must
be regular files; model shard indexes must be closed; configs, tokenizer
vocabularies and safetensors structure must validate. Every copied or merged
file is SHA-256 bound in
`provenance/materialization-manifest.json`, then the bundle-wide
`provenance/SHA256SUMS` is regenerated. Only after all checks pass is the old
directory replaced.

Before success:

```text
recipe/launch.template.sh
launchable: false
```

After success:

```text
recipe/verl-opd-resolved.yaml
recipe/launch.sh
launchable: true
distributed_execution_tested: false
```

Here `launchable: true` means the pinned config and local artifacts are
complete enough to invoke the documented upstream entry point. It is not a
claim that the invocation, a distributed job or miniVERL-to-verl end-to-end
algorithm parity was tested. Recheck the bytes and installed pin in the
current process before use:

```bash
miniverl bridge doctor scaleout \
  --require-verl --require-tokenizer-load --require-adapter-payload
```

## Teacher adapters

If the local run used a teacher adapter that pinned verl cannot consume in the
recorded role, export includes its standard PEFT payload when available and
materialization remains blocked. Merging requires explicit consent:

```bash
miniverl bridge materialize scaleout --download \
  --merge-teacher-adapter
```

The original base is never modified. The merged output is written to a new
teacher snapshot, then reloaded and hashed. Provenance records the base and
adapter identities and hashes, merge software versions, output hashes, and
copied license/notice files. A missing adapter payload or identity mismatch
fails before publication.

## Failure and recovery

Copy, merge, validation and provenance writes happen in a sibling staging
directory. An ordinary exception leaves the existing bundle unchanged and
removes staging files. Final publication uses a same-filesystem directory
rename with in-process rollback. As with any multi-rename update, a power loss
at the final swap is not a cross-platform filesystem transaction; preserve the
source run and exported bundle until the materialized copy is verified.
