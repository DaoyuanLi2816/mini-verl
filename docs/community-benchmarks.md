# Community recipes and hardware submissions

The version-1 registry contains five maintained recipe records: policy-
conditioned alignment, a preference-teacher distillation template, localized
verifier-gated OPD, tool-policy alignment and RecoveryBench. Every entry records the exact model
revision, GPU/VRAM, wall time, benchmark, artifact hash, measurement status,
compatible miniVERL release and bridge profile. Four initial records bind
maintainer measurements already preserved in this repository. The
preference-teacher entry is explicitly `not_measured`: its linked
aligned-adapter recipe is a scaffold that still names the public SFT teacher,
and users must substitute a pinned preference-trained adapter before claiming
that category. None is presented as external adoption.

Create a privacy-safe unmeasured submission template without installing torch:

```bash
miniverl benchmark --export-community submission.json
```

The schema requires hardware/software provenance and a packaged recipe digest.
Measured submissions additionally require wall time and artifact hashes. The
validator rejects recipe mismatches, credentials, environment references and
local absolute paths. Change `measured_status` only after retaining the source
run and matching its published hashes.

To propose a real result, add the JSON plus any new versioned recipe record in
one pull request. Describe unsupported hardware or software explicitly; a
negative or failed run is useful evidence and should not be removed merely
because it does not improve a benchmark.

## OPD hardware records

Completed OPD runs use a separate, stricter version-1 record:

```bash
miniverl hardware record --run runs/my-opd --out hardware-record.json
miniverl hardware validate hardware-record.json
```

The record binds the profile identity, plan and config digests, GPU/VRAM and
software stack, pinned actor/teacher identities, quantization, prompt/response
bounds, teacher-target mode, logical and physical batches, update count,
measured memory and phase timing, resume state, and artifact hashes. Every
numeric evidence item is labelled `measured`, `estimated`, or `unknown`.
Generation and validation are torch-free and perform no upload. The committed
[JSON Schema](generated/hardware-record-v1.schema.json) is generated from the
same Pydantic contract used by the CLI.

| Record origin | Public status | Required gate |
| --- | --- | --- |
| `maintainer_measured` | measured only after repository review | retained run, hashes, schema/privacy validation and maintainer approval |
| `community_submitted` | unreviewed candidate | validated JSON plus explicit publication consent |
| `not_measured` | documented gap | no numeric value may be presented as measured |

`hardware validate` proves schema and privacy conformance only. It does not
change `review_status`, publish a file or add a row to the public matrix. An
issue comment or unreviewed JSON can never become a measured row automatically.
