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
