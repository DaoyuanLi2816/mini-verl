# verl bridge: portable artifacts, bounded semantics

miniVERL is an independent project; no endorsement by the verl project is
implied. The bridge targets the documented
`single-gpu-online-distillation-v1` profile subset of
[`verl v0.8.0`](https://github.com/verl-project/verl/tree/v0.8.0), pinned to
commit `7aed6b230776f963fa09509c10d9c3a767d1102c` (`7aed6b23`). It is
**miniVERL-defined compatibility Level 3**, not full verl compatibility.

<picture class="bridge-architecture">
  <source media="(max-width: 600px)" srcset="../verl-bridge-architecture-mobile.svg">
  <img src="../verl-bridge-architecture.svg" alt="Three verified bridge layers—miniVERL local runtime, a portable artifact bundle, and a pinned upstream parse/load smoke—followed by a dashed arrow to distributed execution marked NOT TESTED.">
</picture>

The solid arrows cover local artifact production, the portable bundle, and the
pinned parse/load smoke. The dashed arrow is deliberate: no Ray, FSDP, vLLM or
distributed verl job ran, and no miniVERL-OPD-to-verl-PPO semantic parity is
claimed.

## Compatibility state

| State | Current value | Meaning |
| --- | --- | --- |
| `artifact_complete` | `false` in a new bundle | Exact student and teacher base snapshots are still required. |
| `config_semantics_supported` | `true` for a compatible OPD run | The source is the bounded pure-GKD profile, not a PPO reinterpretation. |
| `student_artifact_loadable` | `false` in a new bundle | PEFT is present, but the exact base snapshot is not yet bundled. |
| `teacher_artifact_loadable` | `false` in a new bundle | Teacher identity is preserved; the exact snapshot is not bundled. |
| `dataset_loadable` | separate check | Every exported Parquet footer and required column is checked. |
| `upstream_parse_passed` | `false` in a new bundle | Set only when doctor recomputes a merge under the exact installed pin. |
| `upstream_tiny_smoke_passed` | `false` in a new bundle | Materialization performs the bounded local model/tokenizer smoke. |
| `launchable` | `false` | Exact student/teacher snapshots are not materialized in the bundle. |
| `distributed_execution_tested` | `false` | No distributed job ran. |
| `algorithm_semantic_parity` | `false` | Conformance is scoped to documented config/loss behavior, not an end-to-end distributed algorithm. |

The committed [pinned smoke record](generated/verl-bridge-smoke.json) verifies a
specific artifact-only upstream parse/load exercise. It remains separate from
the readiness state of a newly exported bundle and from any execution claim.

## Import the executable OPD v2 profile

The v2 importer consumes the same typed source used by `plan` and `run`; it
does not require a `ToolEnvironment` and does not invent a reward:

```bash
miniverl import-verl \
  --profile verl-opd-v0.8-single-gpu-v1 \
  --config verl-opd.yaml \
  --set 'data.train_files=["data/train.parquet"]' \
  --out local-opd.yaml
```

`local-opd.yaml` is a canonical, round-trippable input to `miniverl run`.
`local-opd.import-report.json` records every source-field classification and
both source/output digests. PG OPD, task rewards, KL penalties, `n>1`,
multi-teacher and distributed semantics remain hard errors.

## Legacy environment-profile import

`import-verl` accepts the documented, resolved field subset—not arbitrary
Hydra/OmegaConf or verl YAML. With only a source profile, it writes
`imported.import-report.json` and a non-executable `imported.template.yaml`:

```bash
miniverl import-verl resolved-verl.yaml \
  --profile single-gpu-online-distillation-v1 \
  --target-verl v0.8.0 \
  --out recipes/imported.yaml
```

The status is `needs_user_input` until the source or command determines the
training environment, qualified teacher, objective and schedule interpretation.
Parquet paths never silently select the calculator environment, and a same-base
standard teacher without a distinct model or adapter is never invented.

To deliberately produce a runnable recipe, supply the missing contract:

```bash
miniverl import-verl resolved-verl.yaml \
  --profile single-gpu-online-distillation-v1 \
  --target-verl v0.8.0 \
  --environment jsonnav \
  --teacher-model Qwen/Qwen3-1.7B \
  --loss-profile topk-tail-reverse-kl \
  --schedule-mapping epochs-as-cycles \
  --out recipes/imported.yaml
```

The explicit schedule option acknowledges that verl epochs/save/test frequency
units are not proven equivalent to miniVERL cycles. Every source field is
classified as `exact`, `derived`, `informational_only`,
`requires_user_confirmation` or `unsupported`. In particular:

| Source field | Classification | Treatment |
| --- | --- | --- |
| `data.train_files`, `data.val_files`, `data.prompt_key` | `informational_only` | Recorded in the report; never substituted for a `ToolEnvironment`. |
| `data.max_response_length` | `exact` | Copied to the per-turn response bound. |
| `data.max_prompt_length` | `derived` | Combined with response length for miniVERL's total trajectory bound. |
| optimizer learning rate and seed | `exact` | Copied after finite numeric validation. |
| `trainer.total_epochs`, `save_freq`, `test_freq` | `requires_user_confirmation` | Copied only after the explicit schedule mapping. |
| algorithm, distributed or unknown fields | `unsupported` | Rejected with a report. |

Finite scientific-notation strings such as `1e-5` are accepted; NaN and
infinity are rejected.

### Unresolved interpolation is never executed

One recursive audit walks strings, lists, tuples and mappings, and runs at
three boundaries: the source fields, the explicit command-line choices, and the
generated recipe immediately before publication. Detection is conservative—any
`${` in a reachable string is a finding, including unterminated and nested
forms. miniVERL never resolves an interpolation for you, so `${oc.env:TOKEN}`
is an input defect rather than a lookup.

An `exact`, `derived` or `requires_user_confirmation` field carrying `${...}`
fails closed and writes a rejection report. An `informational_only` field may
stay unresolved, but the report labels it and it never reaches executable
output:

```json
"data.train_files": {
  "classification": "informational_only",
  "resolution_status": "unresolved_informational_only"
}
```

An accepted recipe is guaranteed to contain no interpolation token, and the
report records `interpolation_audit.runnable_output_clean: true`.

### Outputs are stem-specific and transactional

One invocation owns one `--out` stem. For `--out recipes/foo.yaml` the only
files published are:

```text
recipes/foo.yaml              # or foo.template.yaml, never both as current outputs
recipes/foo.import-report.json
```

Dataset conversion behaves the same way, keyed on the requested Parquet path:
`train.parquet`, `train.parquet.report.json` and, when extensions exist,
`train.parquet.miniverl.json`.

Each invocation takes an exclusive per-stem reservation, refuses to start if
any intended output path already exists, stages every file in a temporary
sibling directory, and publishes the set with same-filesystem renames.

The guarantee is **transactional publication with in-process rollback**: if any
step raises, the previous family is restored and nothing partial is left
behind, so a report from one invocation cannot be paired with a recipe,
template, Parquet or sidecar from another. It is not multi-file crash
atomicity. A `kill -9`, a kernel panic or a power loss between two renames can
still leave a mixed family on disk; recovering from that would need a versioned
output directory behind a single atomically switched pointer, which miniVERL
does not implement. Re-running the invocation with `--overwrite` republishes a
coherent family.

An input file may never also be an output file. `import-verl` and
`convert-dataset` compare the source against every intended output — including
symlink, hard-link, relative and case-insensitive aliases — and refuse before
taking the reservation. `--overwrite` replaces a previous *output* family; it
never authorizes overwriting or deleting an input. There is no in-place mode.

Supplying `--out` does not imply replacement; pass `--overwrite` to replace an
existing family:

```bash
miniverl import-verl resolved-verl.yaml \
  --profile single-gpu-online-distillation-v1 \
  --target-verl v0.8.0 \
  --environment jsonnav \
  --teacher-model Qwen/Qwen3-1.7B \
  --loss-profile topk-tail-reverse-kl \
  --schedule-mapping epochs-as-cycles \
  --out recipes/imported.yaml \
  --overwrite
```

Every runnable output passes `RunConfig` validation before publication.

## Export a pure-OPD portable bundle

```bash
miniverl export-verl --run runs/<run-id> \
  --target-verl v0.8.0 \
  --out exports/<bundle>

miniverl bridge materialize exports/<bundle> --download --offline
miniverl bridge doctor exports/<bundle> --require-verl
```

For a compatible `verl-opd-v0.8-single-gpu-v1` run, the bundle contains:

```text
model/       student PEFT adapter, tokenizer metadata, base-model identity
teacher/     teacher identity and adapter/materialization requirements
data/        original train/validation Parquet bytes, without row reordering
recipe/      verl-opd-overrides.yaml, launch.template.sh, REQUIRED_VERL.txt
provenance/  source config, compiled plan, source manifest,
             compatibility-report.json, SHA256SUMS
README.md
```

The OPD override explicitly enables distillation, selects
`forward_kl_topk`, disables policy-gradient/task-reward/KL-reward paths and
preserves supported source data, optimizer, rollout and schedule values. Pure
OPD has no reward scaffold. A same-base teacher adapter is recorded but blocks
launch until it is explicitly merged/materialized as a teacher snapshot that
the pinned upstream can consume.

`bridge materialize` resolves the two immutable model commits, copies only a
preflighted regular-file tree, validates model/tokenizer/PEFT/data inputs under
the exact installed verl pin and transactionally publishes a checksummed
`launch.sh`. See [materialization](scaleout-materialization.md).

`bridge doctor` verifies pins, adapter structure, tokenizer state, Parquet
schema, pure-OPD override structure, privacy scopes and hashes. An `ok` verdict
means those local artifact checks passed; it does not mean launchable or
distributed-tested. `launch.template.sh` refuses to proceed without both exact
base snapshots and never emits an unverified distributed launch command. On a
materialized bundle, `doctor --require-verl` recomputes launchability in the
current process; it does not merely trust the bundle's compatibility report.

Historical `single-gpu-online-distillation-v1` runs continue to export the
legacy PPO/reward scaffold for compatibility. That output remains explicitly
non-launchable and is not relabelled as OPD.

### Tokenizer verification levels

Filenames and digests are not a compatibility check, so `tokenizer_identity`
reports how far verification actually got:

| Level | Meaning |
| --- | --- |
| `not_present` | The bundle carries no tokenizer file. |
| `metadata_only` | Files exist but the vocabulary is missing, or no local load was performed. Missing components are named. |
| `loadable_local_snapshot` | `AutoTokenizer.from_pretrained(..., local_files_only=True, trust_remote_code=False)` succeeded. |
| `structural_identity_verified` | It loaded *and* its versioned structural digest, vocabulary size and special tokens match the identity recorded by the source run. |

Loading never contacts the network and never executes remote code. A structural
mismatch fails closed. To require a real load rather than accept metadata:

```bash
miniverl bridge doctor exports/<bundle> --require-tokenizer-load
```

The committed pinned smoke record predates these levels: its bundle ships only
`tokenizer_config.json`, which is `metadata_only` under the current check.

### Reward code is inspected, never executed

A bundle is untrusted input. `bridge doctor` parses
`reward/reward_or_verifier_scaffold.py` with `ast.parse` and verifies the
interface statically; it never imports the module, so a bundle cannot act
merely by being diagnosed:

| Level | Meaning |
| --- | --- |
| `not_present` | No scaffold file, or it is unreadable. |
| `syntax_valid` | It parses as Python, but the interface check failed. |
| `interface_shape_verified` | A top-level `compute_score(data_source, solution_str, ground_truth, extra_info=None)` exists, is synchronous, is not bound by assignment, and no definition-time expression forbidden by this policy was found. |
| `trusted_dynamic_import_verified` | The module was actually imported. Reached only through an explicit opt-in. |

The default path stops at `interface_shape_verified`. The level is named for
what it proves: the interface has the expected *shape*. It is not a statement
that the file is safe to import.

Importing a module runs more than its top-level statements, so the check covers
every definition-time position:

| Position | Example |
| --- | --- |
| Top-level statements | `exploit()`, non-literal assignments, loops |
| Class bases | `class Hidden(exploit())` |
| Class keywords | `class Hidden(object, metaclass=exploit())` |
| Annotations | `def compute_score(data_source: exploit())`, `-> exploit()`, `VALUE: exploit() = 1` |
| Type parameters | Python 3.12 bounds and defaults |
| Decorators and defaults | `@exploit()`, `extra_info=exploit()` |

Ordinary type annotations and base classes are unaffected: only expressions
that would actually evaluate — calls, lambdas, comprehensions, `await`, walrus
— are rejected.

The signature contract is enforced including keyword-only parameters, so a
required keyword-only `extra_info` is refused: verl calls `compute_score` with
three positional arguments and that signature would raise `TypeError`.

Inspection is bounded — source bytes, AST node count, AST depth and the number
of reported findings — so a hostile scaffold produces a bounded diagnostic
rather than exhausting the process that asked for a diagnosis. Imports are
listed under `imports_present` with `import_runtime_safety: not_verified`,
because this check never runs them and an imported third-party module can do
anything; a relative, bundle-local import is refused outright.

What this proves is narrow: the interface is present and no forbidden
definition-time expression exists. It proves nothing about whether the reward
logic is correct, whether imported modules are side-effect free, or whether the
file is safe to run later.

If you produced the bundle yourself and want the historical behaviour:

```bash
miniverl bridge doctor exports/<bundle> --trust-and-import-reward-code
```

This executes the bundle's Python in your process with your privileges. It
prints a warning first and reports `untrusted_code_executed: true`. A
subprocess would not be a security sandbox either, so none is claimed.

### What the bundle claims versus what was recomputed

`provenance/SHA256SUMS` lives inside the bundle it describes. Anyone who edits
`compatibility-report.json` can regenerate it, so agreement between them proves
internal consistency and nothing else. miniVERL implements no signature or
transparency-log verification, and does not pretend otherwise.

The diagnosis therefore reports three separate things:

| Field | Meaning |
| --- | --- |
| `bundle_declared_claims` | Copied from the bundle. Events this run did not observe. |
| `locally_recomputed_checks` | Performed in this process against the bytes on disk. |
| `provenance_trust` | `unsigned_self_consistent` at best; `signature_verification: not_available`. |

Historical smoke results, distributed execution and algorithm parity can only
ever be *declared*: no doctor run launches a job or compares algorithms, so the
top-level `distributed_execution_tested` and `algorithm_semantic_parity` flags
are always `false` regardless of what a bundle asserts. Checksum consistency,
config structure, the pinned requirement file, tokenizer load, adapter
structure, Parquet schema, the reward interface and the metadata privacy
heuristic are recomputed every time.

`--require-verl` recomputes the upstream check rather than trusting a record of
it: it loads the installed pinned verl's generated PPO config, parses the
bundle's `verl-overrides.yaml` and performs the structured merge in this
process. It still launches nothing.

### Adapter weights are validated past the header

`adapter_model.safetensors` is checked structurally, not just parsed:

| Level | Meaning |
| --- | --- |
| `not_present` | The file is absent. |
| `header_only` | The header was read but rejected: bad dtype, impossible shape/byte arithmetic, unordered, overlapping, gapped or trailing offsets, or a payload shorter than the header declares. |
| `payload_structure_validated` | Offsets are contiguous and cover the data segment exactly. |
| `tensor_materialization_validated` | Every tensor also resolved through the official `safetensors` reader. |

A header-only result is never called loadable. To require a real payload:

```bash
miniverl bridge doctor exports/<bundle> --require-adapter-payload
```

The structural pass needs no optional dependency, so a torch-free install still
reaches `payload_structure_validated` and reports
`official_reader_status: dependency_missing` rather than pretending the file is
broken. `--require-adapter-payload` demands the strongest level, so it is *not*
satisfied when the official reader is unavailable — install the `[bridge]` extra
to use it.

### Privacy is reported per inspection scope

The default run reads portable metadata files only. It therefore reports three
independent statuses and never widens one into another:

```text
portable_metadata_privacy: passed | failed
dataset_content_privacy:   not_inspected
model_weight_privacy:      not_inspected
```

`not_inspected` never means `passed`. An optional bounded scan inspects
string-like Parquet fields for URL userinfo, private-key blocks, access-key
ids, bearer tokens, credential assignments, absolute local paths and your own
sentinels:

```bash
miniverl bridge doctor exports/<bundle> \
  --scan-dataset-text --sentinel "internal-project-name"
```

It is a heuristic detector, not de-identification proof. It reports only the
detector category, split, column and row index—never the matched text—and never
reads `.safetensors` as text. Model-weight privacy stays `not_inspected`
because no meaningful check exists for it.

The bounds are enforced *while reading*, not after. Row groups are pulled one
at a time through `ParquetFile.iter_batches`, restricted to columns whose Arrow
type can contain a string, and decoding stops the moment `max_rows` or
`max_bytes` is reached; files past the bound contribute their footer row count
and are never decoded. The report states `files_total`, `files_inspected`,
`row_groups_read`, `rows_scanned`, `rows_total`, `bytes_scanned` and whether
the scope was `full` or `sampled`, so a sampled result is visibly sampled.
Schema validation reads the Parquet footer only.

### Dataset conversion is complete-or-nothing

`convert-dataset` is lossless for the rows it accepts and refuses to quietly
drop the rest. One invalid row fails the whole conversion:

```bash
miniverl convert-dataset train.parquet --from verl-parquet --out out.parquet \
  --allow-rejected-rows
```

Only with that flag does it publish a partial dataset, and the report then says
so: `complete_dataset_conversion: false`, `lossless_for_accepted_rows: true`,
plus the output-row-to-source-row index map so a dropped row stays traceable.

A row can carry miniVERL extension data in `miniverl_extensions`, the
conversion sidecar and `extra_info.miniverl`. Identical content in several
places is accepted and recorded as a deduplication; content that disagrees
fails closed, naming the row index and the source locations but never the
values, which may contain teacher targets.

## Unsupported boundary

The bridge does not translate optimizer state, distributed RNG, FSDP or
Megatron checkpoints, Ray state, PPO advantage/clipping semantics, GRPO group
semantics, or a miniVERL teacher cache into PPO reference log-probabilities.
See [compatibility](compatibility.md), [launch requirements](verl-bridge-launch.md)
and the [demo recording script](verl-bridge-demo.md).
