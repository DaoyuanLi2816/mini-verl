# Release checklist

This is the release gate and publication record for miniVERL. A checked item
names an invariant exercised on the stated source. Publication begins only
after the exact release commit and its remote checks are green.

## v0.8.1 development

- [ ] Define the focused v0.8.1 scope before changing runtime semantics.
- [ ] Preserve v0.8.0 tags, public artifacts and frozen benchmark bytes.
- [ ] Complete the full release, compatibility and public-distribution gates.

## v0.8.0 single-GPU verl OPD pivot

- [x] Implement and validate the documented `verl-opd-v0.8-single-gpu-v1`
      subset against official verl `v0.8.0` commit `7aed6b23`: typed config,
      Parquet prompts, current-policy rollout/teacher/update, token-mean
      `forward_kl_topk`, plan/run, PEFT output and OPD import/export.
- [x] Preserve every frozen benchmark and keep policy-gradient, task-reward,
      multi-teacher, multimodal and distributed semantics fail-closed. The
      calculator benchmark remains SHA-256 `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`.
- [x] Complete the v0.8.0 candidate gates; public-distribution verification is
      a post-tag invariant recorded under **After the tag**.

Candidate `4fe229ebd87d2c6ea2e6007033f4fe0cb7877c98` plus validation-record-only
changes passed Ruff, format, mypy, actionlint, strict MkDocs, Markdown/text and
generated-artifact checks; 2,178 CPU tests at 85.30% branch coverage, 8 GPU
tests, 15 network tests and 3 pinned-verl conformance tests. Four Playwright
viewports checked 36 rendered SVG instances and the screenshots were manually
reviewed. The wheel/sdist passed Twine, clean core and `[train]` installs,
pip-only OPD sample/plan/dry-run, and an extracted-sdist test/rebuild whose 142
package-file inventory matched the repository wheel.

## v0.7.1 Product correction

- [x] README, Chinese README, PyPI description and docs landing page lead with
      the installable runtime, hardware boundary, compatibility boundary and
      measured systems evidence; preserved research results follow under
      Research Notes.
- [x] CLI help, doctor wording, project metadata and CFF describe the current
      single-GPU alignment/distillation runtime without promising v0.8
      execution semantics.
- [x] The wheel packages the v0.7 external-study result, schema,
      preregistration and 512 task rows. `evidence show`, `evidence validate`
      and `pilot --builtin-study` work in a clean core-only installation with
      no checkout.
- [x] Ruff, format, mypy, actionlint, CPU/GPU/network tests, strict docs,
      Playwright, package/extracted-sdist, clean installs, bridge and
      frozen-artifact gates pass on the exact release candidate.
- [x] `git shortlog` and commit/body scans show Daoyuan Li as the only source
      author since v0.7.0 and no AI attribution trailers.
- [x] The v0.7.0 tag, published adapter revisions and every frozen benchmark
      remain unchanged.

## v0.7.0 External Alignment Gate evidence release

The preregistered study terminated at checkpoint selection. Later experimental
phases are not unfinished work: they were **not run — scientifically
unauthorized after checkpoint-selection failure**.

- [x] `bridge doctor` validates the bundle tree before opening any file in it.
      Reproduced on Windows with a junction: the hash check walked into a
      directory outside the bundle and the metadata scan reported
      `semantic_secret_key` against a file the bundle did not contain. Symlinks,
      reparse points, non-regular files, escaping entries and oversized trees
      are refused, and a refused bundle reports every check as `not_inspected`.
- [x] The portable metadata privacy scan distinguishes
      `heuristic_passed_full` from `heuristic_incomplete`, records each gap
      with its file and reason, and `--require-complete-metadata-scan` fails on
      an incomplete inspection.
- [x] New extension sidecars are schema version 2 and bind the digest and row
      count of the dataset they are published beside. Sidecars published by
      0.6.0-0.6.3 still read.
- [x] Conversion captures source identity up front and re-checks it before
      publishing, so a report can never describe bytes the conversion did not
      read.
- [x] Output-to-source row provenance is encoded as contiguous runs rather than
      one entry per row, and extension deduplication detail is bounded with an
      exact total.
- [x] The text gate covers Latin-1 range mojibake; two CHANGELOG lines that
      survived v0.6.3 are repaired. The tensor-to-float warnings in
      `test_chunked_equivalence.py` are gone.
- [x] The quality record separates the locally measured commit and platform
      from the exact release commit validated by CI, and states that no GPU
      runner exists.
- [x] External endpoint governance, evaluator adapters, deterministic suite
      reservation, HH-RLHF preparation and two declared candidate lineages are
      present with pinned identities.
- [x] Checkpoint selection completed for two lineages and eight candidates.
      Every candidate measured 0/64 JSONNav retained utility against the
      unchanged `[0.20, 0.90]` gate; no checkpoint was selected.
- [x] Amendment 4 records its post-selection/pre-release timing, the fallback
      lineage-label defect, task-identical selection manifests and unqualified
      judge status. No quantitative value, gate, threshold, endpoint or
      decision changed; final-test access is `not_accessed`.
- [x] Original fallback bytes are preserved at SHA-256
      `53efeb1af196fe8a2fd3733f3f9d6a9ce101fcc76365fc45515adc47cc7d3cd3`;
      the corrected metadata projection is
      `d68ea994672c112b38149c87fed5cb069c26c1be10154187780f98151d19ed65`.
- [x] Primary and fallback source-run selection manifests are both
      `e1e165e3547c7784b17e93b7e665df66ea6cafa70bec093a69377bc6683bc20b`;
      their public LF projections are both
      `2e218db65e39bc7412271e00f7043b287c402c05298bec6618d1a3c3f242a4d5`.
      They were separately generated, task-identical, final-test disjoint and
      are not independent samples.
- [x] The early-stop result is schema validated; 512 privacy-safe JSONNav task
      rows are published with SHA-256
      `694d68cd997bc4b2aa7dd88ebf6572616c9a140fb0df4a672c301095a4f16c7c`.
- [x] Granite Guardian is labelled `unqualified_diagnostic_only`; Granite and
      PairRM qualification, PairRM method preference and teacher qualification
      are `not_run`.
- [x] Teacher qualification: **not run — scientifically unauthorized after
      checkpoint-selection failure**.
- [x] Continuation SFT/DPO/KD/OPD method matrix: **not run — scientifically
      unauthorized after checkpoint-selection failure**.
- [x] Reserved final test: **not accessed — scientifically unauthorized after
      checkpoint-selection failure**; zero tasks scored.
- [x] `miniverl pilot --study-result ...` returns
      `do_not_continue_this_study` / `insufficient_evidence` and preserves the
      existing recipe-evidence path.
- [x] The evidence-release PR's exact final head passes CI, build, docs,
      generated-artifact, visual, package and attribution gates before merge.

## After the tag

- [x] Verify the exact v0.8.0 merge-commit release workflow, OIDC publication,
      PyPI attestations, public clean install and GitHub Release.
- [x] PyPI and the GitHub Release expose identical v0.8.0 files: wheel SHA-256
      `1f1e608f894bc79451db752412fd82101b327ab74ffa7726ebcc98aeb66c57b9`,
      sdist SHA-256
      `c2a6576c4583990900baa0f2b36c25fe37959c6bb773d29529cf50606a7dc175`.
- [x] Advance development to `0.8.1.dev0` in this separate state-sync PR.

- [x] v0.7.1 release run
      [`31566663507`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31566663507)
      completed OIDC Trusted Publishing, attestation verification, exact clean
      install and GitHub Release creation for commit
      `830a4ca5d873bce4cdcc7c43a44d827b096e8c0c`.
- [x] PyPI and the GitHub Release expose identical v0.7.1 files: wheel SHA-256
      `7d29669eaf53de0fd9b3056f3d00ad1d61c990fbe4f6cc97ccaf8e628c60e785`,
      sdist SHA-256
      `8131faee22e8b668bb0ff010f92ffafd9224aa046a017ad6f85fe7774301dea1`.
- [x] This separate state-sync PR advances main to `0.8.0.dev0` after v0.7.1.

- [x] Release run
      [`31468663273`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31468663273)
      completed OIDC Trusted Publishing, attestation verification, exact clean
      install and GitHub Release creation for commit
      `148822964dbb73e97ce06ef740f907364166a724`.
- [x] PyPI and the GitHub Release expose identical files: wheel SHA-256
      `469ae44fe414cd5f56af0fe36091cbd8f12ba34072f67132800aebd7e2746420`,
      sdist SHA-256
      `0ec9e43384b749a64e8263c10e314fce1df87769d053939559917400cfecbce4`.
- [x] Issue #39 has an owner-authored public evidence comment and remains open.
- [x] This separate state-sync PR advances main to `0.7.1.dev0` after merge.

## v0.6.4 (superseded by v0.7.0)

- [x] A reward scaffold saved with a UTF-8 byte-order mark parses instead of
      being reported as a syntax error. CPython strips the BOM when it reads a
      source file; the static checker did not, so a scaffold written by a
      Windows editor was refused as unparseable. Found while verifying the
      published v0.6.3 wheel. Two regressions cover it, including one proving a
      BOM cannot hide a top-level call.
- `pinned-profile-smoke` remains excluded from the required-status-check
      list because `verl-bridge.yml` filters on paths and therefore never
      reports on an unrelated pull request. It still runs, and must pass, on any
      pull request that touches bridge code. Revisit if the workflow ever loses
      its path filter.

## v0.6.3 Security, artifact integrity and release-state hardening

- [x] `bridge doctor` executes zero Python from the inspected bundle by
      default: the reward scaffold is parsed with `ast.parse` and verified
      statically, and the marker-file exploit that ran during a v0.6.2
      diagnosis is a regression test.
- [x] A dynamic reward import happens only under
      `--trust-and-import-reward-code`, which warns before running anything and
      reports `untrusted_code_executed: true` rather than claiming isolation.
- [x] Adapter safetensors are validated past the header: dtype and shape byte
      arithmetic, offset ordering, contiguity and full coverage of the data
      segment, then materialization through the official reader. A truncated
      payload is rejected and `--require-adapter-payload` cannot be satisfied
      by a header.
- [x] No input file can be an output file. `import-verl` and `convert-dataset`
      reject exact, relative, symlink, hard-link and case aliases before a
      transaction exists, and `--overwrite` never authorizes destroying a
      source.
- [x] Conflicting miniVERL extension sources fail closed without printing
      extension values; equal duplicates are recorded as deduplication.
- [x] Invalid rows fail conversion unless `--allow-rejected-rows` is given, and
      a partial run reports `complete_dataset_conversion: false`.
- [x] Parquet schema checks read footer metadata only, and the dataset scan
      stops reading row groups once its row or byte bound is reached.
- [x] `python scripts/release_state.py --check` passes: README English and
      Chinese, the docs selector, `PYPI.md`, `CITATION.cff`, the changelog
      comparison link, the quality record and `PROJECT_STATE.md` all agree with
      `release-state.yaml`.
- [x] Every public scientific figure is readable at 390px and the mobile
      readability exemption list is empty.
- [x] Transactional wording is limited to in-process rollback; no multi-file
      crash atomicity is claimed.
- [x] Every frozen scientific artifact remains byte-identical.

### After the v0.6.3 tag

- [x] Annotated tag `v0.6.3` resolves to release commit
      `005a4549da713716e64c3ae80ff55fb131519f79`; release run
      [`31084165317`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/31084165317)
      completed OIDC Trusted Publishing and GitHub Release creation.
- [x] PyPI and the GitHub Release expose identical files. Wheel SHA-256
      `1c620c310e8e4156f515d52128a0f26a037096347a2bc105c63f173e6563f5e0`,
      sdist SHA-256
      `b93d2fad63432ecd680b72fe0bf79f6b0be212af1aab82780ed8e31636edeefe`.
- [x] The PyPI integrity API exposes one Trusted Publisher attestation bundle
      per distribution, bound to `DaoyuanLi2816/mini-verl` and `release.yml`.
- [x] A clean Windows Python 3.12 install from `https://pypi.org/simple`
      reported `miniverl 0.6.3`, kept torch absent, and refused to execute a
      hostile reward scaffold: the top-level marker write never ran and the
      doctor reported `top_level_call` with `code_executed: false`.
- [x] This separate state-sync change advances development to `0.6.4.dev0`.

## v0.6.1 Visual integrity and bridge correctness release

- [x] Alignment Lab publication is generated as one diverging forest chart and
      two row matrices: every arm exposes all three frozen seeds and its mean,
      quantitative marks remain in-domain, and non-teacher query ratios remain
      `— not applicable` rather than being coerced to zero.
- [x] The scoped safety figure states that both sandbox checks tied at zero
      while utility regressed, records the external endpoints as not run, and
      does not imply a broad safety benchmark.
- [x] Desktop and mobile bridge diagrams distinguish the verified local
      runtime, artifact bundle and pinned upstream parse/load smoke from the
      dashed `Distributed execution: NOT TESTED` layer; teacher, reference and
      reward roles are visually separate.
- [x] Material 9.7.7 builds stable and development documentation with search,
      light/dark modes, copy controls and responsive navigation. Playwright
      checks five pages at 1440x900, 1024x768, 820x1000 and 390x844 and uploads
      all 20 screenshots.
- [x] `import-verl` classifies every supported source field as exact, derived,
      informational only, requiring confirmation or unsupported. Incomplete
      profiles publish a non-executable template with `needs_user_input`; no
      calculator data or unspecified same-base teacher is substituted.
- [x] Runnable imports require explicit environment, teacher, loss and schedule
      choices, safely coerce finite scientific notation, reject unresolved or
      non-finite values, and pass `RunConfig` validation before atomic publish.
- [x] Exported bundles report artifact completeness, upstream parse/load,
      model/data smoke, reward completeness, launchability, distributed testing
      and semantic parity separately. The current fail-closed reward scaffold
      remains `launchable: false` and emits `launch.template.sh`.
- [x] Ruff check/format, mypy across 103 source files, actionlint 1.7.12,
      Markdown/link checks, strict MkDocs, generated-artifact byte comparisons,
      SVG semantics, privacy checks and `git diff --check` pass.
- [x] Full non-GPU/non-network suite passes 1563 tests with 6 deselected and
      86% branch coverage; the available GPU suite passes 5 tests and the
      network suite passes 3.
- [x] Wheel/sdist build, Twine, clean core and `[train]` installs, a real toy
      demo, extracted-sdist tests and import/export end-to-end tests pass.
- [x] The calculator result remains byte-identical at SHA-256
      `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`;
      all RecoveryBench, Consumer Runtime, Alignment Lab and frozen bridge
      result artifacts retain their audited hashes.
- [x] Focused PR [#40](https://github.com/DaoyuanLi2816/mini-verl/pull/40)
      code head `7fed8e7` is green in CI, build, docs and pinned-profile smoke;
      Linux-rendered screenshots were manually inspected at all four widths.
- [x] Release metadata declares exact `0.6.1`; the intended annotated tag is
      exactly `v0.6.1` and will use the existing OIDC-only release workflow.

## After the tag

- [x] Annotated tag `v0.6.1` resolves to release commit
      `48b9e7d9231b5f6cd018f6e927f81df066258f17`; OIDC publication, one
      attestation per file, public hashes, a clean public-wheel install and the
      GitHub Release were verified by tag run
      [`30857762954`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30857762954).
- [x] This separate state-sync change advances subsequent development to
      `0.6.2.dev0`; it merges only after remote checks are green.

Published artifact identity:

- `miniverl-0.6.1-py3-none-any.whl`: SHA-256
  `ef9ce5378e43c0d833b782e431248a0838e3841ace76a8b3a08781dd28007918`
- `miniverl-0.6.1.tar.gz`: SHA-256
  `e5ffd7917035d1f3878b22415dd357cd47fe16b6c77ffdae260e0b85ad7e050f`

## v0.6.0 Verified verl Bridge release

- [x] The documented `single-gpu-online-distillation-v1` profile is pinned to
      official verl `v0.8.0` commit
      `7aed6b230776f963fa09509c10d9c3a767d1102c`; the installed source reports
      `0.8.0.dev0` and no moving branch or mutable revision is accepted.
- [x] The fail-closed importer accepts only 14 named configuration fields; the
      exported bundle supplies the six supported LoRA/reward fields and rejects
      unsupported algorithm, optimizer, distributed-runtime and checkpoint
      semantics instead of claiming PPO/GRPO parity.
- [x] The exact Python 3.12 smoke parses and structurally merges the official
      OmegaConf, loads standard PEFT LoRA and safetensors structure, reads both
      Parquet splits, imports the fail-closed reward scaffold, verifies the
      tokenizer identity, privacy and 14 artifact hashes, and records
      distributed execution as `not tested`.
- [x] Dataset conversion is reversible for the official prompt schema, keeps
      miniVERL extensions in a checksummed sidecar and publishes Parquet through
      a unique temporary file without masking the originating write error.
- [x] Five packaged community records distinguish four existing measured
      artifacts from one explicit `not_measured` preference-teacher template;
      no external adoption, distributed execution or new benchmark result is
      claimed.
- [x] The calculator benchmark JSON remains byte-identical at SHA-256
      `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`,
      and its generated SVG byte comparison passes; prior tags, negative results
      and public adapter revisions remain unchanged.
- [x] Ruff check/format, mypy across 103 source files, actionlint 1.7.12,
      Markdown/link checks, strict MkDocs, PyPI byte comparison and
      `git diff --check` pass.
- [x] Full non-GPU/non-network suite: **1548 passed**, 6 deselected, **85.53%**
      branch coverage; the available GPU/non-network and network suites each
      pass **3 tests**.
- [x] The focused bridge/packaging suite passes 124 tests; wheel and sdist pass
      Twine, a clean Python 3.10 wheel install passes doctor/community export,
      and the wheel contains all bridge modules and five recipe records.
- [x] The architecture SVG passed native and 820-pixel README inspection; the
      1280 by 640 social preview passed native inspection; GitHub Pages deploy
      run `30794655822` is green and the public site returns HTTP 200.
- [x] Focused PR #36 is green and squash-merged as `0d43310`; synchronized main
      CI and build runs `30794655713` and `30794655722` are green, and exact
      pinned-verl PR smoke run `30794329109` is green.
- [x] Release metadata declares exact `0.6.0`; the intended annotated tag is
      exactly `v0.6.0` and will use the existing OIDC-only release workflow.

## After the tag

- [x] Annotated tag `v0.6.0` resolves to exact release commit
      `6cfbdbb7bbf5c6042def4cf154bfe3c3b6530eea`; OIDC publication, public
      hashes/attestations, clean install and one GitHub Release were verified by
      tag run [`30796058250`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30796058250).
- [x] This post-release state-sync change advances development to
      `0.6.1.dev0`; it merges only after remote checks are green.

Published artifact identity:

- `miniverl-0.6.0-py3-none-any.whl`: SHA-256
  `e5fbb99bf410c27d22d6959f9599dfa4fbac2e940dac63f55c9676f68264abd1`
- `miniverl-0.6.0.tar.gz`: SHA-256
  `2e1d85556875f6d23152220897ba919d8b82bda35c5c78fd48efffa7ec22909d`

## v0.5.0 One-GPU Alignment Lab release

- [x] Public preregistration revision 1.4 freezes three seeds, the common SFT
      checkpoint, four continuation updates, 48 test tasks and the disjoint
      seed-1234 recovery rule at SHA-256
      `71307dbfe9a5bb20c686307cafce8bd254c07af8b69c1bf1c6ec0dbf53a8cde0`.
- [x] All 18 method-by-seed arms completed with 48 ordered paired tasks each;
      864 task rows, strict freshness, verifier decisions and DPO provenance
      pass exact publication checks. No completed final arm was rerun.
- [x] The saturated 100% SFT checkpoint, ties from DPO and offline soft
      distillation, and regressions from continued SFT, standard OPD and
      verifier-gated OPD are all reported without suppressing negative results.
- [x] The frozen result, task rows, state diagnostic and six-page PDF have
      SHA-256 values `584752dccb91654109c357b8ebb12681a12a9c1476a9ba539dd35e4d860a22ef`,
      `8d7fc723436d7377d196fc44046d960e3cb7f0aa81e03d49ef05b627eb84630f`,
      `9e08129ba4cd9e460c189b94b4e421d881ba69e3938f02eac95d251f50c88788`
      and `adbffa967f6b9a25d2cdb0cc4464a93c13db4615a1e91499585fb199285d980b`.
- [x] The legacy calculator JSON remains byte-identical at SHA-256
      `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`;
      v0.3/v0.4 tags, artifacts and public adapter revisions are unchanged.
- [x] Ruff check/format, mypy, actionlint, Markdown links, generated-artifact
      byte comparison and `git diff --check` pass; four SVGs, banner, social
      preview and every PDF page were visually inspected.
- [x] Full non-GPU/non-network suite: **1508 passed**, 6 deselected, **86.13%**
      branch coverage.
- [x] Available RTX 4080 GPU suite: **5 passed**, 1509 deselected.
- [x] Network suite: **3 passed**, 1511 deselected, including immutable public
      adapters used by the release.
- [x] Wheel and sdist build and pass Twine; clean core and CPU-training wheel
      installs, a real no-network toy demo, and the isolated extracted-sdist
      suite pass at `miniverl 0.5.0.dev0`.
- [x] Focused PR #33 is green and merged as `f9dae54`; synchronized main CI
      and build runs `30788116397` and `30788116373` are green.
- [x] Release metadata declares exact `0.5.0`; the intended annotated tag is
      exactly `v0.5.0` and will use the existing OIDC-only release workflow.

## After the tag

- [x] OIDC publication, public hashes/attestations, clean install and one
      GitHub Release were verified from the same distributions by tag run
      [`30789267409`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30789267409).
- [x] This post-release state-sync change advances development to
      `0.6.0.dev0`; it merges only after remote checks are green.

Published artifact identity:

- `miniverl-0.5.0-py3-none-any.whl`: SHA-256
  `ff60cb747e2ad1fd74575dd8920b11b48c6c16cc97743e94f9583d162d18819c`
- `miniverl-0.5.0.tar.gz`: SHA-256
  `d3340e0526eb4b20bb9ee15960c27c740be0fdacb9a51b029627faa63ca5276d`

## v0.4.0 Consumer Runtime release

- [x] Preregistration revisions 1.1 and 1.2 were public before the sole
      headline run; the eager-attention and FP32-compute amendments retain the
      earlier SDPA and BF16 diagnostics as non-headline negative evidence.
- [x] All eight dual/shared by sequential/2/4/auto cells completed with one
      trajectory digest and one teacher-target digest; all 12 preregistered
      loss, full-gradient and post-update-logit equivalence gates passed.
- [x] The frozen result, profiler and Pareto SVG have SHA-256 values
      `a302da31af99f1d29f1efd4e6b3dbeb6ea4ac956bba102ca8a1bee8dff0319eb`,
      `66111cd7fc876cf1befea3297a1a51bcd99252c0bf8989c029381e1dc155a98b`
      and `98645a668a7832423d28b621262292619615917f037adf7219ff1bf071fb2fea`.
- [x] Typed batching, mask isolation, all three objectives, shared adapter-role
      restoration, optimizer isolation, one-base loading, reference isolation
      and standard student export have focused CPU and tiny-HF coverage.
- [x] The legacy calculator JSON remains byte-identical at SHA-256
      `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`;
      v0.3 artifacts and negative results are unchanged.
- [x] Ruff check/format, mypy, actionlint, Markdown links, generated SVG byte
      comparison and `git diff --check` pass; native and 820-pixel SVG renders
      were visually inspected.
- [x] Full non-GPU/non-network suite: **1448 passed**, 6 deselected, **86.26%**
      branch coverage.
- [x] Available RTX 4080 GPU suite: **5 passed**, 1449 deselected.
- [x] Network suite: **3 passed**, 1451 deselected, including all immutable
      public adapters used by the current tests.
- [x] Wheel and sdist build and pass Twine; a clean Python 3.10 wheel install
      reports `miniverl 0.4.0.dev0` and passes the core command smoke tests.
- [x] Focused PR #30 is green and merged as `9914c6d`; synchronized main CI
      and build runs `30776659196` and `30776659178` are green.
- [x] Release metadata declares exact `0.4.0`; the intended annotated tag is
      exactly `v0.4.0` and will use the existing OIDC-only release workflow.

## After the tag

- [x] OIDC publication, public hashes/attestations, clean install and one
      GitHub Release were verified from the same distributions by tag run
      [`30777530767`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30777530767).
- [x] This post-release state-sync change advances development to
      `0.5.0.dev0`; it merges only after remote checks are green.

Published artifact identity:

- `miniverl-0.4.0-py3-none-any.whl`: SHA-256
  `8204c2f015e017ff15aadb465d0afa689949979f4aeba3b9b3abcc3f01c2511a`
- `miniverl-0.4.0.tar.gz`: SHA-256
  `b6f87ff3a2a97f926301683b983920082719c893bb22a31759b17b0309b1e053`

## v0.3.0 RecoveryBench release

- [x] Revision-1.3 preregistration predates the valid final run and remains
      byte-frozen at SHA-256
      `9c4c2ec19a56cebb2b2c1c0f3c7e504a9285467c99ae1590488251fbf2ff3934`.
- [x] All three schema-v3 budget results validate; 36 task files, 4,608
      trajectories, 128 task records per arm, task pairing, cold checkpoints,
      frozen datasets and all 75 fresh-policy update versions were audited.
- [x] Negative results, failed teacher candidates, the invalid v1.2 partial
      run and the post-run aborted wall-budget replacement are disclosed and
      excluded from headline analysis.
- [x] Equal-updates, selected-position and cycle-capped wall diagnostics are
      reported without changing the frozen experiment or analysis after seeing
      outcomes.
- [x] The five RecoveryBench publication artifacts and three SVGs have exact
      hash tests; SVG generator output byte-matches the committed figures.
- [x] The six-page technical PDF is data-bound, reproducible byte-for-byte and
      visually inspected on every page; the three SVGs and banner are visually
      clean at their intended sizes.
- [x] The legacy calculator JSON remains byte-identical at SHA-256
      `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`.
- [x] Ruff check/format, mypy, actionlint and `git diff --check` pass.
- [x] Full non-GPU/non-network suite: **1408 passed**, 6 deselected, **86.68%**
      branch coverage.
- [x] Available RTX 4080 GPU suite: **5 passed**, 1409 deselected.
- [x] Network suite: **3 passed**, 1411 deselected, including both immutable
      public teacher adapters.
- [x] Development wheel and sdist build; both pass Twine. Clean Python 3.10
      core and clean Python 3.12 training-extra installs pass their command and
      demo/report/cache smoke tests.
- [x] Extracted sdist passes PyPI-description, Ruff, format, mypy and all 1408
      CPU tests; its rebuilt wheel has the same 81-file package inventory.
- [x] Python 3.10.11 minimum boundary passes with Torch 2.3.1+cpu,
      Transformers 4.51.3, PEFT 0.12.0, Accelerate 0.33.0, NumPy 1.24.4 and
      bitsandbytes 0.43.3. Python 3.13.13 latest boundary passes with Torch
      2.13.0+cpu, Transformers 5.14.1, PEFT 0.20.0, Accelerate 1.14.0, NumPy
      2.5.1 and bitsandbytes 0.50.0.
- [x] Focused PR #27 is green and merged as `bee82d3`; synchronized `main` CI
      and build workflows are green.
- [x] Release metadata declares exact `0.3.0`; the intended annotated tag is
      exactly `v0.3.0` and will use the existing OIDC-only release workflow.

## After the tag

- [x] OIDC publication, public hashes/attestations, clean install and one
      GitHub Release were verified from the same distributions by tag run
      [`30772772078`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30772772078).
- [x] This post-release state-sync change advances development to
      `0.4.0.dev0`; it merges only after remote checks are green.

Published artifact identity:

- `miniverl-0.3.0-py3-none-any.whl`: SHA-256
  `e42404ced88b75ba4ff31541cb3df697da0eee09a68f603e4eadbf69a11d2032`
- `miniverl-0.3.0.tar.gz`: SHA-256
  `006ce418243286dd27e0731090bf9fa1711a1abc962ecfcc4d39807257539cb2`

## Historical v0.2.6 release record

### Version consistency

- [x] Tagged source `v0.2.6` declares package version `0.2.6`.
- [x] `CHANGELOG.md` has a dated `## [0.2.6]` section.
- [x] `CITATION.cff` version and release date match.
- [x] The intended annotated tag is exactly `v0.2.6`.
- [x] The package/project name remains `miniverl`.

### Correctness and lifecycle

- [x] Strict model-output JSON rejects non-finite values, duplicate keys,
      oversized integers, excessive depth/members and invalid surrogates before
      environment execution; protocol-v1 byte fixtures remain unchanged.
- [x] Calculator verifier-v2 requires a complete finite answer and compatible
      units while verifier-v1 remains identifiable for historical artifacts.
- [x] Calculator, JSON-navigation and SQLite direct/rollout/property tests turn
      arbitrary strings into bounded verification results; non-finite SQLite
      answers are malformed rather than process exceptions.
- [x] Deterministic Event-based regressions prove train/evaluate/checkpoint
      save/load/close exclusion, no-mutation losing operations, later and
      repeated close, and READY/terminal/evaluation-only lifecycle behavior.
- [x] Windows multiprocessing regressions prove same-run exclusion before model
      loading, report completion under lock, checkpoint selection under lock,
      bounded timeout, killed-owner recovery and different-run progress.
- [x] Public checkpoint load checks READY before and after exclusive ownership;
      load validation failure preserves model/progress/state and releases the
      guard, while exact resume behavior remains unchanged.
- [x] Evaluation restores the exact prior model mode after success and injected
      failures in rollout, trajectory write, diagnostics, metrics and events.
- [x] Numerical property tests for exact/bucketed objectives, weighted
      reductions, chunking, finite gradients, zero tails and OOM RNG
      equivalence remain green.

### Quality gates

- [x] `git diff --check`.
- [x] `ruff check .`.
- [x] `ruff format --check .`.
- [x] `mypy src/miniverl`.
- [x] Full non-GPU/non-network pytest suite passes with branch coverage above
      the required 80%.
- [x] All GPU tests pass on an NVIDIA GeForce RTX 4080.
- [x] All opt-in network tests pass.
- [x] Transformers 4.51.x and 5.x compatibility checks pass on the PR head.
- [x] The declared minimum Python 3.10 training dependency bundle passes the
      no-network toy/HF contract.
- [x] The current Python 3.13 training dependency bundle passes the same
      contract.
- [x] Core Python 3.10, 3.11, 3.12 and 3.13 checks pass on the PR head.
- [x] `actionlint` passes with the repository's `cuda` self-hosted label
      declared in `.github/actionlint.yaml`.
- [x] No unfinished implementation markers remain under `src`, `tests`,
      `examples` or `scripts`.

### Packaging and clean installs

- [x] A clean `python -m build` produces one `0.2.6` wheel and one sdist.
- [x] `python -m twine check dist/*` passes.
- [x] The wheel contains the report template and no tests.
- [x] The sdist contains the full shipped test surface, including scripts and
      workflow fixtures, and its complete non-GPU/non-network suite passes from
      an extracted directory outside the checkout.
- [x] A wheel rebuilt from the extracted sdist has the same runtime-package
      inventory as the repository wheel.
- [x] A clean core-only wheel install runs `--help`, `--version` and
      `doctor --json` with torch, Transformers, PEFT and bitsandbytes absent
      and unimported.
- [x] A clean install of the same wheel with `[train]` runs demo, inspect,
      report and weights-only standalone eval.
- [x] Reusing the demo output without `--overwrite` fails without changing a
      file; explicit overwrite produces a fresh completed run.

### Artifacts, documentation and hygiene

- [x] Every shipped run recipe validates; every benchmark config resolves.
- [x] The committed benchmark JSON Schema is byte-identical to generated
      output and all published results validate.
- [x] The frozen
      `benchmarks/results/gpu-calc-hard-equal-update-v2.json` SHA-256 remains
      `53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc`.
- [x] The benchmark SVG remains generated from and bound to that exact JSON.
- [x] All tracked JSON/JSONL parses strictly.
- [x] README and documentation Markdown links pass link checking.
- [x] `PYPI.md` byte-matches its generator, stable repository links use
      `v0.2.6`, and built wheel metadata contains no relative or local links.
- [x] Shareable HTML/Markdown/JSON/YAML and benchmark exports contain no fake
      credential/path sentinel; benign tokenizer/count metadata remains intact.
- [x] No model weights, checkpoints, caches or databases are tracked.
- [x] The banner and benchmark SVG were rendered and visually inspected.
- [x] Benchmark grids begin below their tick labels; the generated dark SVG is
      still bound to the immutable source JSON.
- [x] The supported single-GPU recipe uses automatic bf16/fp16 selection and
      carries no GPU model or VRAM-tier tag.
- [x] The base-vs-`[train]` installation split and v1 scientific confound are
      stated explicitly.
- [x] `PROJECT_STATE.md`, the compatibility policy, support claims and
      dependency boundaries describe the validated implementation.
- [x] No RecoveryBench or unrelated feature expansion was added.

### Trusted publishing readiness

- [x] PyPI reports project `miniverl`; `v0.2.6` is the current public version.
- [x] GitHub environment `pypi` exists and has a deployment branch policy.
- [x] `release.yml` requests `id-token: write`, uses the `pypi` environment and
      publishes only on a tag push.
- [x] The maintainer registered the pending publisher for
      `DaoyuanLi2816/mini-verl`, workflow `release.yml`, environment `pypi`.
- [x] The immutable `v0.2.0` through `v0.2.6` tags, frozen calculator JSON and
      pinned public protocol-teacher adapter are unchanged.

### After the tag

- [x] Annotated tag `v0.2.6` resolves to exact validated merge commit
      `59fe738709526a13f354a744ab763f13530de4d1`.
- [x] Tag workflow
      [`30722451004`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30722451004)
      passed metadata, tests, one-time build, OIDC publication, attestations,
      public verification and GitHub Release creation.
- [x] Public PyPI and GitHub Release wheel/sdist hashes match workflow
      artifacts.
- [x] This green-gated state-sync change advances development to
      `0.2.7.dev0`.

Publication completed on 2026-08-01 UTC:

- [`miniverl 0.2.6`](https://pypi.org/project/miniverl/0.2.6/) exposes Trusted
  Publishing provenance for both distributions.
- [`miniVERL v0.2.6`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.6)
  contains the byte-identical wheel, sdist and `SHA256SUMS`.
- An independent no-cache Windows Python 3.10 install from the public PyPI
  index reported `miniverl 0.2.6`, passed `doctor`, and kept torch,
  Transformers, PEFT and bitsandbytes absent.

Published artifact identity:

- `miniverl-0.2.6-py3-none-any.whl`: SHA-256
  `11d6b001752c41a0100f12c29b125a9dc082703dbeadc6b0317a88ac818d8695`
- `miniverl-0.2.6.tar.gz`: SHA-256
  `91e7b2918286c342cacaf2582dbed57c1e7a1bf4e1064d327e349b1d77c28886`

## Historical v0.2.5 record

- [x] Annotated tag `v0.2.5` resolves to exact validated merge commit
      `a9a84510741b4ade8a405c100affdf1caed55ae6`.
- [x] Tag workflow
      [`30611603505`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30611603505)
      passed metadata, tests, one-time build, OIDC publication, attestations,
      public verification and GitHub Release creation.
- [x] Public PyPI and GitHub Release wheel/sdist hashes match workflow
      artifacts.
- [x] This green-gated state-sync change advances development to
      `0.2.6.dev0`.

Publication completed on 2026-07-31 UTC:

- [`miniverl 0.2.5`](https://pypi.org/project/miniverl/0.2.5/) exposes Trusted
  Publishing provenance for both distributions.
- [`miniVERL v0.2.5`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.5)
  contains the byte-identical wheel, sdist and `SHA256SUMS`.
- An independent no-cache Windows Python 3.10 install from the public PyPI
  index reported `miniverl 0.2.5`, passed `doctor`, and kept torch,
  Transformers, PEFT and bitsandbytes absent.

Published artifact identity:

- `miniverl-0.2.5-py3-none-any.whl`: SHA-256
  `70c98284bce151fc74b508047b354929846efb71c3fe8f451c0d0ba1bec48e9d`
- `miniverl-0.2.5.tar.gz`: SHA-256
  `d30bb07ebca676a3960d4b5c46075a8a2e13e58629b96984e30f8f7bab67dce0`

## Historical v0.2.4 record

- [x] Annotated tag `v0.2.4` resolves to exact validated commit
      `57dec193af88b462dcc41d82fc6fecb813e161fd`.
- [x] Tag run
      [`30522484949`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30522484949)
      passed metadata, tests, one-time build, OIDC publication and attestation
      generation. Recovery run
      [`30524088015`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30524088015)
      verified the original artifacts, public hashes/attestations and clean
      install, then created the GitHub Release without rebuilding or uploading.
- [x] Public PyPI and GitHub Release wheel/sdist hashes match the original
      workflow artifacts.
- [x] Development advances to `0.2.5.dev0` through a green state-sync PR.

Publication completed on 2026-07-30:

- [`miniVERL v0.2.4`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.4)
  contains the byte-identical wheel, sdist and `SHA256SUMS`.
- An independent no-cache Windows Python 3.10 install from the public PyPI
  index reported `miniverl 0.2.4`, passed `doctor`, and kept torch,
  Transformers, PEFT and bitsandbytes absent.

Published artifact identity:

- `miniverl-0.2.4-py3-none-any.whl`: SHA-256
  `3f5a239bbbd2f85217cf11f691fbb63f647092f67b82da4de38bd6907c5ab0f1`
- `miniverl-0.2.4.tar.gz`: SHA-256
  `03f0e844df2c91deed5c211cdd2dd598d22f03d59d99cd8e792a58211c0b2296`

## Historical v0.2.3 record

Publication completed on 2026-07-29:

- [x] Annotated tag `v0.2.3` resolves to exact validated commit
      `38924da743180e6767f1e3b252feafdccd70759b`.
- [x] Release run
      [`30513947051`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30513947051)
      passed metadata, tests, build, OIDC publication, public verification and
      GitHub Release creation.
- [x] Public PyPI hashes and Trusted Publishing attestations match the workflow
      artifacts; clean workflow Python 3.12 and independent Windows Python 3.10
      core installs reported `miniverl 0.2.3`.
- [x] The
      [`miniVERL v0.2.3`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.3)
      GitHub Release contains the byte-identical wheel, sdist and `SHA256SUMS`.
- [x] Development advanced to `0.2.4.dev0`.

Published artifact identity:

- `miniverl-0.2.3-py3-none-any.whl`: SHA-256
  `033e51bfbdae20a91d942ef7a5c22ef6c8a00317cc9b775b102d303f2e1a6619`
- `miniverl-0.2.3.tar.gz`: SHA-256
  `6f7d20fd4b4a90e6a3fe1e97c9ced26268e013bb87462ba75a7d09510bd2f011`

## Historical v0.2.2 record

Publication completed on 2026-07-29:

- [x] Annotated tag `v0.2.2` resolves to exact validated commit
      `518590cb43ff788fa65f73ee9cf3a7afb6dfba5a`.
- [x] Release run
      [`30494182647`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30494182647)
      passed metadata, tests, build, OIDC publication, public verification and
      GitHub Release creation.
- [x] Public PyPI hashes and Trusted Publishing attestations passed; clean
      Python 3.10 and workflow Python 3.12 core installs reported
      `miniverl 0.2.2`.
- [x] The
      [`miniVERL v0.2.2`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.2)
      GitHub Release contains the byte-identical wheel, sdist and `SHA256SUMS`.
- [x] Development advanced to `0.2.3.dev0`.

Published artifact identity:

- `miniverl-0.2.2-py3-none-any.whl`: SHA-256
  `1ead97173bb11ce3da963b94f628df825a5b14648fed488cf4d88c47cba9dd59`
- `miniverl-0.2.2.tar.gz`: SHA-256
  `3951dd4addc5d85b3e58ce72ecffac65c38bf2eab951d2c08cce8f20c886185c`

The first two install-verification attempts saw PyPI's JSON/file APIs before
the public simple index had propagated. Attempt 3 passed unchanged. The
post-release workflow now retries that final public-index install so future
releases do not report this expected propagation window as a package defect.

## Historical v0.2.1 record

Publication completed on 2026-07-29:

- [x] Create annotated tag `v0.2.1` on exact validated commit
      `591881b0d094f5c53ff47a9419e679b762fb44b0`.
- [x] Verify release run
      [`30474597179`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30474597179)
      tests and builds the distributions once.
- [x] Verify OIDC publication, public PyPI hashes and Trusted Publishing
      attestations for
      [`miniverl 0.2.1`](https://pypi.org/project/miniverl/0.2.1/).
- [x] Verify the
      [`miniVERL v0.2.1`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.1)
      GitHub Release contains the same wheel, sdist and
      `SHA256SUMS`.
- [x] Install public `miniverl==0.2.1` in a clean Windows environment and run
      `miniverl --version` plus `miniverl doctor --json`; the core verdict
      passed and Torch remained absent.
- [x] Open the post-release state-sync change and advance development to
      `0.2.2.dev0`.

Published artifact identity:

- `miniverl-0.2.1-py3-none-any.whl`: SHA-256
  `0177d50026da86047c2a03f90e7786c794b26c5b0d6fef193c58ed35c08d8cda`
- `miniverl-0.2.1.tar.gz`: SHA-256
  `80f890c1ab8be0ccdf6c5ce293a5c4d7bb6a6f7ab7a57db34090384fcaa7e16c`

Independent downloads from PyPI and the GitHub Release reproduced both
digests.

## Historical v0.2.0 record

Tag workflow
[`30421231859`](https://github.com/DaoyuanLi2816/mini-verl/actions/runs/30421231859)
published `v0.2.0` from commit
`6092706b4a4e750c4571d7d6a7decbc26af851b2` on 2026-07-28
(2026-07-29 UTC).

- Wheel SHA-256:
  `cf850a6333483a3ee22c0c0e98df1e1b2e6faa184480573e0666658b53a29262`
- Sdist SHA-256:
  `3d5107b4f6351204335f800ce924208843f08f54441378bd9f25c3c6fa17456b`
- PyPI:
  [`miniverl 0.2.0`](https://pypi.org/project/miniverl/0.2.0/)
- GitHub:
  [`miniVERL v0.2.0`](https://github.com/DaoyuanLi2816/mini-verl/releases/tag/v0.2.0)
