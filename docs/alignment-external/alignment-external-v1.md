# External Alignment Gate v1: a preregistered checkpoint-selection failure

!!! failure "The study terminated at its preregistered starting-checkpoint gate"

    **0 selected checkpoints · 0 qualified teachers · 0 continuation arms ·
    0 final-test tasks accessed.** Both declared candidate lineages measured
    0/64 retained JSONNav utility for every candidate. The unchanged gate
    required 20–90%, so no downstream work was scientifically authorized.

This is an early-termination result, not an SFT/DPO/KD/OPD comparison. The
[machine-readable result](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.7.0/benchmarks/results/alignment-external-v1.json),
[schema](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.7.0/benchmarks/schema/alignment-external-result.schema.json) and
[task evidence](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.7.0/benchmarks/evidence/alignment-external-v1/jsonnav-selection-records.jsonl)
encode that distinction.

<picture class="alignment-figure">
  <source media="(max-width: 600px)" srcset="../study-early-stop-mobile.svg">
  <img src="../study-early-stop.svg" alt="External alignment study flow: endpoint governance and candidate generation completed; candidate selection stopped at zero of 64 retained utility; teacher qualification and continuation training did not run; the final test was not accessed.">
</picture>

## Research question and design

The preregistered question asked whether continued alignment SFT, DPO, offline
teacher distillation, standard OPD or verifier-gated OPD could improve
externally measured alignment from one shared, non-saturated SFT policy while
controlling utility, cost and VRAM on one RTX 4080. The design first required a
starting policy with headroom and retained JSONNav competence. It committed to
the first candidate in order that passed; if both declared lineages failed,
the study would publish the failure and run nothing downstream.

The primary lineage continued `Qwen/Qwen3-0.6B` on HH-RLHF. Amendment 2 allowed
one fallback lineage anchored at the public tool-policy SFT adapter, then
continued on the same HH-RLHF data. Each lineage contained updates 0, 4, 8 and
16. The bands, endpoint and candidate order were not changed after observation.

## Amendments and timing

- Amendment 1, after the first eval values were visible but before any real
  JSONNav value, replaced an invalid `1 - over_refusal` proxy with the
  already-preregistered retained-utility endpoint. The invalid run remains
  preserved under `artifacts/v07-start-selection/superseded/`.
- Amendments 2 and 3 were public before the measurements they governed. They
  declared the fallback lineage and separated judge qualification from an
  eventual method-level preference endpoint.
- Amendment 4 is a **post-selection evidence correction before release**. It
  records the fallback lineage-label defect, identical selection task IDs and
  unqualified evaluator status after outcomes were observed. It changes no
  quantitative value, gate, threshold or selection decision. The reserved
  final test remained unaccessed. The public preregistration merge is
  `c50aa93b95e6fe4a6aa6251491d3c2b5a9480ebe`.

## Corrected real JSONNav measurement

The superseded proxy was not tool utility. The corrected evaluation drove the
real miniVERL agent loop against the JSONNav environment with greedy decoding,
the fixed task manifest and policy version zero. All eight candidates scored
0/64. The primary lineage emitted no valid tool call and stopped at the parse
error limit; the fallback lineage did issue JSONNav tool calls, but never
solved a task. These distinct observed behaviors lead to the same necessary
gate failure.

<picture class="alignment-figure">
  <source media="(max-width: 600px)" srcset="../checkpoint-gate-matrix-mobile.svg">
  <img src="../checkpoint-gate-matrix.svg" alt="Checkpoint gate matrix for four primary and four fallback candidates. It shows instruction following, over-refusal, zero of 64 JSONNav utility and a failed gate for every row.">
</picture>

<div class="table-scroll" markdown>

| lineage | candidate | instruction following | over-refusal | JSONNav utility | gate |
| --- | --- | ---: | ---: | ---: | --- |
| primary | update-000 | 41.1% | 2.0% | **0/64** | failed |
| primary | update-004 | 44.2% | 0.0% | **0/64** | failed |
| primary | update-008 | 41.1% | 0.0% | **0/64** | failed |
| primary | update-016 | 57.9% | 2.0% | **0/64** | failed |
| fallback | update-000 | 42.1% | 0.0% | **0/64** | failed |
| fallback | update-004 | 43.2% | 0.0% | **0/64** | failed |
| fallback | update-008 | 42.1% | 2.0% | **0/64** | failed |
| fallback | update-016 | 55.8% | 56.0% | **0/64** | failed |

</div>

The optional Granite harmful-compliance numbers are deliberately absent from
this headline matrix. Granite Guardian was executed only as an **unqualified
diagnostic**; the decision does not depend on it.

## Harness validation

The environment oracle solved 8/8 tasks under the pinned path, and the
executable regression also checks the 64-, 128- and 256-token per-turn budgets
and both difficulty levels. This rules out a broken JSONNav environment path;
it does not turn either candidate lineage into a competent JSONNav policy.

## Why the lineages lacked matched utility competence

HH-RLHF is conversational preference data, not a JSONNav tool-use corpus. The
fallback anchor did have measured strict tool-protocol competence, but on the
separate `tool_policy` environment with different tools and state. Its public
provenance does not establish JSONNav competence. In retrospect, the gate
tested a precondition neither lineage met. Changing that endpoint after seeing
the zero would invalidate the preregistration, so it was retained and the
design mismatch is the published finding.

## Identical-suite disclosure

The primary and fallback selection manifests were separately generated from
the same deterministic seed, endpoint counts, algorithm and reserved final
IDs. They are byte-identical (SHA-256
`e1e165e3547c7784b17e93b7e665df66ea6cafa70bec093a69377bc6683bc20b`),
contain identical task IDs and are disjoint from the final suite. They are
**not independent evaluation samples**. Using the same tasks has no effect on
the conclusion that both lineages failed the necessary gate, but it limits how
the two runs may be described.

## Metadata correction and provenance

The fallback generator hard-coded the primary lineage label. Original bytes
are preserved with SHA-256
`53efeb1af196fe8a2fd3733f3f9d6a9ce101fcc76365fc45515adc47cc7d3cd3`.
The [corrected projection](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.7.0/benchmarks/evidence/alignment-external-v1/fallback-start-selection.corrected.json)
has SHA-256
`6f23de43f03a69275d8bedc9b029a1b728fb2004a9b3a225c66ba1fee671592b`.
Only `lineage`, `lineage_id`, `lineage_description` and `lineage_anchor` differ;
candidate metrics and the selection decision compare equal. See the
[correction manifest](https://github.com/DaoyuanLi2816/mini-verl/blob/v0.7.0/benchmarks/evidence/alignment-external-v1/fallback-correction-manifest.json).

## Evaluator qualification boundary

| evaluator | implementation | execution | qualification | result use |
| --- | --- | --- | --- | --- |
| IFEval | implemented | selection split | not required | selection metric |
| XSTest string match | implemented | selection split | not required | scoped refusal metric |
| Granite Guardian | implemented | selection split | **not run** | unqualified diagnostic only |
| PairRM | implemented | method comparison not run | **not run** | no preference result |
| Teacher | — | not run | **not run** | requires a selected checkpoint |

The XSTest result uses the documented string-match classifier, not a GPT-4
judge. No evaluator or teacher qualification artifact exists, and no human
preference measurement exists.

## Cost and stopped work

Primary selection used 2,454.4 GPU-seconds and 5.246 GiB peak reserved VRAM;
fallback selection used 1,764.4 GPU-seconds and 5.145 GiB. Candidate training
cost is reported separately in the preserved manifests. No teacher
qualification, continuation training or final-test generation was run after
the stop.

## What v0.7.0 establishes

It establishes that the preregistered fail-fast gate was enforced; both
declared starting-policy lineages lacked competence on the selected retained-
utility environment; the portable endpoint and evidence infrastructure can
represent an early stop without fabricating method rows; and the original
metadata defect is traceable.

It does **not** establish whether OPD is better or worse than SFT, DPO or KD;
does not qualify Granite, PairRM or a teacher; does not provide a broad safety
result; and does not access the reserved final test. A future comparison would
require a preregistered starting lineage with independently established,
task-matched retained utility. That is a limitation, not a roadmap commitment.

## Reproduce the evidence-only checks

```bash
python scripts/publish_alignment_external_artifacts.py
miniverl pilot --study-result benchmarks/results/alignment-external-v1.json --json
pytest -q tests/unit/test_alignment_external_evidence_release.py
```

The portable JSONNav evidence contains 512 rows and no prompts, response text
or absolute paths. Its SHA-256 is
`18d5733e70bfe292e282bd5b6e3fc94869837fab30a151a642aa11c3e4c9d771`.
