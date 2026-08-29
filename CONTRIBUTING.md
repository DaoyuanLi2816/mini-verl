# Contributing to miniVERL

Thank you for helping make one-GPU distillation easier to inspect, reproduce
and extend. This guide covers the checks that keep code and published evidence
aligned.

## Measurement provenance

**Every reported number needs a measurement artifact or command.** When a check
was unavailable, write "not run" and include the exact command for a future
run. This convention keeps reviews useful across different hardware.

## Setup

```bash
git clone https://github.com/DaoyuanLi2816/mini-verl
cd mini-verl

python -m venv .venv && . .venv/bin/activate       # or: uv venv
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,train]"                      # CPU test stack
# optional, for the 4-bit path:
pip install -e ".[cuda]"

miniverl doctor
```

## The gate

Everything below must pass before you open a pull request. CI runs the same
commands.

```bash
ruff check .
ruff format --check .
mypy src/miniverl
pytest -q -m "not gpu and not network"
python -m build
python -m twine check dist/*
```

If you have a CUDA device:

```bash
pytest -q -m gpu
```

Start with the focused test file for the code you changed, then run the full
CPU gate above. Network and GPU tests are opt-in; state `not run` when the
required hardware or credentials are unavailable.

## What a good change looks like

* **A test that can fail.** Not `assert True`, not a test that passes because
  everything is rejected. If you fixed a bug, the test should fail on the
  previous commit.
* **The smallest complete vertical slice.** A schema change plus its validator
  plus its test, not three of the four.
* **A docstring that says what invariant the code protects**, not what the code
  does. The code already says what it does.
* **Lazy heavy imports.** Nothing in `miniverl.config`, `miniverl.schemas`,
  `miniverl.trajectory`, `miniverl.inspection`, `miniverl.reporting`,
  `miniverl.cache` or `miniverl.cli` may import torch at module scope. The
  `core` CI job asserts this.

## Things the review will reject

* `torch.save`, `pickle`, `eval`, `exec`, `os.system`, `subprocess` with a shell,
  or any network call at import time.
* A new required dependency in the base install.
* A benchmark number without the hardware and the command that produced it.
* A comparison table cell asserting that another project lacks a feature,
  without a link to the code or docs that show it.
* Marketing language. No "blazing", "seamless", "state of the art", or claims
  that miniVERL is first, fastest or best at anything.
* Silent behaviour: truncating a sequence, changing a model, reusing a stale
  cache, or swallowing an exception. Fail loudly with a hint instead.

## Documentation voice

Lead with what the workflow does, who it helps and which artifact it produces.
State a local constraint where it changes the reader's next action; link to
the [compatibility policy](docs/compatibility.md) or
[limitations](docs/limitations.md) for complete exclusion lists and scientific
boundaries. Avoid repeating a catalogue of absent features after every product
capability. Research reports and security contracts should keep their precise
caveats close to the evidence or decision they qualify.

## Adding an environment

See the "New environment" issue template for the checklist. In short: seeded and
deterministic, disjoint splits, an exact verifier, a deterministic oracle, no
network, no shell, no arbitrary filesystem access. Register it in
`src/miniverl/environments/registry.py` — there is no plugin discovery on
purpose, so the set of things a recipe can execute stays auditable.

## Adding a divergence or changing the objective

Read `docs/math.md` first. Then:

1. Add the function next to its siblings in `src/miniverl/losses/`.
2. Add a **brute-force reference test** in `tests/unit/`, written from the
   textbook definition with plain Python loops, so a bug in the vectorized code
   cannot hide behind the same expression on both sides.
3. Add a Hypothesis property in `tests/property/` for whatever must always hold
   (non-negativity, symmetry, a bound, invariance).
4. Confirm the chunked and unchunked gradients still agree.
5. Update `docs/math.md` in the same commit.

## Contributing a benchmark result

For the versioned community hardware matrix, start with a privacy-safe template:

```bash
miniverl benchmark --export-community submission.json
```

Fill measured fields only from retained artifacts and include their SHA-256
digests. CI validates the schema, packaged recipe digest, hardware/software
provenance, privacy and artifact hashes. Negative and failed runs are useful;
label them rather than removing them. See the
[community submission guide](docs/community-benchmarks.md).

```bash
miniverl train <recipe>
miniverl export-benchmark runs/<run-id> --out benchmarks/results/<gpu>-<recipe>.json
```

`export-benchmark` sanitizes the run: it keeps the GPU model, VRAM, OS family and
library versions, and drops absolute paths and anything identifying. Read the
file before you post it. Open a pull request adding it under
`benchmarks/results/`; `benchmarks/README.md` explains the schema.

Good first contributions are bounded fixtures and compatibility checks:
documentation corrections, a deterministic adversarial case for an existing
environment, or an old-artifact reader regression. Changes to objectives,
numerical reductions, cache transactions or checkpoint semantics need a
maintainer-designed test plan first.

## Schema changes

`Trajectory`, the teacher cache and the benchmark result all carry a
`schema_version`. If you change any of their shapes, bump the version and make
the reader **reject** the old one with an actionable message. Silently
reinterpreting an old artifact is worse than refusing to read it.

## Commit and PR style

Imperative subject line under 72 characters, a body that explains *why*. Keep
one logical change per commit. The pull-request template lists the evidence to
paste; paste the real output.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Contributions are accepted under the Apache-2.0 license of this repository. If
you adapt code from another project, keep its notice and add it to
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) in the same commit.
