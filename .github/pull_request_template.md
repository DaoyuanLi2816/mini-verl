## What this changes

<!-- One paragraph. What is different afterwards, and why. -->

## Evidence

<!--
Paste the actual output, not a description of it. A claim without a command is
not evidence. Sanitize commands and output first: do not paste tokens, home
paths, proprietary prompts or model data.
-->

```text
ruff check .
ruff format --check .
mypy src/miniverl
pytest -q -m "not gpu and not network"
```

## Checklist

- [ ] `ruff check .`, `ruff format --check .` and `mypy src/miniverl` pass.
- [ ] `pytest -q -m "not gpu and not network"` passes, and the new behaviour has
      a test that can actually fail.
- [ ] No number in the diff is estimated. Measured numbers name the hardware and
      the command; anything not run is marked "not run" rather than omitted.
- [ ] No new required dependency in the base install; heavy imports stay lazy.
- [ ] No `torch.save`, no pickle, no `eval`, no shell execution, no network call
      at import time.
- [ ] Public functions have type hints and docstrings.
- [ ] If the trajectory, cache or benchmark schema changed, its
      `schema_version` was bumped and the reader rejects the old version.
- [ ] If this changes what a user sees, the README and the relevant `docs/` page
      were updated in the same commit.
- [ ] Configs, manifests, reports and logs pasted here were reviewed for tokens,
      private paths and proprietary data.

## If this touches the objective, the masks, or the cache

- [ ] I re-read `docs/math.md` and the claim I am making matches the code.
- [ ] Tool output still cannot enter the loss
      (`tests/unit/test_token_provenance.py`).
- [ ] Chunked and unchunked gradients still agree
      (`tests/unit/test_chunked_equivalence.py`).
- [ ] Resume is still exact (`tests/integration/test_resume_and_swap.py`).
