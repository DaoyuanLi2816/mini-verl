# Verified-bridge demo recording script

This recording is an artifact-only CPU demo. It needs the `bridge` extra but
does not need CUDA, model downloads, Ray or a verl installation.

## Before recording

```bash
python -m pip install "miniverl[bridge]"
python scripts/prepare_verl_bridge_smoke.py --out _demo-source
```

Keep the terminal at least 110 columns wide. Start the recording after setup so
the visible flow fits in 60–90 seconds.

## Recording flow

```bash
# 0–10 s: show the exact compatibility target
miniverl --version
cat _demo-source/README.md

# 10–35 s: make the transactional standard-artifact bundle
miniverl export-verl --run _demo-source --target-verl v0.8.0 \
  --out _demo-bundle

# 35–65 s: inspect model, tokenizer, Parquet, config, reward, privacy and hashes
miniverl bridge doctor _demo-bundle --json

# 65–80 s: show the pin and the unsupported-semantics boundary
cat _demo-bundle/recipe/REQUIRED_VERL.txt
python -c "import json; p=json.load(open('_demo-bundle/provenance/compatibility-report.json')); print(p['unsupported_semantics']); print('launchable:', p['launchable']); print('distributed tested:', p['distributed_execution_tested'])"

# 80–90 s: show the portable tree
python -c "from pathlib import Path; print('\n'.join(p.as_posix() for p in sorted(Path('_demo-bundle').rglob('*')) if p.is_file()))"
```

Narration: “This proves that the handoff is pinned, structurally checked and
checksummed. It does not prove a distributed verl job ran: the reward scaffold
still fails closed and the bundle is not launchable.”

The release's stronger compatibility evidence additionally installs the exact
official verl commit and uses OmegaConf plus PEFT to load the relevant surfaces.
Its machine-readable record is
[`generated/verl-bridge-smoke.json`](generated/verl-bridge-smoke.json).
