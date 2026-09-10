"""Package the pinned corpus leaf schema; this does not grant field acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            result.update(flatten(item, path))
        else:
            result[path] = item
    return result


def generate():
    sources = {}
    fields = {}
    for path in sorted((ROOT / "benchmarks/compatibility/verl-v0.9.0").glob("*-complete.yaml")):
        content = path.read_bytes()
        sources[path.name] = hashlib.sha256(content).hexdigest()
        for name, value in flatten(yaml.safe_load(content)).items():
            values = fields.setdefault(name, [])
            if value not in values:
                values.append(value)
    return json.dumps({"sources": sources, "fields": fields}, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "src/miniverl/resources/verl_v09_direct_schema.json"
    generated = generate().encode()
    if args.check:
        if not target.is_file() or target.read_bytes() != generated:
            raise SystemExit("direct schema is stale")
    else:
        target.write_bytes(generated)
