"""Publish the pinned OPD field matrix from the typed compiler."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from miniverl.bridge.opd_v08 import load_verl_opd_v08
from miniverl.utils.runs import canonical_json


def build_matrix(source: Path) -> dict[str, Any]:
    compiled = load_verl_opd_v08(source)
    fields = [item.model_dump(mode="json") for item in compiled.compatibility]
    return {
        "schema_version": 1,
        "profile": compiled.profile,
        "upstream": compiled.upstream,
        "source_fixture": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiled_digest": compiled.compiled_digest,
        "executable": compiled.executable,
        "field_count": len(fields),
        "classification_counts": dict(
            sorted(Counter(item["classification"] for item in fields).items())
        ),
        "fields": fields,
        "scope": (
            "config conformance for one documented local pure-OPD profile; "
            "not full verl compatibility or distributed execution"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("examples/verl-opd-v0.8-single-gpu.yaml")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/generated/verl-opd-v0.8-compatibility.json")
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = canonical_json(build_matrix(args.source))
    if args.check:
        return 0 if args.out.read_text(encoding="utf-8") == rendered else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
