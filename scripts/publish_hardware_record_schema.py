"""Publish the JSON Schema for portable hardware records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from miniverl.evidence.hardware import HardwareRecord

DEFAULT_OUT = Path("docs/generated/hardware-record-v1.schema.json")


def render() -> str:
    """Return deterministic schema bytes."""
    return (
        json.dumps(
            HardwareRecord.model_json_schema(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != expected:
            parser.error(f"generated hardware schema is stale: {args.out}")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(expected, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
