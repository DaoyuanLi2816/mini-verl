"""Generate the committed strict GPU qualification JSON Schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from miniverl.qualification import qualification_json_schema

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs/generated/gpu-qualification-v1.schema.json"


def render() -> str:
    return json.dumps(qualification_json_schema(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not TARGET.is_file() or TARGET.read_text(encoding="utf-8") != expected:
            print(f"stale generated schema: {TARGET.relative_to(ROOT).as_posix()}")
            return 1
        print("GPU qualification schema is current")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(expected, encoding="utf-8", newline="\n")
    print(TARGET.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
