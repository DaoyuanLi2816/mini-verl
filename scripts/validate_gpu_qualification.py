"""Validate one GPU qualification artifact without importing torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from miniverl.qualification import validate_qualification_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualification", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--wheel-sha256")
    parser.add_argument("--known-good-sha256")
    parser.add_argument("--required-gpu-name")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    problems = validate_qualification_file(
        args.qualification,
        expected_commit=args.commit,
        expected_wheel_sha256=args.wheel_sha256,
        expected_known_good_sha256=args.known_good_sha256,
        required_gpu_name=args.required_gpu_name,
    )
    payload = {"valid": not problems, "problems": problems}
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    elif problems:
        print("GPU qualification is invalid:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("GPU qualification is valid and artifact-bound")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
