"""Validate the exact candidate-to-qualification release evidence chain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miniverl.release_chain import validate_release_chain  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--known-good-sha256", required=True)
    parser.add_argument("--required-gpu-name", required=True)
    args = parser.parse_args()
    problems = validate_release_chain(
        args.candidate_dir,
        args.candidate_manifest,
        args.qualification,
        expected_commit=args.commit,
        expected_known_good_sha256=args.known_good_sha256,
        required_gpu_name=args.required_gpu_name,
    )
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print("release candidate and GPU qualification form one exact byte-identity chain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
