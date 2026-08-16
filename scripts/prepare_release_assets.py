"""Build or check the canonical future GitHub Release asset layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from miniverl.release_assets import check_release_assets, prepare_release_assets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--qualification-root", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        problems = check_release_assets(args.output)
        if problems:
            raise SystemExit("\n".join(problems))
        print(f"canonical release assets valid: {args.output}")
        return 0
    missing = [
        name
        for name in (
            "candidate_dir",
            "candidate_manifest",
            "qualification_root",
            "qualification",
            "verification",
        )
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "build mode requires: " + ", ".join(name.replace("_", "-") for name in missing)
        )
    prepare_release_assets(
        args.candidate_dir,
        args.candidate_manifest,
        args.qualification_root,
        args.qualification,
        args.verification,
        args.output,
    )
    print(f"canonical release assets prepared: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
