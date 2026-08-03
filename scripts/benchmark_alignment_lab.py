#!/usr/bin/env python3
"""Run one frozen Alignment Lab v1 arm from its preregistration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from miniverl.alignment import run_alignment
from miniverl.alignment.benchmark import (
    ALIGNMENT_BENCHMARK_METHODS,
    build_alignment_benchmark_config,
    load_alignment_preregistration,
)
from miniverl.config import RunConfig


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-recipe", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256")
    parser.add_argument("--starting-checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=ALIGNMENT_BENCHMARK_METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("eval", "test"), required=True)
    parser.add_argument("--dpo-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/alignment-lab-v1"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    preregistration = load_alignment_preregistration(
        args.preregistration,
        expected_sha256=args.preregistration_sha256,
    )
    config = build_alignment_benchmark_config(
        RunConfig.from_yaml(args.base_recipe),
        preregistration,
        method=args.method,
        seed=args.seed,
        split=args.split,
        starting_checkpoint=args.starting_checkpoint,
        dpo_manifest=args.dpo_manifest,
    )
    run_id = f"{args.split}-{args.method}-seed-{args.seed}"
    if args.dry_run:
        print(config.to_yaml(), end="")
        return
    result = run_alignment(
        config,
        output_dir=args.output,
        run_id=run_id,
        local_files_only=args.local_files_only,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
