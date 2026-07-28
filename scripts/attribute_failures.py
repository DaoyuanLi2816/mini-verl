"""Post-hoc failure attribution for a finished run's evaluation trajectories.

A single success rate cannot distinguish "the policy cannot do the task" from
"the policy can do the task but writes the answer in a shape the verifier does
not accept".  Both score zero, and they call for opposite fixes.

This script re-reads ``eval_trajectories.jsonl`` and re-scores every failure
under a *lenient* parser that unwraps ``<answer>...</answer>`` before comparing
numbers.  The gap between the strict and lenient rates is the share of failures
that are purely presentational.

The strict number is always the reported one.  This is an analysis aid, not an
alternative metric: never quote the lenient rate without the strict rate beside
it.

    python scripts/attribute_failures.py runs/my-run
    python scripts/attribute_failures.py runs/benchmarks/*-s1234 --json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

from miniverl.evaluation.diagnostics import lenient_answer_matches


def attribute(run_dir: Path, last_n: int | None) -> dict[str, Any]:
    """Summarize the strict and lenient outcomes of one run directory."""
    path = run_dir / "eval_trajectories.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no eval_trajectories.jsonl in {run_dir}")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"{path} is empty")
    # A run that evaluates periodically appends every evaluation to one file;
    # the final evaluation is the last `last_n` rows.
    if last_n:
        rows = rows[-last_n:]

    strict = sum(1 for row in rows if row["verification"]["solved"])
    presentational = 0
    categories: collections.Counter[str] = collections.Counter()
    for row in rows:
        verification = row["verification"]
        categories[verification.get("failure_category") or "unknown"] += 1
        if verification["solved"]:
            continue
        if lenient_answer_matches(
            verification.get("predicted") or "",
            verification.get("expected") or "",
        ):
            presentational += 1

    total = len(rows)
    return {
        "run": run_dir.name,
        "trajectories": total,
        "strict_solved": strict,
        "strict_rate": strict / total,
        "presentational_failures": presentational,
        "lenient_solved": strict + presentational,
        "lenient_rate": (strict + presentational) / total,
        "substantive_failures": total - strict - presentational,
        "failure_categories": dict(sorted(categories.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="run directories to analyse")
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="N",
        help="use only the last N trajectories (the final evaluation of a periodic run)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    reports = []
    for run_dir in args.run_dirs:
        try:
            reports.append(attribute(run_dir, args.last))
        except (FileNotFoundError, ValueError) as exc:
            print(f"skipped: {exc}", file=sys.stderr)
    if not reports:
        return 1

    if args.json:
        print(json.dumps(reports, indent=2))
        return 0

    width = max(len(r["run"]) for r in reports)
    print(
        f"{'run':{width}s} {'n':>4s} {'strict':>8s} {'lenient':>8s} {'presentational':>15s} {'substantive':>12s}"
    )
    for r in reports:
        print(
            f"{r['run']:{width}s} {r['trajectories']:>4d} {r['strict_rate']:>7.1%} "
            f"{r['lenient_rate']:>8.1%} {r['presentational_failures']:>15d} {r['substantive_failures']:>12d}"
        )
    print("\nstrict is the reported metric; lenient unwraps <answer> tags and is diagnostic only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
