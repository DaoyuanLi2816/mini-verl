"""Build or validate the single miniVERL release-candidate byte set."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miniverl import __version__  # noqa: E402
from miniverl.release_candidate import (  # noqa: E402
    build_release_candidate,
    validate_candidate_directory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--repository")
    parser.add_argument("--workflow-path")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--run-attempt", type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    commit = args.commit
    if commit is None:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if args.check:
        problems = validate_candidate_directory(
            args.output,
            expected_commit=commit,
            expected_version=__version__,
            expected_repository=args.repository,
            expected_workflow_path=args.workflow_path,
            expected_run_id=args.run_id,
            expected_run_attempt=args.run_attempt,
        )
        print(json.dumps({"valid": not problems, "problems": problems}, sort_keys=True))
        return 1 if problems else 0
    record = build_release_candidate(args.output, source_commit=commit, project_root=ROOT)
    print(json.dumps(record.model_dump(mode="json"), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
