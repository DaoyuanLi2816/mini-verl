"""Package immutable upstream Hydra YAML data (not Python resolution code)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from miniverl.algorithms.contract import UPSTREAM_VERL_COMMIT, UPSTREAM_VERL_TAG

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "src/miniverl/resources/verl_v09_hydra_tree.json"


def generated(upstream: Path) -> bytes:
    prefix = "verl/trainer/config/"
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", UPSTREAM_VERL_COMMIT, "--", prefix],
        cwd=upstream,
        text=True,
    ).splitlines()
    files = {}
    for name in sorted(names):
        if not name.endswith(".yaml"):
            continue
        content = subprocess.check_output(
            ["git", "show", f"{UPSTREAM_VERL_COMMIT}:{name}"],
            cwd=upstream,
        )
        files[name.removeprefix(prefix)] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "text": content.decode("utf-8"),
        }
    return (
        json.dumps(
            {
                "schema_version": 1,
                "upstream_tag": UPSTREAM_VERL_TAG,
                "upstream_commit": UPSTREAM_VERL_COMMIT,
                "source": "https://github.com/verl-project/verl",
                "license": "Apache-2.0",
                "files": files,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = generated(args.upstream)
    if args.check:
        if DESTINATION.read_bytes() != content:
            raise SystemExit("pinned Hydra config tree is stale")
    else:
        DESTINATION.write_bytes(content)


if __name__ == "__main__":
    main()
