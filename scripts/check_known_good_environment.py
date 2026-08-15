"""Check the measured CUDA manifest, constraints and install documentation agree."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

MANIFEST = Path("environments/known-good-rtx4080-cu130.json")
REQUIRED_PACKAGES = {
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "numpy",
    "pyarrow",
    "safetensors",
}


def _constraints(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = line.split(";", 1)[0].strip()
        if "==" not in requirement:
            raise ValueError(f"constraint is not exact: {raw}")
        name, version = requirement.split("==", 1)
        versions[name.strip().lower()] = version.strip()
    return versions


def check_known_good_environment(root: Path) -> list[str]:
    problems: list[str] = []
    try:
        payload: dict[str, Any] = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"manifest: {exc}"]
    if payload.get("schema_version") != 1:
        problems.append("manifest: schema_version must be 1")
    if payload.get("status") != "maintainer_measured":
        problems.append("manifest: status must be maintainer_measured")
    packages = payload.get("packages")
    if not isinstance(packages, dict):
        return [*problems, "manifest: packages must be an object"]
    missing = sorted(REQUIRED_PACKAGES - packages.keys())
    if missing:
        problems.append("manifest: missing packages " + ", ".join(missing))
    constraints_value = payload.get("constraints")
    if not isinstance(constraints_value, str):
        return [*problems, "manifest: constraints must be a path"]
    constraints_path = root / constraints_value
    try:
        pinned = _constraints(constraints_path)
    except (OSError, UnicodeError, ValueError) as exc:
        return [*problems, f"constraints: {exc}"]
    for package, expected in packages.items():
        actual = pinned.get(package.lower())
        if actual != expected:
            problems.append(
                f"constraints: {package} is {actual!r}, expected manifest value {expected!r}"
            )
    pytorch = payload.get("pytorch") or {}
    if pytorch.get("version") != packages.get("torch"):
        problems.append("manifest: pytorch.version and packages.torch disagree")
    if not str(pytorch.get("index_url", "")).startswith("https://download.pytorch.org/whl/"):
        problems.append("manifest: PyTorch index must be an official CUDA wheel index")
    private = re.compile(r"(?i)(?:[a-z]:\\users\\|/home/|onedrive|token|credential)")
    if private.search(json.dumps(payload, sort_keys=True)):
        problems.append("privacy: known-good manifest contains private text")
    guide = " ".join((root / "docs/single-gpu-guide.md").read_text(encoding="utf-8").split())
    for required in (
        constraints_value,
        str(pytorch.get("index_url")),
        f"torch=={pytorch.get('version')}",
        "other GPUs are unmeasured",
    ):
        if required not in guide:
            problems.append(f"docs: single-GPU guide is missing {required!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    problems = check_known_good_environment(args.root)
    payload = {"valid": not problems, "problems": problems}
    print(
        json.dumps(payload, sort_keys=True)
        if args.json
        else "\n".join(problems) or "known-good environment agrees"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
