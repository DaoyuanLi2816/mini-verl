"""Publish compiler-bound compatibility reports for the pinned OPD profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from miniverl.bridge.contract import VERL_COMMIT, VERL_REPOSITORY, VERL_TAG
from miniverl.bridge.opd_v08 import load_verl_opd_v08
from miniverl.utils.runs import canonical_json


def build_matrix(source: Path) -> dict[str, Any]:
    compiled = load_verl_opd_v08(source)
    fields = [item.model_dump(mode="json") for item in compiled.compatibility]
    return {
        "schema_version": 2,
        "profile": compiled.profile,
        "upstream": compiled.upstream,
        "source_fixture": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiled_digest": compiled.compiled_digest,
        "executable": compiled.executable,
        "field_count": len(fields),
        "classification_counts": dict(
            sorted(Counter(item["classification"] for item in fields).items())
        ),
        "reinterpretation_acceptance": compiled.reinterpretation_acceptance,
        "fields": fields,
        "scope": (
            "config conformance for one documented local pure-OPD profile; "
            "not full verl compatibility or distributed execution"
        ),
    }


def build_official_report(root: Path) -> dict[str, object]:
    fixture = root / "tests/fixtures/verl/opd-v0.8-official-example-fields.yaml"
    plan = load_verl_opd_v08(fixture, require_executable=False)
    counts = Counter(item.classification for item in plan.compatibility)
    return {
        "schema_version": 1,
        "profile": plan.profile,
        "provenance": {
            "repository": VERL_REPOSITORY,
            "tag": VERL_TAG,
            "commit": VERL_COMMIT,
            "source_path": "examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh",
            "license": "Apache-2.0",
            "fixture": fixture.relative_to(root).as_posix(),
        },
        "official_example_fields_total": len(plan.compatibility),
        "classifications": {
            name: counts.get(name, 0)
            for name in (
                "exact",
                "semantically_conformant",
                "locally_reinterpreted",
                "derived",
                "informational_only",
                "unsupported",
            )
        },
        "executable": plan.executable,
        "unsupported_fields": [
            item.upstream_field
            for item in plan.compatibility
            if item.classification == "unsupported"
        ],
        "claim": (
            "Every fixture leaf is classified; coverage does not imply that "
            "policy-gradient, FSDP, or distributed execution is supported."
        ),
    }


def render_official_report(root: Path) -> str:
    return json.dumps(build_official_report(root), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("examples/verl-opd-v0.8-single-gpu.yaml")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/generated/verl-opd-v0.8-compatibility.json")
    )
    parser.add_argument(
        "--official-out",
        type=Path,
        default=Path("docs/generated/verl-opd-v08-official-fields.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rendered = canonical_json(build_matrix(args.source))
    official = render_official_report(root)
    if args.check:
        current = args.out.is_file() and args.out.read_text(encoding="utf-8") == rendered
        official_current = (
            args.official_out.is_file()
            and args.official_out.read_text(encoding="utf-8") == official
        )
        return 0 if current and official_current else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8", newline="\n")
    args.official_out.parent.mkdir(parents=True, exist_ok=True)
    args.official_out.write_text(official, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
