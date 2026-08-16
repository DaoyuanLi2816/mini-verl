"""Run the non-publishing miniVERL pre-release gate and emit a strict JSON summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Gate:
    name: str
    command: tuple[str, ...]
    timeout: int = 1800


def gate_plan(
    *,
    candidate_dir: Path,
    candidate_wheel: Path,
    candidate_sdist: Path,
    qualification: Path,
    site: Path,
    screenshots: Path,
    source_commit: str | None = None,
    known_good_sha256: str | None = None,
) -> list[Gate]:
    python = sys.executable
    source_commit = source_commit or _git_head()
    known_good_sha256 = known_good_sha256 or _sha256(
        ROOT / "environments/known-good-rtx4080-cu130.json"
    )
    return [
        Gate("release_state", (python, "scripts/release_state.py", "--check")),
        Gate("pypi_readme", (python, "scripts/build_pypi_readme.py", "--check")),
        Gate("known_good_environment", (python, "scripts/check_known_good_environment.py")),
        Gate(
            "qualification_schema",
            (python, "scripts/publish_gpu_qualification_schema.py", "--check"),
        ),
        Gate(
            "candidate_artifact",
            (
                python,
                "scripts/build_release_candidate.py",
                "--output",
                str(candidate_dir),
                "--commit",
                source_commit,
                "--check",
            ),
        ),
        Gate("ruff_check", (python, "-m", "ruff", "check", ".")),
        Gate("ruff_format", (python, "-m", "ruff", "format", "--check", ".")),
        Gate("mypy", (python, "-m", "mypy", "src/miniverl")),
        Gate("actionlint", ("actionlint",)),
        Gate("markdown_links", (python, "scripts/check_markdown_links.py")),
        Gate("text_integrity", (python, "scripts/check_text_integrity.py")),
        Gate(
            "generated_alignment_lab",
            (python, "scripts/publish_alignment_lab_artifacts.py", "--check"),
        ),
        Gate(
            "generated_bridge_diagrams",
            (python, "scripts/publish_verl_bridge_diagrams.py", "--check"),
        ),
        Gate(
            "generated_compatibility",
            (python, "scripts/publish_verl_opd_compatibility.py", "--check"),
        ),
        Gate(
            "generated_hardware_schema",
            (python, "scripts/publish_hardware_record_schema.py", "--check"),
        ),
        Gate(
            "cpu_tests_coverage",
            (
                python,
                "-m",
                "pytest",
                "-q",
                "-m",
                "not gpu and not network",
                "--cov=miniverl",
                "--cov-report=term-missing",
                "--cov-fail-under=80",
            ),
            7200,
        ),
        Gate("docs_build", ("mkdocs", "build", "--strict", "--site-dir", str(site))),
        Gate(
            "docs_visual",
            (
                python,
                "scripts/check_docs_visual.py",
                "--site",
                str(site),
                "--screenshots",
                str(screenshots),
            ),
            3600,
        ),
        Gate(
            "twine",
            (python, "-m", "twine", "check", str(candidate_wheel), str(candidate_sdist)),
        ),
        Gate(
            "release_evidence_chain",
            (
                python,
                "scripts/validate_release_chain.py",
                "--candidate-dir",
                str(candidate_dir),
                "--candidate-manifest",
                str(candidate_dir / "candidate-manifest.json"),
                "--qualification",
                str(qualification),
                "--commit",
                source_commit,
                "--known-good-sha256",
                known_good_sha256,
                "--required-gpu-name",
                "NVIDIA GeForce RTX 4080",
            ),
        ),
    ]


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_gate(gate: Gate, env: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    command = list(gate.command)
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=gate.timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "name": gate.name,
            "status": "failed",
            "seconds": round(time.perf_counter() - started, 3),
            "detail": str(exc),
        }
    output = (completed.stdout + completed.stderr).strip()
    return {
        "name": gate.name,
        "status": "passed" if completed.returncode == 0 else "failed",
        "seconds": round(time.perf_counter() - started, 3),
        "returncode": completed.returncode,
        "detail": output[-4000:],
    }


def _base_wheel_smoke(candidate_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    wheels = list(candidate_dir.glob("*.whl"))
    if len(wheels) != 1:
        return {
            "name": "clean_base_wheel_install",
            "status": "failed",
            "seconds": 0.0,
            "detail": f"expected one wheel, found {len(wheels)}",
        }
    with tempfile.TemporaryDirectory(prefix="miniverl-base-wheel-") as temporary:
        venv = Path(temporary) / "venv"
        commands = [[sys.executable, "-m", "venv", str(venv)]]
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        commands.extend(
            [
                [str(python), "-m", "pip", "install", str(wheels[0])],
                [
                    str(python),
                    "-c",
                    (
                        "import json,sys,miniverl,miniverl.cli; "
                        "assert 'torch' not in sys.modules; "
                        "print(json.dumps({'version':miniverl.__version__}))"
                    ),
                ],
            ]
        )
        for command in commands:
            completed = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, timeout=600
            )
            if completed.returncode:
                return {
                    "name": "clean_base_wheel_install",
                    "status": "failed",
                    "seconds": round(time.perf_counter() - started, 3),
                    "detail": (completed.stdout + completed.stderr)[-4000:],
                }
    return {
        "name": "clean_base_wheel_install",
        "status": "passed",
        "seconds": round(time.perf_counter() - started, 3),
        "detail": "wheel imported without torch",
    }


def _tree_clean(summary: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in completed.stdout.splitlines() if str(summary) not in line]
    return {
        "name": "working_tree_clean",
        "status": "passed" if not lines else "failed",
        "seconds": 0.0,
        "detail": "clean" if not lines else "\n".join(lines),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--summary", type=Path, default=ROOT / "runs/release-gate-summary.json")
    parser.add_argument("--list", action="store_true", help="Print gate names without running.")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    required = (args.candidate_dir, args.candidate_manifest, args.qualification)
    if not args.list and any(value is None for value in required):
        parser.error(
            "--candidate-dir, --candidate-manifest and --qualification are required "
            "unless --list is used"
        )
    with tempfile.TemporaryDirectory(prefix="miniverl-release-gate-") as temporary:
        root = Path(temporary)
        qualification = args.qualification or root / "qualification-not-used.json"
        candidate_dir = args.candidate_dir or root / "candidate-not-used"
        candidate_manifest = args.candidate_manifest or candidate_dir / "candidate-manifest.json"
        if args.list:
            candidate_wheel = candidate_dir / "not-used.whl"
            candidate_sdist = candidate_dir / "not-used.tar.gz"
            wheel_sha256 = "not-used-by-list"
            manifest_sha256 = "not-used-by-list"
        else:
            sys.path.insert(0, str(ROOT / "src"))
            from miniverl.release_candidate import load_candidate_manifest

            expected_manifest = candidate_dir / "candidate-manifest.json"
            if candidate_manifest.resolve() != expected_manifest.resolve():
                parser.error(
                    "--candidate-manifest must name candidate-manifest.json inside --candidate-dir"
                )
            record = load_candidate_manifest(candidate_manifest)
            candidate_wheel = candidate_dir / record.wheel.filename
            candidate_sdist = candidate_dir / record.sdist.filename
            wheel_sha256 = record.wheel.sha256
            manifest_sha256 = _sha256(candidate_manifest)
        plan = gate_plan(
            candidate_dir=candidate_dir.resolve(),
            candidate_wheel=candidate_wheel.resolve(),
            candidate_sdist=candidate_sdist.resolve(),
            qualification=qualification.resolve(),
            site=root / "site",
            screenshots=root / "screenshots",
            source_commit="not-used-by-list" if args.list else None,
            known_good_sha256="not-used-by-list" if args.list else None,
        )
        if args.list:
            print(json.dumps([gate.name for gate in plan]))
            return 0
        env = dict(os.environ)
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        results: list[dict[str, Any]] = []
        for gate in plan:
            result = _run_gate(gate, env)
            results.append(result)
            print(f"{result['status']}: {gate.name}")
            if result["status"] == "failed" and args.fail_fast:
                break
        if all(item["status"] == "passed" for item in results) and len(results) == len(plan):
            install = _base_wheel_smoke(candidate_dir)
            results.append(install)
            print(f"{install['status']}: {install['name']}")
        else:
            results.append(
                {
                    "name": "clean_base_wheel_install",
                    "status": "not_run",
                    "seconds": 0.0,
                    "detail": "an earlier gate failed",
                }
            )
        results.append(_tree_clean(args.summary))
        status = "passed" if all(item["status"] == "passed" for item in results) else "failed"
        payload = {
            "schema_version": 1,
            "kind": "miniverl_release_gate",
            "status": status,
            "source_commit": _git_head(),
            "candidate_manifest_sha256": manifest_sha256,
            "candidate_wheel_sha256": wheel_sha256,
            "created_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "publishes": False,
            "checks": results,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(args.summary)
        return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
