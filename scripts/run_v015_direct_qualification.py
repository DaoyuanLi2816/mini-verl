"""Qualify direct upstream-config workflows using an installed candidate wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from importlib.resources import files
from pathlib import Path
from typing import Any

from miniverl import __version__
from miniverl.bridge.direct import DIRECT_PROFILE, direct_rules_digest
from miniverl.utils.runs import write_json_atomic


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    checkout = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout))
    origin = json.loads(
        importlib.metadata.distribution("miniverl").read_text("direct_url.json") or "{}"
    )
    if (
        head != args.commit
        or (dirty and not args.rehearsal)
        or digest(args.wheel) != args.wheel_sha256
        or origin.get("archive_info", {}).get("hashes", {}).get("sha256") != args.wheel_sha256
    ):
        raise RuntimeError(
            "direct qualification requires the exact installed candidate wheel/source"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("direct qualification requires exactly one CUDA device")
    args.work.mkdir(parents=True, exist_ok=False)
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update(PYTHONUTF8="1", MINIVERL_LOG_LEVEL="WARNING")
    commands = []

    def cli(*words: str, as_json: bool = True) -> Any:
        command = [*words, *(["--json"] if as_json else [])]
        result = subprocess.run(
            [
                str(
                    Path(sys.executable).with_name(
                        "miniverl.exe" if os.name == "nt" else "miniverl"
                    )
                ),
                *command,
            ],
            cwd=args.work,
            env=environment,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=900,
        )
        number = len(commands)
        (args.work / f"command-{number:02d}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (args.work / f"command-{number:02d}.stderr.log").write_text(result.stderr, encoding="utf-8")
        commands.append({"command": ["miniverl", *command], "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(
                f"direct command {number} failed: {command}\n{result.stderr[-4000:]}"
            )
        return json.loads(result.stdout) if as_json else None

    cli(
        "data",
        "sample",
        "--reward-profile",
        "target-length",
        "--rows",
        "8",
        "--out",
        "data/rl-prompts.parquet",
        as_json=False,
    )
    cases = {}
    offline = ["--offline"] if args.offline else []
    for algorithm in ("ppo", "grpo"):
        started = time.perf_counter()
        source_name = f"verl-{algorithm}.yaml"
        source = args.work / source_name
        original = (
            files("miniverl.resources").joinpath(f"verl_direct_{algorithm}.yaml").read_bytes()
        )
        source.write_bytes(original)  # exact packaged upstream-shaped input; no native YAML editing
        bindings = ["--bind", "reward.provider=target_length"]
        dry = cli("run", source_name, *bindings, "--dry-run")
        if dry["semantic_status"] != "accepted" or dry["profile"] != DIRECT_PROFILE:
            raise RuntimeError("direct source was not automatically accepted by v4")
        run_root = f"runs/direct-{algorithm}"
        result = cli(
            "run",
            source_name,
            *bindings,
            "--output",
            "runs",
            "--run-id",
            f"direct-{algorithm}",
            *offline,
        )
        summary = cli("inspect", run_root)
        cli("report", run_root)
        final = args.work / run_root / "checkpoints/final"
        before = {path.name: digest(path) for path in final.glob("*.safetensors")}
        cli(
            "run",
            source_name,
            *bindings,
            "--resume-from",
            f"{run_root}/checkpoints/step-000002",
            *offline,
        )
        after = {path.name: digest(path) for path in final.glob("*.safetensors")}
        resumed = cli("inspect", run_root)
        if not before or before != after:
            raise RuntimeError(f"direct {algorithm} exact tensor replay failed")
        if resumed["updates"]["actor_records"] != 4 or resumed["updates"]["critic_records"] != (
            4 if algorithm == "ppo" else 0
        ):
            raise RuntimeError("direct replay changed actor/critic update counts")
        if resumed["rollouts"]["trajectories"] != 8 or resumed["rewards"]["count"] != 8:
            raise RuntimeError("direct replay changed sample/reward counts")
        cli("export-adapter", "--run", run_root, "--out", f"{run_root}/model", *offline)
        cli(
            "export-verl",
            "--run",
            run_root,
            "--target-verl",
            "v0.9.0",
            "--out",
            f"{algorithm}-handoff",
        )
        doctor = cli("bridge", "doctor", f"{algorithm}-handoff")
        compatibility = json.loads(
            (args.work / f"{algorithm}-handoff/provenance/compatibility-report.json").read_text()
        )
        if (
            doctor["verdict"] != "ok"
            or compatibility["launchable"]
            or compatibility["distributed_execution_tested"]
        ):
            raise RuntimeError("direct handoff validation/readiness failed")
        if (
            source.read_bytes() != original
            or (args.work / run_root / "verl-source.yaml").read_bytes() != original
        ):
            raise RuntimeError("direct run did not retain the exact upstream source bytes")
        if (
            summary["memory"]["status"] != "measured"
            or not 0 < summary["memory"]["peak_reserved_gib"] < 14.5
        ):
            raise RuntimeError("direct workflow exceeded the qualification envelope")
        report = json.loads((args.work / run_root / "verl-direct-report.json").read_text())
        cases[algorithm] = {
            "profile": result["profile"],
            "field_rules_sha256": direct_rules_digest(),
            "semantic_status": result["semantic_status"],
            "execution_status": result["execution_status"],
            "source_config_sha256": digest(source),
            "source_bytes_preserved": True,
            "manual_native_yaml_edits": False,
            "runtime_bindings": report["runtime_bindings"],
            "hardware_plan": report["hardware_plan"],
            "ir_sha256": report["ir_sha256"],
            "loss_aggregation": report["resolved_native_config"]["loss"]["aggregation"],
            "schedule": report["schedule"],
            "updates": summary["updates"],
            "rollouts": summary["rollouts"],
            "rewards": summary["rewards"],
            "actor_metrics": summary["actor_metrics"],
            "policy_loss": summary["policy_loss"],
            "value_loss": summary["value_loss"],
            "memory": summary["memory"],
            "models": summary["models"],
            "exact_tensor_resume": True,
            "tensor_hashes": after,
            "resume_counts_verified": True,
            "handoff_verdict": doctor["verdict"],
            "launchable": False,
            "distributed_execution_tested": False,
            "workflow_seconds": time.perf_counter() - started,
        }
    return {
        "schema_version": 1,
        "kind": "installed_direct_verl_workflows",
        "status": "passed",
        "source_commit": args.commit,
        "miniverl_version": __version__,
        "wheel_sha256": args.wheel_sha256,
        "installed_wheel_verified": True,
        "source_dirty": dirty,
        "rehearsal": args.rehearsal,
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cases": cases,
        "commands": commands,
        "scientific_scope": "bounded explicitly rebound length-reward systems workflows; no task-quality claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="Dirty development source is never release evidence.",
    )
    args = parser.parse_args()
    args.work = args.work.resolve()
    write_json_atomic(args.output, run(args))


if __name__ == "__main__":
    main()
