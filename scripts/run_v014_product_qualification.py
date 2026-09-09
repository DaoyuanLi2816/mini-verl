"""Exercise the documented installed PPO/GRPO CLI workflow on the exact candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from miniverl import __version__
from miniverl.utils.runs import write_json_atomic


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("product qualification requires one CUDA device")
    if sha256(args.wheel) != args.wheel_sha256:
        raise RuntimeError("candidate wheel checksum mismatch")
    checkout = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True).strip()
    )
    if head != args.commit or (dirty and not args.rehearsal):
        raise RuntimeError("product qualification requires the clean exact candidate commit")
    origin = json.loads(
        importlib.metadata.distribution("miniverl").read_text("direct_url.json") or "{}"
    )
    if origin.get("archive_info", {}).get("hashes", {}).get("sha256") != args.wheel_sha256:
        raise RuntimeError("installed miniVERL is not the exact candidate wheel")
    args.work.mkdir(parents=True, exist_ok=False)
    commands = []
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment["PYTHONUTF8"] = "1"
    environment["MINIVERL_LOG_LEVEL"] = "WARNING"

    def cli(*words: str, as_json: bool = True) -> Any:
        command = [*words, *(["--json"] if as_json else [])]
        completed = subprocess.run(
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
        (args.work / f"command-{number:02d}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (args.work / f"command-{number:02d}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        commands.append({"command": ["miniverl", *command], "exit_code": completed.returncode})
        if completed.returncode:
            raise RuntimeError(
                f"documented command {number} failed: {' '.join(command)}\n{completed.stderr[-4000:]}"
            )
        return json.loads(completed.stdout) if as_json else None

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
    for algorithm, updates, checkpoint in (("ppo", 4, "step-000002"), ("grpo", 2, "step-000001")):
        started = time.perf_counter()
        recipe = f"local-{algorithm}.yaml"
        run_root = f"runs/local-{algorithm}"
        imported = cli(
            "import-verl",
            "--profile",
            "verl-rl-v0.9-single-gpu-v3",
            "--example",
            algorithm,
            "--out",
            recipe,
        )
        if imported["status"] != "accepted":
            raise RuntimeError(f"{algorithm} example did not import")
        cli("validate", recipe)
        planned = cli("train", recipe, "--dry-run")
        if planned["planned_optimizer_steps"] != updates:
            raise RuntimeError("incorrect actor schedule plan")
        result = cli("train", recipe, "--run-id", f"local-{algorithm}", *offline)
        summary = cli("inspect", run_root)
        cli("inspect", run_root, as_json=False)
        cli("report", run_root)
        final = args.work / run_root / "checkpoints/final"
        before = {path.name: sha256(path) for path in final.glob("*.safetensors")}
        cli("train", recipe, "--resume-from", f"{run_root}/checkpoints/{checkpoint}", *offline)
        after = {path.name: sha256(path) for path in final.glob("*.safetensors")}
        if not before or before != after:
            raise RuntimeError(f"{algorithm} actor/critic/optimizer replay differs")
        resumed = cli("inspect", run_root)
        if (
            resumed["updates"]["actor_records"] != updates
            or resumed["rollouts"]["trajectories"] != 8
        ):
            raise RuntimeError("resumed records double-count or lose samples")
        if resumed["updates"]["critic_records"] != (updates if algorithm == "ppo" else 0):
            raise RuntimeError("incorrect resumed critic count")
        cli("export-adapter", "--run", run_root, "--out", f"{run_root}/model", *offline)
        bundle = cli(
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
            raise RuntimeError("handoff inspection misrepresents readiness")
        if (
            summary["memory"]["status"] != "measured"
            or summary["memory"]["peak_reserved_gib"] >= 14.5
        ):
            raise RuntimeError("missing GPU measurement or exceeded qualification envelope")
        cases[algorithm] = {
            "import_status": imported["status"],
            "profile": imported["profile"],
            "field_rules_sha256": imported["profile_field_rules_sha256"],
            "recipe_sha256": imported["generated_recipe_sha256"],
            "updates": summary["updates"],
            "rollouts": summary["rollouts"],
            "rewards": summary["rewards"],
            "actor_metrics": summary["actor_metrics"],
            "policy_loss": summary["policy_loss"],
            "value_loss": summary["value_loss"],
            "memory": summary["memory"],
            "models": summary["models"],
            "training_seconds": result["duration_seconds"],
            "workflow_seconds": time.perf_counter() - started,
            "exact_tensor_resume": True,
            "tensor_hashes": after,
            "resume_counts_verified": True,
            "handoff_verdict": doctor["verdict"],
            "launchable": compatibility["launchable"],
            "distributed_execution_tested": False,
            "bundle_schema_version": bundle.get("schema_version"),
        }
    return {
        "schema_version": 1,
        "kind": "installed_ppo_grpo_product_workflows",
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
        "scientific_scope": "bounded length-reward systems qualification; no external task-quality comparison",
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
        help="Allow a dirty development checkout; never release evidence.",
    )
    args = parser.parse_args()
    args.work = args.work.resolve()
    write_json_atomic(args.output, run(args))


if __name__ == "__main__":
    main()
