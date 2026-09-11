"""Qualify native Hydra PPO/GRPO using the exact installed candidate wheel."""

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
from miniverl.bridge.direct_v5 import PROFILE, rules_digest
from miniverl.bridge.hydra import compose_verl
from miniverl.bridge.hydra_examples import qualification_overrides
from miniverl.utils.runs import write_json_atomic


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def journal_digest(path: Path) -> str:
    if path.name != "rewards.jsonl":
        return digest(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row.pop("duration_ms", None)
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()


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
        raise RuntimeError("Hydra qualification requires exact installed candidate wheel/source")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Hydra qualification requires exactly one CUDA device")
    args.work.mkdir(parents=True, exist_ok=False)
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update(PYTHONUTF8="1", MINIVERL_LOG_LEVEL="WARNING")
    commands = []

    def cli(*words: str, as_json: bool = True) -> Any:
        command = [*words, *(["--json"] if as_json else [])]
        executable = str(
            Path(sys.executable).with_name("miniverl.exe" if os.name == "nt" else "miniverl")
        )
        result = subprocess.run(
            [executable, *command],
            cwd=args.work,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=1800,
        )
        number = len(commands)
        (args.work / f"command-{number:02d}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (args.work / f"command-{number:02d}.stderr.log").write_text(result.stderr, encoding="utf-8")
        commands.append({"command": ["miniverl", *command], "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(
                f"Hydra command {number} failed: {result.stdout[-2000:]}\n{result.stderr[-4000:]}"
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
        overrides = qualification_overrides(algorithm)
        expected = compose_verl(config_name="ppo_trainer", overrides=overrides)
        prefix = [
            "run",
            "--verl-config-name",
            "ppo_trainer",
            "--bind",
            "reward.provider=target_length",
            "--bind",
            "sampling.max_attempted_groups=64",
        ]
        dry = cli(*prefix, "--dry-run", *overrides)
        if (
            dry["semantic_status"] != "accepted"
            or dry["composition_status"] != "resolved"
            or dry["profile"] != PROFILE
        ):
            raise RuntimeError("native Hydra workflow not accepted by v5")
        run_root = f"runs/hydra-{algorithm}"
        result = cli(
            *prefix, "--output", "runs", "--run-id", f"hydra-{algorithm}", *offline, *overrides
        )
        summary = cli("inspect", run_root)
        cli("report", run_root)
        root = args.work / run_root
        final = root / "checkpoints/final"
        before = {path.name: digest(path) for path in final.glob("*.safetensors")}
        journals = {
            name: journal_digest(root / name)
            for name in ("trajectories.jsonl", "rewards.jsonl", "advantages.jsonl")
        }
        write_json_atomic(
            args.work / f"{algorithm}-before-resume.json", {"tensors": before, "journals": journals}
        )
        cli(*prefix, "--resume-from", f"{run_root}/checkpoints/step-000002", *offline, *overrides)
        resumed = cli("inspect", run_root)
        after = {path.name: digest(path) for path in final.glob("*.safetensors")}
        write_json_atomic(
            args.work / f"{algorithm}-after-resume.json",
            {
                "tensors": after,
                "journals": {name: journal_digest(root / name) for name in journals},
            },
        )
        if (
            not before
            or before != after
            or journals != {name: journal_digest(root / name) for name in journals}
        ):
            raise RuntimeError(f"Hydra {algorithm} exact tensor/journal replay failed")
        if resumed["updates"]["actor_records"] != 4 or resumed["updates"]["critic_records"] != (
            4 if algorithm == "ppo" else 0
        ):
            raise RuntimeError("Hydra replay changed actor/critic update counts")
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
            raise RuntimeError("Hydra handoff status failed")
        if (root / "verl-source.yaml").read_bytes() != expected.resolved_yaml:
            raise RuntimeError("Hydra resolved bytes were not preserved")
        if (
            summary["memory"]["status"] != "measured"
            or not 0 < summary["memory"]["peak_reserved_gib"] < 14.5
        ):
            raise RuntimeError("Hydra qualification memory envelope exceeded")
        report = json.loads((root / "verl-direct-report.json").read_text())
        metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        cases[algorithm] = {
            "profile": result["profile"],
            "field_rules_sha256": rules_digest(),
            "composition": expected.provenance,
            "composition_status": result["composition_status"],
            "semantic_status": result["semantic_status"],
            "execution_status": result["execution_status"],
            "resolved_bytes_preserved": True,
            "manual_native_yaml_edits": False,
            "runtime_bindings": report["runtime_bindings"],
            "hardware_plan": report["hardware_plan"],
            "ir_sha256": report["ir_sha256"],
            "sampling_semantics": report["sampling_semantics"],
            "sampling": [row for row in metrics if row["phase"] == "rl_sampling"],
            "minibatches": [
                {key: row[key] for key in ("phase", "ppo_epoch", "minibatch_trajectory_ids")}
                for row in metrics
                if "minibatch_trajectory_ids" in row
            ],
            "updates": summary["updates"],
            "rollouts": summary["rollouts"],
            "rewards": summary["rewards"],
            "memory": summary["memory"],
            "models": summary["models"],
            "exact_tensor_resume": True,
            "exact_semantic_journal_resume": True,
            "journal_normalization": "rewards.jsonl: omit top-level duration_ms only; other journals byte-exact",
            "tensor_hashes": after,
            "journal_hashes": journals,
            "handoff_verdict": doctor["verdict"],
            "launchable": False,
            "distributed_execution_tested": False,
            "workflow_seconds": time.perf_counter() - started,
        }
    return {
        "schema_version": 1,
        "kind": "installed_native_hydra_workflows",
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
        "scientific_scope": "bounded length-reward systems qualification; no task-quality claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--rehearsal", action="store_true")
    args = parser.parse_args()
    args.work = args.work.resolve()
    write_json_atomic(args.output, run(args))


if __name__ == "__main__":
    main()
