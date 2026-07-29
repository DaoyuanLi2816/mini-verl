"""End-to-end propagation checks for the explicit no-network policy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from miniverl.cli import app
from miniverl.evaluation.benchmark import run_benchmark
from miniverl.evaluation.evaluator import evaluate_run
from miniverl.evaluation.schema import BenchmarkConfig


def test_every_network_capable_cli_uses_the_offline_term() -> None:
    runner = CliRunner()
    for command in ("train", "benchmark", "eval", "export-adapter"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--offline" in result.stdout, command

    export_help = runner.invoke(app, ["export-adapter", "--help"])
    assert "--local-files-only" in export_help.stdout


def test_train_cli_passes_offline_policy_to_trainer(tmp_path: Path, monkeypatch) -> None:
    from miniverl.trainer import OPDTrainer

    seen: list[bool] = []
    run_root = tmp_path / "train-result"
    result = SimpleNamespace(
        to_dict=lambda: {
            "run_id": "offline",
            "run_dir": str(run_root),
            "mode": "opd",
            "cycles_completed": 0,
            "global_step": 0,
            "policy_version": 0,
            "duration_seconds": 0.0,
            "final_metrics": {},
            "eval": None,
            "baseline_eval": None,
        },
        baseline_eval=None,
        eval=None,
        global_step=0,
        duration_seconds=0.0,
    )

    class FakeTrainer:
        paths = SimpleNamespace(root=run_root)

        def train(self):
            return result

        def close(self) -> None:
            return None

    def from_config(*args, **kwargs):
        seen.append(kwargs["local_files_only"])
        return FakeTrainer()

    monkeypatch.setattr(OPDTrainer, "from_config", staticmethod(from_config))
    recipe = Path(__file__).resolve().parents[2] / "recipes" / "toy_cpu.yaml"
    cli_result = CliRunner().invoke(
        app,
        ["train", str(recipe), "--offline", "--no-report", "--json"],
    )

    assert cli_result.exit_code == 0, cli_result.output
    assert seen == [True]


def test_export_adapter_offline_and_legacy_alias_are_equivalent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from miniverl.models import adapter_io

    policies: list[bool] = []

    def fake_export(*args, **kwargs):
        policies.append(kwargs["local_files_only"])
        return {}, tmp_path / f"adapter-{len(policies)}"

    monkeypatch.setattr(adapter_io, "export_adapter", fake_export)
    runner = CliRunner()
    common = ["--run", str(tmp_path / "run"), "--out"]
    offline = runner.invoke(app, ["export-adapter", *common, str(tmp_path / "one"), "--offline"])
    alias = runner.invoke(
        app,
        ["export-adapter", *common, str(tmp_path / "two"), "--local-files-only"],
    )

    assert offline.exit_code == 0, offline.output
    assert alias.exit_code == 0, alias.output
    assert policies == [True, True]


def test_evaluate_run_passes_offline_policy_to_reconstructed_trainer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from miniverl.config import RunConfig
    from miniverl.trainer import OPDTrainer

    run_dir = tmp_path / "runs" / "finished"
    run_dir.mkdir(parents=True)
    config = RunConfig.from_mapping(
        {
            "run": {"name": "offline-eval", "output_dir": str(run_dir.parent)},
            "models": {
                "backend": "toy",
                "student": {"model_id": "toy-student"},
                "teacher": {"model_id": "toy-teacher"},
            },
            "environment": {
                "name": "calculator",
                "train_tasks": 1,
                "eval_tasks": 1,
                "test_tasks": 1,
            },
            "train": {"cycles": 0, "rollouts_per_cycle": 1},
            "eval": {"enabled": False},
            "report": {"enabled": False},
        }
    )
    (run_dir / "config.resolved.yaml").write_text(config.to_yaml(), encoding="utf-8")
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    seen: list[bool] = []

    class FakeTrainer:
        student = SimpleNamespace(device="cpu")
        optimizer = None

        def evaluate(self, **kwargs):
            return {"split": kwargs.get("split") or "test"}

        def close(self) -> None:
            return None

    def from_config(*args, **kwargs):
        seen.append(kwargs["local_files_only"])
        return FakeTrainer()

    monkeypatch.setattr(OPDTrainer, "from_config", staticmethod(from_config))
    payload = evaluate_run(run_dir, local_files_only=True)

    assert seen == [True]
    assert payload["split"] == "test"


@pytest.mark.torch
def test_benchmark_passes_offline_policy_to_cold_start_and_every_arm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from miniverl.trainer import OPDTrainer

    spec = BenchmarkConfig.model_validate(
        {
            "schema_version": 2,
            "name": "offline-propagation",
            "base": {
                "models": {
                    "backend": "toy",
                    "student": {"model_id": "toy-student"},
                    "teacher": {"model_id": "toy-teacher"},
                },
                "environment": {
                    "name": "calculator",
                    "difficulty": "easy",
                    "train_tasks": 1,
                    "eval_tasks": 1,
                    "test_tasks": 1,
                },
                "train": {
                    "cycles": 0,
                    "rollouts_per_cycle": 1,
                    "gradient_accumulation_steps": 1,
                },
                "eval": {"enabled": False, "tasks": 1},
                "report": {"enabled": False},
            },
            "cold_start_cycles": 1,
            "allowed_differences": ["run.mode", "train.cycles"],
            "seeds": [7],
            "arms": [
                {
                    "name": "cold-start-only",
                    "overrides": {"run": {"mode": "sft"}, "train": {"cycles": 0}},
                },
                {
                    "name": "continued-sft",
                    "overrides": {"run": {"mode": "sft"}, "train": {"cycles": 0}},
                },
            ],
        }
    )
    seen: list[bool] = []
    original = OPDTrainer.from_config.__func__

    def from_config(cls, *args, **kwargs):
        seen.append(kwargs["local_files_only"])
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(OPDTrainer, "from_config", classmethod(from_config))
    result = run_benchmark(
        spec,
        output_dir=tmp_path / "benchmark",
        local_files_only=True,
    )

    assert len(result.arms) == 2
    assert seen == [True, True, True]
