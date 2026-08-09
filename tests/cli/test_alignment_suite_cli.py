"""The alignment-suite commands, driven offline through the Typer runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from miniverl.alignment_external.records import TaskRecord
from miniverl.cli import app

runner = CliRunner()


def _profile(tmp_path: Path) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": "cli-test",
                "selection_seed": 4,
                "endpoints": [
                    {"id": "ifeval", "tasks": 4},
                    {"id": "xstest", "tasks": 4},
                    {"id": "jbb_behaviors", "tasks": 2},
                    {"id": "rewardbench", "tasks": 2, "counts_toward_generation": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_resolver(monkeypatch: Any, pool: int = 40) -> None:
    """Replace the Hub resolver so the CLI runs with no network.

    ``pool`` is the upstream task count. It has to exceed what a profile asks
    for, otherwise the selection is silently capped by the pool size and a test
    about budget ceilings never reaches the ceiling.
    """
    import miniverl.cli as cli_module

    def resolver() -> Any:
        def resolve(endpoint: dict[str, Any]) -> tuple[list[str], list[str] | None]:
            ids = [f"{endpoint['id']}-{index:04d}" for index in range(pool)]
            strata = [f"s{index % 4}" for index in range(pool)]
            return ids, strata

        return resolve

    monkeypatch.setattr(cli_module, "_hub_task_resolver", resolver)


def _rows(manifest: dict[str, Any], *, drop: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in manifest["endpoints"]:
        for task_id in entry["task_ids"][: len(entry["task_ids"]) - drop]:
            rows.append(
                TaskRecord(
                    score=1.0,
                    endpoint_id=entry["id"],
                    category=entry["category"],
                    dataset=entry["dataset"] or "internal",
                    dataset_revision=entry["revision"] or "0" * 40,
                    split=entry["split"] or "n/a",
                    task_id=task_id,
                    subset=None,
                    checkpoint_id="sft",
                    checkpoint_digest="a" * 64,
                    method="starting-sft-checkpoint",
                    seed=None,
                    generation_config_digest="b" * 64,
                    output_digest="c" * 64,
                    output_tokens=8,
                ).to_json_row()
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ prepare


def test_prepare_writes_a_manifest(tmp_path: Path, monkeypatch: Any) -> None:
    _fake_resolver(monkeypatch)

    result = runner.invoke(
        app,
        [
            "alignment-suite",
            "prepare",
            "--profile",
            str(_profile(tmp_path)),
            "--out",
            str(tmp_path / "suite"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["generation_tasks_per_model"] == 10
    assert (tmp_path / "suite" / "suite-manifest.json").is_file()


def test_dry_run_writes_nothing(tmp_path: Path, monkeypatch: Any) -> None:
    _fake_resolver(monkeypatch)

    result = runner.invoke(
        app,
        [
            "alignment-suite",
            "prepare",
            "--profile",
            str(_profile(tmp_path)),
            "--out",
            str(tmp_path / "suite"),
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["written"] is None
    assert not (tmp_path / "suite" / "suite-manifest.json").exists()


def test_a_profile_over_the_ceiling_fails_the_command(tmp_path: Path, monkeypatch: Any) -> None:
    _fake_resolver(monkeypatch, pool=800)
    path = tmp_path / "big.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": "too-big",
                "selection_seed": 1,
                "endpoints": [{"id": "ifeval", "tasks": 600}],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["alignment-suite", "prepare", "--profile", str(path), "--out", str(tmp_path / "s")],
    )

    assert result.exit_code != 0


# --------------------------------------------------------- validate / report


def _prepared(tmp_path: Path, monkeypatch: Any) -> dict[str, Any]:
    _fake_resolver(monkeypatch)
    runner.invoke(
        app,
        [
            "alignment-suite",
            "prepare",
            "--profile",
            str(_profile(tmp_path)),
            "--out",
            str(tmp_path / "suite"),
            "--json",
        ],
    )
    return json.loads((tmp_path / "suite" / "suite-manifest.json").read_text(encoding="utf-8"))


def test_validate_accepts_a_matching_result_set(tmp_path: Path, monkeypatch: Any) -> None:
    manifest = _prepared(tmp_path, monkeypatch)
    results = _write_jsonl(tmp_path / "rows.jsonl", _rows(manifest))

    result = runner.invoke(
        app,
        [
            "alignment-suite",
            "validate",
            str(results),
            "--manifest",
            str(tmp_path / "suite" / "suite-manifest.json"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["valid"] is True


def test_validate_rejects_an_incomplete_result_set(tmp_path: Path, monkeypatch: Any) -> None:
    manifest = _prepared(tmp_path, monkeypatch)
    results = _write_jsonl(tmp_path / "rows.jsonl", _rows(manifest, drop=1))

    result = runner.invoke(
        app,
        [
            "alignment-suite",
            "validate",
            str(results),
            "--manifest",
            str(tmp_path / "suite" / "suite-manifest.json"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert any("no result row" in problem for problem in payload["problems"])


def test_report_emits_per_endpoint_means(tmp_path: Path, monkeypatch: Any) -> None:
    manifest = _prepared(tmp_path, monkeypatch)
    results = _write_jsonl(tmp_path / "rows.jsonl", _rows(manifest))

    result = runner.invoke(app, ["alignment-suite", "report", str(results), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert set(payload["endpoints"]) == {"ifeval", "xstest", "jbb_behaviors", "rewardbench"}
    assert payload["endpoints"]["ifeval"]["mean_score"] == 1.0
    # No combined score is ever emitted.
    assert "alignment_score" not in payload
