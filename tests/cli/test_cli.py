"""The command line surface of ``miniverl``.

This file protects the contract a user touches first: every command must be
reachable and self-documenting, ``--json`` must emit a *parsable* document with
the documented keys, the read-only commands must work without torch, the
inspection commands must refuse a path that is not what they claim to read, and
``train --dry-run`` must not create a single file.  The end-to-end block runs
the embedded toy demo and then walks the whole ``inspect`` / ``cache`` /
``report`` / ``export-benchmark`` chain over its output, so a break anywhere in
the artifact contract fails here instead of in a user's first session.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import Result
from typer.main import get_command
from typer.testing import CliRunner

from miniverl import __version__
from miniverl.cli import app
from tests.conftest import HAS_CUDA, requires_torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TOY_RECIPE = REPO_ROOT / "recipes" / "toy_cpu.yaml"

#: Every documented command, spelled exactly as it is typed.
EXPECTED_COMMANDS = {
    "doctor",
    "validate",
    "demo",
    "train",
    "eval",
    "benchmark",
    "inspect",
    "report",
    "export-benchmark",
    "schema",
    "cache stats",
    "cache validate",
}

#: What a finished run must contain, relative to the run directory.
DEMO_ARTIFACTS = (
    "config.original.yaml",
    "config.resolved.yaml",
    "manifest.json",
    "environment.json",
    "metrics.jsonl",
    "events.jsonl",
    "trajectories.jsonl",
    "teacher-cache/index.json",
    "eval.json",
    "report.html",
)

_STATUSES = {"ok", "warn", "missing", "fail"}


def _invoke(*args: str) -> Result:
    """Run the CLI in-process with a fresh runner."""
    return CliRunner().invoke(app, list(args))


def _payload(result: Result) -> Any:
    """Assert the command succeeded and parse its ``--json`` document.

    ``stdout`` only: log records go to stderr, so mixing the two would make the
    JSON unparsable and hide a real regression behind a decoding error.
    """
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _collapse(text: str) -> str:
    """Undo rich's console wrapping so a wrapped message can be searched."""
    return re.sub(r"\s+", " ", text)


def _walk_commands(*, groups: bool) -> list[tuple[str, ...]]:
    """Enumerate the app's commands as argument tuples.

    ``groups=True`` also yields the sub-command groups themselves (``cache``),
    which have their own help page.  The root is never yielded.
    """
    paths: list[tuple[str, ...]] = []

    def walk(prefix: tuple[str, ...], command: Any) -> None:
        children = getattr(command, "commands", None)
        if not children:
            paths.append(prefix)
            return
        if groups and prefix:
            paths.append(prefix)
        for name, child in sorted(children.items()):
            walk((*prefix, name), child)

    walk((), get_command(app))
    return paths


def _top_level_names() -> list[str]:
    """Names listed under ``Commands:`` in the root help."""
    return sorted(get_command(app).commands)


def _write_recipe(path: Path, mutate: dict[str, Any]) -> Path:
    """Copy the toy recipe with ``mutate`` applied to its ``train`` block."""
    data = yaml.safe_load(TOY_RECIPE.read_text(encoding="utf-8"))
    data["train"].update(mutate)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


COMMAND_PATHS = _walk_commands(groups=False)
HELP_PATHS = _walk_commands(groups=True)


# ------------------------------------------------------------------ discovery


def test_version_flag_prints_name_and_version() -> None:
    result = _invoke("--version")
    assert result.exit_code == 0
    assert result.stdout.strip() == f"miniverl {__version__}"


def test_short_version_flag_matches_long_flag() -> None:
    assert _invoke("-V").stdout == _invoke("--version").stdout


def test_root_help_lists_every_command() -> None:
    result = _invoke("--help")
    assert result.exit_code == 0
    for name in _top_level_names():
        assert name in result.stdout, f"{name} missing from --help"


def test_command_set_matches_the_documented_set() -> None:
    assert {" ".join(path) for path in COMMAND_PATHS} == EXPECTED_COMMANDS


@pytest.mark.parametrize("path", HELP_PATHS, ids=[" ".join(path) for path in HELP_PATHS])
def test_every_command_has_help(path: tuple[str, ...]) -> None:
    result = _invoke(*path, "--help")
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.stdout
    assert result.stdout.strip()


def test_no_arguments_shows_help_and_exits_nonzero() -> None:
    result = _invoke()
    assert result.exit_code != 0
    assert "Usage:" in result.output


# --------------------------------------------------------------------- doctor


def test_doctor_json_reports_checks_capabilities_and_verdict(tmp_path: Path) -> None:
    target = tmp_path / "probe-runs"
    payload = _payload(_invoke("doctor", "--json", "--output", str(target)))

    assert set(payload) >= {"miniverl_version", "checks", "capabilities", "verdict"}
    assert payload["miniverl_version"] == __version__

    assert payload["checks"], "doctor reported no checks at all"
    for check in payload["checks"]:
        assert set(check) == {"name", "status", "detail", "hint"}
        assert check["status"] in _STATUSES

    names = [check["name"] for check in payload["checks"]]
    assert any(name.startswith("dependency:") for name in names)
    assert "output directory" in names

    verdict = payload["verdict"]
    assert set(verdict) == {"core_commands", "cpu_training", "gpu_training", "qlora_4bit"}
    assert all(isinstance(value, bool) for value in verdict.values())
    # The test suite itself is running, so the base dependencies are importable.
    assert verdict["core_commands"] is True
    assert verdict["gpu_training"] is HAS_CUDA

    capabilities = payload["capabilities"]
    assert capabilities["environments"], "no environments registered"
    assert capabilities["cuda_available"] is HAS_CUDA


def test_doctor_probes_the_requested_directory_and_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "runs"
    payload = _payload(_invoke("doctor", "--json", "--output", str(target)))
    status = next(c for c in payload["checks"] if c["name"] == "output directory")
    assert status["status"] == "ok"
    assert target.is_dir()
    # The writability probe file must not survive the check.
    assert list(target.iterdir()) == []


def test_doctor_renders_a_table_without_json(tmp_path: Path) -> None:
    result = _invoke("doctor", "--output", str(tmp_path / "probe-runs"))
    assert result.exit_code == 0
    collapsed = _collapse(result.stdout)
    assert "core commands" in collapsed
    assert "4-bit QLoRA" in collapsed


# ------------------------------------------------------------------- validate


def test_validate_toy_recipe_json() -> None:
    payload = _payload(_invoke("validate", str(TOY_RECIPE), "--json"))
    assert payload["valid"] is True
    assert payload["mode"] == "opd"
    assert payload["is_on_policy"] is True
    assert payload["backend"] == "toy"
    assert payload["environment"] == "calculator"
    assert payload["eval_tasks"] > 0
    # One optimizer step per rollout batch is what makes the toy recipe strictly
    # on-policy; the planned step count must follow from it.
    assert payload["optimizer_steps_per_cycle"] == 1
    assert payload["planned_optimizer_steps"] == payload["optimizer_steps_per_cycle"] * (
        payload["cycles"] + payload["sft_warmup_cycles"]
    )
    assert payload["warnings"] == []


def test_validate_renders_a_table_without_json() -> None:
    result = _invoke("validate", str(TOY_RECIPE))
    assert result.exit_code == 0
    collapsed = _collapse(result.stdout)
    assert "valid" in collapsed
    assert "toy-cpu-opd" in collapsed


def test_validate_missing_recipe_exits_nonzero(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    result = _invoke("validate", str(missing))
    assert result.exit_code != 0
    assert "not found" in _collapse(result.output)


def test_validate_out_of_range_field_exits_nonzero_and_names_it(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path / "bad-range.yaml", {"cycles": -1})
    result = _invoke("validate", str(recipe))
    assert result.exit_code != 0
    assert "train.cycles" in _collapse(result.output)


def test_validate_unknown_field_exits_nonzero_and_names_it(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path / "bad-field.yaml", {"nonexistent_knob": 3})
    result = _invoke("validate", str(recipe))
    assert result.exit_code != 0
    assert "nonexistent_knob" in _collapse(result.output)


def test_validate_json_reports_the_failing_location(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path / "bad-range.yaml", {"cycles": -1})
    result = _invoke("validate", str(recipe), "--json")
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert ["train", "cycles"] in [error["loc"] for error in payload["errors"]]


# ---------------------------------------------------------------------- train


def test_train_dry_run_plans_without_writing_anything(tmp_path: Path) -> None:
    target = tmp_path / "run-output"
    default_output = Path.cwd() / "runs"
    default_existed = default_output.exists()

    payload = _payload(
        _invoke("train", str(TOY_RECIPE), "--dry-run", "--json", "--output", str(target))
    )

    assert payload["dry_run"] is True
    assert payload["downloads_required"] is False
    assert payload["mode"] == "opd"
    assert payload["backend"] == "toy"
    assert payload["output_dir"] == str(target)
    assert payload["planned_optimizer_steps"] > 0
    assert payload["planned_rollouts"] > 0
    assert payload["resolved_config"]["models"]["backend"] == "toy"

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
    assert default_output.exists() is default_existed


def test_train_dry_run_rejects_a_missing_recipe(tmp_path: Path) -> None:
    result = _invoke("train", str(tmp_path / "absent.yaml"), "--dry-run")
    assert result.exit_code != 0
    assert "not found" in _collapse(result.output)


# --------------------------------------------------------------------- schema


def test_schema_emits_a_json_schema() -> None:
    payload = _payload(_invoke("schema"))
    assert payload["$schema"].startswith("https://json-schema.org/")
    assert payload["$id"].endswith("benchmark-result.schema.json")
    assert {"miniverl_version", "name", "created_at", "arms"} <= set(payload["properties"])
    assert "miniverl_version" in payload["required"]


def test_schema_out_writes_the_same_document(tmp_path: Path) -> None:
    target = tmp_path / "schema.json"
    result = _invoke("schema", "--out", str(target))
    assert result.exit_code == 0
    assert json.loads(target.read_text(encoding="utf-8")) == _payload(_invoke("schema"))


# ------------------------------------------------------------- failure paths


def test_inspect_missing_file_exits_nonzero(tmp_path: Path) -> None:
    result = _invoke("inspect", str(tmp_path / "trajectories.jsonl"))
    assert result.exit_code != 0
    assert "not found" in _collapse(result.output)


def test_cache_stats_without_index_exits_nonzero(tmp_path: Path) -> None:
    empty = tmp_path / "teacher-cache"
    empty.mkdir()
    result = _invoke("cache", "stats", str(empty))
    assert result.exit_code != 0
    assert "index.json" in _collapse(result.output)


def test_cache_validate_without_index_exits_nonzero(tmp_path: Path) -> None:
    empty = tmp_path / "teacher-cache"
    empty.mkdir()
    result = _invoke("cache", "validate", str(empty), "--json")
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert payload["problems"]


def test_report_on_a_non_run_directory_exits_nonzero(tmp_path: Path) -> None:
    stray = tmp_path / "not-a-run"
    stray.mkdir()
    (stray / "notes.txt").write_text("hello", encoding="utf-8")
    result = _invoke("report", str(stray))
    assert result.exit_code != 0
    assert "manifest.json" in _collapse(result.output)


def test_report_on_a_missing_directory_exits_nonzero(tmp_path: Path) -> None:
    result = _invoke("report", str(tmp_path / "absent"))
    assert result.exit_code != 0
    assert "not found" in _collapse(result.output)


# ------------------------------------------------- behaviour on a bare install

#: Runs ``miniverl demo`` in a child interpreter where ``import torch`` fails the
#: way it fails on ``pip install miniverl`` without the ``[train]`` extra.
_NO_TORCH_PROBE = """
import sys
from importlib.abc import MetaPathFinder


class _Blocker(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name=fullname)
        return None


sys.meta_path.insert(0, _Blocker())

from typer.testing import CliRunner

from miniverl.cli import app

result = CliRunner().invoke(app, ["demo", "--output", sys.argv[1], "--fast"])
print("EXIT", result.exit_code)
print("EXCEPTION", type(result.exception).__name__ if result.exception else "none")
print("OUTPUT", " | ".join(result.output.splitlines()))
"""


def test_demo_without_torch_names_the_missing_extra(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_PROBE, str(tmp_path / "run")],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=REPO_ROOT,
    )
    report = completed.stdout
    assert "EXIT 1" in report, report
    # typer.Exit surfaces as SystemExit, which is the intended clean exit path.
    assert "EXCEPTION ModuleNotFoundError" not in report, (
        "the missing extra escaped as a raw ModuleNotFoundError"
    )
    assert "EXCEPTION ImportError" not in report, report
    assert "miniverl[train]" in report, "the error must name the extra to install"


# ------------------------------------------------------- end-to-end toy flow


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run ``miniverl demo --fast`` once and hand back its run directory."""
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("demo-flow") / "run"
    result = _invoke("demo", "--output", str(root), "--fast", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["run_dir"] == str(root)
    assert payload["mode"] == "opd"
    assert payload["global_step"] > 0
    return root


@requires_torch
@pytest.mark.torch
def test_demo_writes_every_documented_artifact(demo_run: Path) -> None:
    for relative in DEMO_ARTIFACTS:
        assert (demo_run / relative).is_file(), f"{relative} missing from {demo_run}"
    checkpoints = demo_run / "checkpoints"
    assert checkpoints.is_dir()
    assert any(checkpoints.iterdir()), "no checkpoint was written"


@requires_torch
@pytest.mark.torch
def test_inspect_json_accounts_for_every_token(demo_run: Path) -> None:
    payload = _payload(_invoke("inspect", str(demo_run / "trajectories.jsonl"), "--json"))

    assert payload["trajectories"] > 0
    fraction = payload["model_token_fraction"]
    assert 0.0 < fraction < 1.0, "a tool-using rollout is neither all context nor all model"

    provenance = payload["provenance_check"]
    excluded = provenance["context_span_types_excluded_from_loss"]
    trainable = set(provenance["trainable_span_types"])
    assert "tool_result" in excluded, "tool output must never be trainable"
    assert trainable.isdisjoint(excluded)

    by_span = payload["tokens_by_span_type"]
    assert sum(by_span.values()) == payload["tokens"]
    assert payload["tokens"] == payload["model_tokens"] + payload["context_tokens"]
    assert payload["model_tokens"] == sum(
        count for name, count in by_span.items() if name in trainable
    )
    assert payload["model_tokens"] / payload["tokens"] == pytest.approx(fraction)


@requires_torch
@pytest.mark.torch
def test_inspect_unknown_trajectory_id_exits_nonzero(demo_run: Path) -> None:
    result = _invoke("inspect", str(demo_run / "trajectories.jsonl"), "--trajectory", "no-such-id")
    assert result.exit_code != 0
    assert "no trajectories matched" in _collapse(result.output)


@requires_torch
@pytest.mark.torch
def test_inspect_renders_spans_without_json(demo_run: Path) -> None:
    trajectories = demo_run / "trajectories.jsonl"
    first = json.loads(trajectories.read_text(encoding="utf-8").splitlines()[0])
    result = _invoke(
        "inspect", str(trajectories), "--trajectory", first["trajectory_id"], "--spans"
    )
    assert result.exit_code == 0, result.output
    assert "span" in _collapse(result.stdout)


@requires_torch
@pytest.mark.torch
def test_cache_stats_json_shows_real_compression(demo_run: Path) -> None:
    payload = _payload(_invoke("cache", "stats", str(demo_run / "teacher-cache"), "--json"))
    assert payload["compression_ratio"] > 0.0
    assert payload["problems"] == []
    assert payload["trajectories"] > 0
    assert payload["selected_positions"] > 0
    assert payload["checksums_verified"] is True
    assert payload["actual_bytes"] < payload["theoretical_full_logit_bytes"]
    assert payload["shards"], "a written cache must have at least one shard"


@requires_torch
@pytest.mark.torch
def test_cache_validate_json_accepts_the_written_cache(demo_run: Path) -> None:
    payload = _payload(_invoke("cache", "validate", str(demo_run / "teacher-cache"), "--json"))
    assert payload["valid"] is True
    assert payload["problems"] == []
    assert payload["entries"] > 0
    assert payload["shards"] > 0


@requires_torch
@pytest.mark.torch
def test_report_html_is_self_contained(demo_run: Path) -> None:
    payload = _payload(_invoke("report", str(demo_run), "--json"))
    html_path = Path(payload["html"])
    assert html_path.is_file()
    assert Path(payload["markdown"]).is_file()
    assert payload["json"] is None
    assert payload["summary"]["run_id"] == demo_run.name

    html = html_path.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "<style" in html, "the stylesheet must be inlined, not linked"
    assert not re.search(r"https?://", html), "the report must not reference the network"
    assert 'src="//' not in html, "protocol-relative reference"
    for attribute, value in re.findall(r'\b(src|href)="([^"]*)"', html):
        assert value.startswith(("#", "data:")), f"external {attribute}: {value}"


@requires_torch
@pytest.mark.torch
def test_export_benchmark_matches_the_published_schema(demo_run: Path) -> None:
    from miniverl.evaluation.schema import BenchmarkResult

    payload = _payload(_invoke("export-benchmark", str(demo_run), "--json"))
    written = Path(payload["written"])
    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8")) == payload["result"]

    result = BenchmarkResult.model_validate(payload["result"])
    assert result.miniverl_version == __version__
    assert len(result.arms) == 1
    arm = result.arms[0]
    assert arm.run_dir == demo_run.name, "run_dir must be sanitized to a bare name"
    assert 0.0 <= arm.success_rate <= 1.0
    assert arm.tasks > 0
    assert arm.selected_training_tokens > 0

    serialized = json.dumps(payload["result"])
    assert str(demo_run) not in serialized, "the submission leaks a local path"
    assert demo_run.as_posix() not in serialized
