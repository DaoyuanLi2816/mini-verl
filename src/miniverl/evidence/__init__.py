"""Self-contained, read-only evidence shipped with the core wheel."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Final

from miniverl.alignment_external.result import load_alignment_external_result
from miniverl.errors import ConfigError

__all__ = ["BuiltinStudy", "get_builtin_study", "show_builtin_study", "validate_builtin_study"]


@dataclass(frozen=True, slots=True)
class BuiltinStudy:
    """Paths and immutable digests for one packaged evidence bundle."""

    study_id: str
    result_path: Path
    schema_path: Path
    preregistration_path: Path
    task_evidence_path: Path
    result_sha256: str
    schema_sha256: str
    preregistration_sha256: str
    task_evidence_sha256: str


_ALIGNMENT_EXTERNAL_V1: Final = {
    "result_sha256": "085cbe1f8035a0904482332d60f9f46ae3039d2e5ac4725e2ecafb7b42d0eda8",
    "schema_sha256": "d41dc15bbd0d3b6852e858142f11c5b89adf0ce591676abc7e42082665c82044",
    "preregistration_sha256": "b87596f05d6c411ac5a2f982729200287d5bc917b1708b1fc1640bf53e2ca379",
    "task_evidence_sha256": "694d68cd997bc4b2aa7dd88ebf6572616c9a140fb0df4a672c301095a4f16c7c",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_selection_rows(rows: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    required = {
        "lineage_id",
        "candidate_id",
        "suite_task_id",
        "trajectory_digest",
        "schema_version",
        "solved",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            problems.append(f"row {index}: missing {', '.join(missing)}")
            continue
        key = (str(row["lineage_id"]), str(row["candidate_id"]), str(row["suite_task_id"]))
        if key in seen:
            problems.append(f"row {index}: duplicate {key}")
        seen.add(key)
        if row["schema_version"] != 1:
            problems.append(f"row {index}: schema_version is not 1")
        if row["solved"] is not False:
            problems.append(f"row {index}: frozen selection evidence unexpectedly solved a task")
        digest = row["trajectory_digest"]
        if not isinstance(digest, str) or len(digest) != 64:
            problems.append(f"row {index}: invalid trajectory digest")
    if len(rows) != 512:
        problems.append(f"task evidence contains {len(rows)} rows, expected 512")
    return problems


def get_builtin_study(study_id: str) -> BuiltinStudy:
    """Resolve a named study from installed package data, never from the checkout."""
    if study_id != "alignment-external-v1":
        raise ConfigError(
            f"unknown built-in study {study_id!r}",
            hint="available built-in studies: alignment-external-v1",
        )
    root = Path(str(files("miniverl.evidence").joinpath("data", study_id)))
    if root.is_dir():
        result_path = root / "result.json"
        schema_path = root / "result.schema.json"
        preregistration_path = root / "preregistration.yaml"
        task_evidence_path = root / "task-evidence.jsonl"
    else:
        # A source checkout has not passed through Hatch's force-include mapping.
        # Installed wheels always take the branch above.
        repository = Path(__file__).resolve().parents[3]
        result_path = repository / "benchmarks/results/alignment-external-v1.json"
        schema_path = repository / "benchmarks/schema/alignment-external-result.schema.json"
        preregistration_path = repository / "benchmarks/preregistration/alignment-external-v1.yaml"
        task_evidence_path = (
            repository / "benchmarks/evidence/alignment-external-v1/jsonnav-selection-records.jsonl"
        )
    return BuiltinStudy(
        study_id=study_id,
        result_path=result_path,
        schema_path=schema_path,
        preregistration_path=preregistration_path,
        task_evidence_path=task_evidence_path,
        **_ALIGNMENT_EXTERNAL_V1,
    )


def show_builtin_study(study_id: str) -> dict[str, Any]:
    """Return the typed result as a JSON-friendly document."""
    study = get_builtin_study(study_id)
    result = load_alignment_external_result(study.result_path)
    return {
        "study_id": study.study_id,
        "result_sha256": study.result_sha256,
        "result": result.model_dump(mode="json"),
    }


def validate_builtin_study(study_id: str) -> dict[str, Any]:
    """Validate every packaged byte binding and the task-level row contract."""
    study = get_builtin_study(study_id)
    problems: list[str] = []
    paths = {
        "result": (study.result_path, study.result_sha256),
        "schema": (study.schema_path, study.schema_sha256),
        "preregistration": (study.preregistration_path, study.preregistration_sha256),
        "task_evidence": (study.task_evidence_path, study.task_evidence_sha256),
    }
    observed: dict[str, str] = {}
    for name, (path, expected) in paths.items():
        if not path.is_file():
            problems.append(f"missing packaged {name}: {path.name}")
            continue
        actual = _sha256(path)
        observed[name] = actual
        if actual != expected:
            problems.append(f"{name} SHA-256 is {actual}, expected {expected}")

    task_rows: list[dict[str, Any]] = []
    if not problems:
        result = load_alignment_external_result(study.result_path)
        if result.preregistration.sha256 != study.preregistration_sha256:
            problems.append("result preregistration binding does not match the packaged artifact")
        if result.checkpoint_selection.task_evidence.sha256 != study.task_evidence_sha256:
            problems.append("result task-evidence binding does not match the packaged artifact")
        try:
            json.loads(study.schema_path.read_text(encoding="utf-8"))
            task_rows = [
                json.loads(line)
                for line in study.task_evidence_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (json.JSONDecodeError, UnicodeError) as exc:
            problems.append(f"packaged evidence is not valid UTF-8 JSON: {exc}")
        else:
            problems.extend(_validate_selection_rows(task_rows))

    return {
        "study_id": study.study_id,
        "valid": not problems,
        "task_rows": len(task_rows),
        "sha256": observed,
        "problems": problems,
    }
