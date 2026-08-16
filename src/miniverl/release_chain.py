"""Torch-free validation for one exact release candidate and its qualification."""

from __future__ import annotations

import json
from pathlib import Path

from miniverl.qualification import GPUQualification, validate_qualification_file
from miniverl.release_candidate import (
    CandidateManifest,
    load_candidate_manifest,
    sha256_file,
    validate_candidate_directory,
)

__all__ = ["validate_release_chain"]


def validate_release_chain(
    candidate_dir: str | Path,
    candidate_manifest: str | Path,
    qualification: str | Path,
    *,
    expected_commit: str,
    expected_known_good_sha256: str,
    required_gpu_name: str,
) -> list[str]:
    """Return all byte-identity and provenance errors for a release evidence chain."""
    directory = Path(candidate_dir)
    manifest_path = Path(candidate_manifest)
    qualification_path = Path(qualification)
    problems = validate_candidate_directory(
        directory,
        manifest_path=manifest_path,
        expected_commit=expected_commit,
    )
    if problems:
        return problems
    try:
        candidate = load_candidate_manifest(manifest_path)
    except (OSError, UnicodeError, ValueError) as exc:
        return [f"manifest: {exc}"]
    problems.extend(
        validate_qualification_file(
            qualification_path,
            expected_commit=expected_commit,
            expected_wheel_sha256=candidate.wheel.sha256,
            expected_candidate_manifest_sha256=sha256_file(manifest_path),
            expected_known_good_sha256=expected_known_good_sha256,
            required_gpu_name=required_gpu_name,
        )
    )
    if problems:
        return problems
    try:
        payload = json.loads(qualification_path.read_text(encoding="utf-8"))
        evidence = GPUQualification.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return [f"qualification: {exc}"]
    return _cross_record_bindings(candidate, evidence)


def _cross_record_bindings(candidate: CandidateManifest, evidence: GPUQualification) -> list[str]:
    problems: list[str] = []
    bindings = (
        (candidate.source_commit, evidence.source_commit, "source commit"),
        (candidate.miniverl_version, evidence.miniverl_version, "miniVERL version"),
        (candidate.artifact_name, evidence.candidate.artifact_name, "artifact name"),
        (
            candidate.workflow.repository,
            evidence.candidate.workflow_repository,
            "workflow repository",
        ),
        (
            candidate.workflow.workflow_path,
            evidence.candidate.workflow_path,
            "workflow path",
        ),
        (
            candidate.workflow.run_id,
            evidence.candidate.workflow_run_id,
            "workflow run id",
        ),
        (
            candidate.workflow.run_attempt,
            evidence.candidate.workflow_run_attempt,
            "workflow run attempt",
        ),
    )
    for candidate_value, evidence_value, label in bindings:
        if candidate_value != evidence_value:
            problems.append(
                f"binding: candidate {label} {candidate_value!r} does not match "
                f"qualification {evidence_value!r}"
            )
    if candidate.wheel.filename != evidence.wheel.filename:
        problems.append("binding: candidate wheel filename does not match qualification")
    return problems
