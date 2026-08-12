"""Installed evidence is self-contained and byte-bound."""

from __future__ import annotations

import hashlib


def test_builtin_external_study_resolves_without_a_repository_checkout() -> None:
    from miniverl.evidence import get_builtin_study

    study = get_builtin_study("alignment-external-v1")

    assert study.result_path.is_file()
    assert study.schema_path.is_file()
    assert study.preregistration_path.is_file()
    assert study.task_evidence_path.is_file()
    assert hashlib.sha256(study.result_path.read_bytes()).hexdigest() == study.result_sha256


def test_builtin_external_study_validates_every_packaged_binding() -> None:
    from miniverl.evidence import validate_builtin_study

    report = validate_builtin_study("alignment-external-v1")

    assert report["valid"] is True
    assert report["task_rows"] == 512
    assert report["problems"] == []
