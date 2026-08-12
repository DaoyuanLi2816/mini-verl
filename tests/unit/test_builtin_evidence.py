"""Installed evidence is self-contained and byte-bound."""

from __future__ import annotations

import hashlib


def test_builtin_study_uses_python_310_traversable_joinpath_contract(monkeypatch) -> None:
    import miniverl.evidence as evidence

    class SingleSegmentTraversable:
        def __init__(self) -> None:
            self.parts: list[str] = []

        def joinpath(self, child: str) -> SingleSegmentTraversable:
            self.parts.append(child)
            return self

        def __str__(self) -> str:
            return "missing-packaged-evidence"

    traversable = SingleSegmentTraversable()
    monkeypatch.setattr(evidence, "files", lambda package: traversable)

    study = evidence.get_builtin_study("alignment-external-v1")

    assert traversable.parts == ["data", "alignment-external-v1"]
    assert study.result_path.is_file()


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
