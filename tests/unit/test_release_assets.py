"""Canonical, deterministic and fail-closed future release assets."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    from miniverl.release_candidate import PINNED_BUILD_TOOLS

    candidate = tmp_path / "candidate"
    qualification = tmp_path / "qualification"
    candidate.mkdir(parents=True)
    qualification.mkdir(parents=True)
    version = "0.11.0.dev0"
    commit = "a" * 40
    wheel = candidate / f"miniverl-{version}-py3-none-any.whl"
    sdist = candidate / f"miniverl-{version}.tar.gz"
    wheel.write_bytes(b"qualified wheel")
    sdist.write_bytes(b"qualified sdist")
    manifest = {
        "schema_version": 1,
        "kind": "miniverl_release_candidate",
        "source_commit": commit,
        "miniverl_version": version,
        "created_at": "2026-08-16T12:00:00Z",
        "artifact_name": "candidate-distributions",
        "workflow": {
            "kind": "github_actions",
            "repository": "DaoyuanLi2816/mini-verl",
            "workflow_path": ".github/workflows/gpu.yml",
            "run_id": 42,
            "run_attempt": 1,
        },
        "build": {"os": "Linux", "python": "3.12.0", "tools": PINNED_BUILD_TOOLS},
        "wheel": {"filename": wheel.name, "bytes": wheel.stat().st_size, "sha256": _sha(wheel)},
        "sdist": {"filename": sdist.name, "bytes": sdist.stat().st_size, "sha256": _sha(sdist)},
    }
    manifest_path = candidate / "candidate-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (candidate / "SHA256SUMS").write_text(
        f"{_sha(wheel)}  {wheel.name}\n{_sha(sdist)}  {sdist.name}\n", encoding="utf-8"
    )

    evidence = {
        "run_summary": ("run-summary.json", b'{"status":"completed"}\n'),
        "inputs_prompts.parquet": ("inputs/prompts.parquet", b"PAR1 prompts"),
        "adapter_adapter_model.safetensors": (
            "adapter/adapter_model.safetensors",
            b"safe tensors",
        ),
        "adapter_adapter_config.json": ("adapter/adapter_config.json", b'{"r":8}\n'),
        "adapter_miniverl_adapter_manifest.json": (
            "adapter/miniverl_adapter_manifest.json",
            b'{"schema_version":1}\n',
        ),
        "candidate-manifest.json": ("candidate-manifest.json", manifest_path.read_bytes()),
        "release_smoke_record": ("full/release-smoke.json", b'{"kind":"smoke"}\n'),
        "full_direct_result": ("full/direct.json", b'{"kind":"direct"}\n'),
        "full_pg_k1_result": ("full/pg_k1.json", b'{"kind":"pg-k1"}\n'),
        "full_smollm2_result": ("full/smollm2.json", b'{"kind":"smollm2"}\n'),
    }
    artifacts = []
    for name, (relative, content) in evidence.items():
        path = qualification / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts.append(
            {"name": name, "path": relative, "sha256": _sha(path), "bytes": len(content)}
        )
    qualification_wheel = qualification / wheel.name
    qualification_wheel.write_bytes(wheel.read_bytes())
    payload = {
        "schema_version": 1,
        "kind": "miniverl_gpu_qualification",
        "level": "full_qualification",
        "status": "passed",
        "measured_at": "2026-08-16T12:01:00Z",
        "source_commit": commit,
        "miniverl_version": version,
        "wheel": {"filename": wheel.name, "sha256": _sha(wheel)},
        "candidate": {
            "manifest_sha256": _sha(manifest_path),
            "artifact_name": "candidate-distributions",
            "workflow_repository": "DaoyuanLi2816/mini-verl",
            "workflow_path": ".github/workflows/gpu.yml",
            "workflow_run_id": 42,
            "workflow_run_attempt": 1,
            "installed_from_candidate": True,
            "import_origin_verified": True,
            "cli_origin_verified": True,
            "import_origin": "qualification_venv_site_packages",
        },
        "profile": {
            "name": "verl-opd-v0.8-single-gpu-v1",
            "identity_digest": "b" * 64,
            "upstream_tag": "v0.8.0",
            "upstream_commit": "c" * 40,
        },
        "environment": {
            "known_good_manifest_sha256": "d" * 64,
            "gpu_name": "NVIDIA GeForce RTX 4080",
            "gpu_count": 1,
            "vram_gib": 15.99,
            "driver": "596.49",
            "cuda_runtime": "13.0",
            "python": "3.10.11",
            "packages": dict.fromkeys(
                (
                    "torch",
                    "transformers",
                    "peft",
                    "accelerate",
                    "bitsandbytes",
                    "numpy",
                    "pyarrow",
                    "safetensors",
                ),
                "1.0",
            ),
        },
        "models": [
            {"role": "actor", "model_id": "org/actor", "revision": "e" * 40},
            {"role": "teacher", "model_id": "org/teacher", "revision": "f" * 40},
        ],
        "execution": {
            "rollout_completed": True,
            "teacher_scoring_completed": True,
            "optimizer_updates": 1,
            "peft_adapter_exported": True,
            "peft_adapter_reload_verified": True,
            "cuda_allocated_before_bytes": 0,
            "cuda_allocated_after_teardown_bytes": 0,
            "cuda_teardown_tolerance_bytes": 2097152,
        },
        "inputs": {
            "config_sha256": "1" * 64,
            "plan_sha256": "2" * 64,
            "parquet_sha256": "3" * 64,
        },
        "artifacts": artifacts,
        "checks": {
            "executed": [
                "wheel_install",
                "doctor",
                "plan",
                "run_dry_run",
                "real_actor_teacher_update",
                "peft_reload",
                "cuda_teardown",
                "direct_gkd_resume_equivalence",
                "sampled_k1_resume_equivalence",
                "smollm2_resume_equivalence",
                "export_materialize_doctor",
            ],
            "skipped": [],
            "not_applicable": ["distributed_verl_execution"],
        },
        "scientific_scope": {
            "runtime_correctness_only": True,
            "task_quality_evaluated": False,
            "alignment_quality_evaluated": False,
            "distributed_execution_tested": False,
            "other_hardware_measured": False,
        },
    }
    qualification_path = qualification / "qualification.json"
    qualification_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = tmp_path / "verification.json"
    verification.write_text(
        json.dumps(
            {
                "candidate_manifest_sha256": _sha(manifest_path),
                "candidate_wheel_sha256": _sha(wheel),
                "candidate_sdist_sha256": _sha(sdist),
                "qualification_sha256": _sha(qualification_path),
                "repository": "DaoyuanLi2816/mini-verl",
                "source_commit": commit,
                "valid": True,
                "workflow_run_attempt": 1,
                "workflow_run_id": 42,
            }
        ),
        encoding="utf-8",
    )
    return candidate, manifest_path, qualification, verification


def _build(tmp_path: Path, *, name: str = "out") -> Path:
    from miniverl.release_assets import prepare_release_assets

    candidate, manifest, qualification, verification = _fixture(tmp_path)
    output = tmp_path / name
    prepare_release_assets(
        candidate_dir=candidate,
        candidate_manifest=manifest,
        qualification_root=qualification,
        qualification=qualification / "qualification.json",
        verification=verification,
        output=output,
    )
    return output


def test_builds_exact_canonical_layout_and_checks_it(tmp_path: Path) -> None:
    from miniverl.release_assets import QUALIFICATION_SUM_FILES, check_release_assets

    output = _build(tmp_path)
    expected = {
        "dist/miniverl-0.11.0.dev0-py3-none-any.whl",
        "dist/miniverl-0.11.0.dev0.tar.gz",
        "SHA256SUMS",
        "candidate-manifest.json",
        "release-verification.json",
        "qualification.json",
        "qualification-SHA256SUMS",
        "qualification-release-smoke.json",
        "qualification-direct-gkd.json",
        "qualification-pg-k1.json",
        "qualification-smollm2.json",
        "qualification-evidence.tar.gz",
        "qualification-evidence-manifest.json",
    }
    assert {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    } == expected
    assert check_release_assets(output) == []
    sums = (output / "qualification-SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in sums] == list(QUALIFICATION_SUM_FILES)
    assert all(
        not path.name.endswith((".json.json", ".parquet.parquet", ".safetensors.safetensors"))
        for path in output.rglob("*")
    )


def test_archive_and_manifest_are_reproducible_and_explicit(tmp_path: Path) -> None:
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second")
    assert (first / "qualification-evidence.tar.gz").read_bytes() == (
        second / "qualification-evidence.tar.gz"
    ).read_bytes()
    assert (first / "qualification-evidence-manifest.json").read_bytes() == (
        second / "qualification-evidence-manifest.json"
    ).read_bytes()
    manifest = json.loads((first / "qualification-evidence-manifest.json").read_text())
    assert [member["semantic_role"] for member in manifest["members"]] == [
        "adapter_config",
        "adapter_weights",
        "adapter_manifest",
        "input_prompts",
        "run_summary",
    ]
    with tarfile.open(first / "qualification-evidence.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == sorted(member.name for member in members)
    assert all(
        member.isfile() and member.uid == member.gid == member.mtime == 0 for member in members
    )


@pytest.mark.parametrize(
    "names",
    [
        ["result.json", "RESULT.JSON"],
        ["result.json.json"],
        ["bad/name.json"],
        ["..tar.gz"],
        ["CON.json"],
        ["caf\u00e9.json", "cafe\u0301.json"],
    ],
)
def test_filename_hygiene_rejects_collisions_and_noncanonical_names(names: list[str]) -> None:
    from miniverl.release_assets import validate_canonical_names

    with pytest.raises(ValueError):
        validate_canonical_names(names)


def test_fails_closed_for_unknown_role_and_evidence_drift(tmp_path: Path) -> None:
    from miniverl.release_assets import prepare_release_assets

    candidate, manifest, qualification, verification = _fixture(tmp_path)
    qualification_path = qualification / "qualification.json"
    payload = json.loads(qualification_path.read_text())
    payload["artifacts"][0]["name"] = "unknown_role"
    qualification_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown evidence role"):
        prepare_release_assets(
            candidate, manifest, qualification, qualification_path, verification, tmp_path / "out"
        )
    candidate, manifest, qualification, verification = _fixture(tmp_path / "drift")
    (qualification / "run-summary.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match=r"size mismatch|checksum mismatch"):
        prepare_release_assets(
            candidate,
            manifest,
            qualification,
            qualification / "qualification.json",
            verification,
            tmp_path / "drift-out",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda payload: payload["artifacts"][0].update(path="../escape.json"),
            r"qualification validation|release chain validation",
        ),
        (
            lambda payload: payload["environment"].update(vram_gib=float("nan")),
            r"qualification validation",
        ),
        (
            lambda payload: payload["artifacts"][0].update(sha256="0" * 64),
            r"checksum mismatch|release chain validation",
        ),
        (
            lambda payload: payload["artifacts"][0].update(bytes=999999),
            r"size mismatch|release chain validation",
        ),
    ],
)
def test_fails_closed_for_traversal_nonfinite_and_declared_identity(
    tmp_path: Path, mutation, match: str
) -> None:
    from miniverl.release_assets import prepare_release_assets

    candidate, manifest, qualification, verification = _fixture(tmp_path)
    qualification_path = qualification / "qualification.json"
    payload = json.loads(qualification_path.read_text())
    mutation(payload)
    qualification_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        prepare_release_assets(
            candidate,
            manifest,
            qualification,
            qualification_path,
            verification,
            tmp_path / "out",
        )


def test_archive_mapping_collisions_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl import release_assets

    candidate, manifest, qualification, verification = _fixture(tmp_path)
    monkeypatch.setitem(
        release_assets.ARCHIVE_EVIDENCE,
        "run_summary",
        ("run_summary", "ADAPTER/adapter_config.json"),
    )
    with pytest.raises(ValueError, match="collision"):
        release_assets.prepare_release_assets(
            candidate,
            manifest,
            qualification,
            qualification / "qualification.json",
            verification,
            tmp_path / "out",
        )


def test_rejects_symlink_private_verification_and_nonempty_output(tmp_path: Path) -> None:
    from miniverl.release_assets import prepare_release_assets

    candidate, manifest, qualification, verification = _fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="empty"):
        prepare_release_assets(
            candidate,
            manifest,
            qualification,
            qualification / "qualification.json",
            verification,
            output,
        )

    verification.write_text('{"path":"C:\\\\Users\\\\secret\\\\file"}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"privacy|verification"):
        prepare_release_assets(
            candidate,
            manifest,
            qualification,
            qualification / "qualification.json",
            verification,
            tmp_path / "private",
        )

    target = qualification / "run-summary.json"
    target.unlink()
    try:
        target.symlink_to(qualification / "full/direct.json")
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match=r"regular|checksum|size"):
        prepare_release_assets(
            candidate,
            manifest,
            qualification,
            qualification / "qualification.json",
            verification,
            tmp_path / "linked",
        )


def test_check_rejects_extra_and_malicious_archive_members(tmp_path: Path) -> None:
    from miniverl.release_assets import check_release_assets

    output = _build(tmp_path)
    (output / "extra.json").write_text("{}")
    assert any("unexpected" in problem for problem in check_release_assets(output))
    (output / "extra.json").unlink()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in ("../escape", "duplicate", "duplicate"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    (output / "qualification-evidence.tar.gz").write_bytes(stream.getvalue())
    problems = check_release_assets(output)
    assert any("unsafe archive member" in problem for problem in problems)
    assert any("duplicate archive member" in problem for problem in problems)


def test_check_rebinds_primary_and_archive_members_to_qualification(tmp_path: Path) -> None:
    from miniverl.release_assets import QUALIFICATION_SUM_FILES, check_release_assets

    output = _build(tmp_path)
    primary = output / "qualification-direct-gkd.json"
    primary.write_text('{"substituted":true}\n', encoding="utf-8")
    (output / "qualification-SHA256SUMS").write_text(
        "".join(f"{_sha(output / name)}  {name}\n" for name in QUALIFICATION_SUM_FILES),
        encoding="utf-8",
    )
    assert any("qualification evidence" in problem for problem in check_release_assets(output))

    output = _build(tmp_path / "archive")
    manifest_path = output / "qualification-evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["members"][0]["semantic_role"] = "unknown_role"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (output / "qualification-SHA256SUMS").write_text(
        "".join(f"{_sha(output / name)}  {name}\n" for name in QUALIFICATION_SUM_FILES),
        encoding="utf-8",
    )
    assert any("semantic mapping" in problem for problem in check_release_assets(output))


def test_check_rejects_extra_directories(tmp_path: Path) -> None:
    from miniverl.release_assets import check_release_assets

    output = _build(tmp_path)
    (output / "unexpected-empty-directory").mkdir()
    assert any("directory set" in problem for problem in check_release_assets(output))


def test_module_import_is_torch_free() -> None:
    code = "import sys, miniverl.release_assets; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
