from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def _payload(tmp_path: Path) -> tuple[dict[str, object], Path]:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir(parents=True)
    wheel = artifact_root / "miniverl-0.10.1.dev0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    run_manifest = artifact_root / "run-manifest.json"
    run_manifest.write_text('{"status":"completed"}\n', encoding="utf-8")

    from miniverl.qualification import sha256_file

    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "miniverl_gpu_qualification",
        "level": "release_smoke",
        "status": "passed",
        "measured_at": "2026-08-14T12:00:00Z",
        "source_commit": "a" * 40,
        "miniverl_version": "0.10.1.dev0",
        "wheel": {
            "filename": wheel.name,
            "sha256": sha256_file(wheel),
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
            "packages": {
                "torch": "2.13.0+cu130",
                "transformers": "5.14.1",
                "peft": "0.18.0",
                "accelerate": "1.14.0",
                "bitsandbytes": "0.50.0",
                "numpy": "2.2.6",
                "pyarrow": "25.0.0",
            },
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
        "artifacts": [
            {
                "name": "run_manifest",
                "path": run_manifest.name,
                "sha256": sha256_file(run_manifest),
                "bytes": run_manifest.stat().st_size,
            }
        ],
        "checks": {
            "executed": [
                "wheel_install",
                "doctor",
                "plan",
                "run_dry_run",
                "real_actor_teacher_update",
                "peft_reload",
                "cuda_teardown",
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
    return payload, artifact_root


def test_strict_qualification_round_trip_and_artifact_hashes(tmp_path: Path) -> None:
    from miniverl.qualification import validate_qualification_payload

    payload, root = _payload(tmp_path / "private")
    assert validate_qualification_payload(payload, artifact_root=root) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda p: p.update(extra="invented"), "schema"),
        (lambda p: p["execution"].update(optimizer_updates=0), "schema"),  # type: ignore[union-attr]
        (lambda p: p["environment"].update(vram_gib=float("nan")), "schema"),  # type: ignore[union-attr]
        (lambda p: p.update(status="passed", source_commit="short"), "schema"),
    ],
)
def test_qualification_rejects_invalid_or_invented_fields(
    tmp_path: Path, mutation, expected: str
) -> None:
    from miniverl.qualification import validate_qualification_payload

    payload, root = _payload(tmp_path / "checksum")
    mutation(payload)
    assert validate_qualification_payload(payload, artifact_root=root)[0].startswith(expected)


def test_qualification_rejects_private_paths_and_hash_mismatch(tmp_path: Path) -> None:
    from miniverl.qualification import validate_qualification_payload

    payload, root = _payload(tmp_path / "private")
    payload["artifacts"][0]["path"] = "C:\\Users\\someone\\secret.json"  # type: ignore[index]
    assert validate_qualification_payload(payload, artifact_root=root)[0].startswith("privacy")

    payload, root = _payload(tmp_path / "checksum")
    payload["artifacts"][0]["sha256"] = "0" * 64  # type: ignore[index]
    assert any(
        "checksum" in problem
        for problem in validate_qualification_payload(payload, artifact_root=root)
    )


def _full_result(name: str) -> dict[str, object]:
    kinds = {
        "direct": ("single_gpu_opd_developer_workload", "measured"),
        "pg_k1": ("single_gpu_verl_pg_k1_systems_workload", "measured"),
        "smollm2": (
            "single_gpu_smollm2_direct_gkd_developer_workload",
            "maintainer_measured",
        ),
    }
    kind, status = kinds[name]
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "status": status,
        "source_commit": "a" * 40,
        "miniverl_version": "0.10.1.dev0",
        "hardware": {"gpu": "NVIDIA GeForce RTX 4080", "gpu_count": 1},
        "resume": {
            "status": "exact_match",
            "adapter_and_optimizer_byte_identical": True,
            "training_state_fields_identical": True,
            "trajectories_byte_identical": True,
        },
        "resource_contract": {"peak_reserved_within_limit": True},
        "verl": {"distributed_execution_tested": False},
        "artifacts": {"standard_peft_load_verified": name != "pg_k1"},
    }
    if name == "smollm2":
        payload["scaleout"] = {
            "artifact_bundle_complete": True,
            "upstream_config_parse_passed": True,
            "model_data_load_smoke_passed": True,
            "launchable": True,
            "distributed_execution_tested": False,
        }
    return payload


def test_full_qualification_promotion_binds_all_canonical_results(tmp_path: Path) -> None:
    from miniverl.qualification import validate_qualification_file
    from scripts.promote_full_gpu_qualification import promote

    payload, root = _payload(tmp_path)
    qualification = root / "qualification.json"
    qualification.write_text(json.dumps(payload), encoding="utf-8")
    results: dict[str, Path] = {}
    for name in ("direct", "pg_k1", "smollm2"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_full_result(name)), encoding="utf-8")
        results[name] = path

    promoted = promote(
        qualification,
        direct=results["direct"],
        pg_k1=results["pg_k1"],
        smollm2=results["smollm2"],
    )

    assert promoted.level == "full_qualification"
    assert validate_qualification_file(qualification) == []
    assert {item.name for item in promoted.artifacts} >= {
        "release_smoke_record",
        "full_direct_result",
        "full_pg_k1_result",
        "full_smollm2_result",
    }


def test_full_qualification_rejects_failed_resume_equivalence(tmp_path: Path) -> None:
    from scripts.promote_full_gpu_qualification import promote

    payload, root = _payload(tmp_path)
    qualification = root / "qualification.json"
    qualification.write_text(json.dumps(payload), encoding="utf-8")
    results: dict[str, Path] = {}
    for name in ("direct", "pg_k1", "smollm2"):
        result = _full_result(name)
        if name == "pg_k1":
            result["resume"]["trajectories_byte_identical"] = False  # type: ignore[index]
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        results[name] = path
    with pytest.raises(ValueError, match="equivalence"):
        promote(
            qualification,
            direct=results["direct"],
            pg_k1=results["pg_k1"],
            smollm2=results["smollm2"],
        )


def test_process_teardown_clears_bnb_and_cublas_caches() -> None:
    from scripts.run_gpu_qualification import _clear_process_global_cuda_caches

    calls: list[str] = []
    torch_module = SimpleNamespace(
        _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: calls.append("cublas"))
    )
    bnb_functional = SimpleNamespace(name2qmap={"dynamic": object()})
    _clear_process_global_cuda_caches(torch_module, bnb_functional)
    assert bnb_functional.name2qmap == {}
    assert calls == ["cublas"]


def test_committed_schema_is_generated_from_the_strict_model() -> None:
    from miniverl.qualification import qualification_json_schema

    committed = json.loads(
        Path("docs/generated/gpu-qualification-v1.schema.json").read_text(encoding="utf-8")
    )
    assert committed == qualification_json_schema()


def test_release_artifact_extraction_rejects_traversal_and_symlinks(tmp_path: Path) -> None:
    from scripts.verify_release_qualification import _safe_extract

    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escape.json", "{}")
    with pytest.raises(ValueError, match="unsafe"):
        _safe_extract(traversal, tmp_path / "out-traversal")

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("qualification.json")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(ValueError, match="symlink"):
        _safe_extract(symlink, tmp_path / "out-symlink")
