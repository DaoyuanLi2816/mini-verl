from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import pytest
import yaml


def _safetensors_bytes() -> bytes:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0)


def _snapshot(root: Path, revision: str) -> Path:
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["FixtureForCausalLM"], "model_type": "fixture"}),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(_safetensors_bytes())
    (snapshot / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "model": {"type": "WordLevel", "vocab": {"x": 0}}}),
        encoding="utf-8",
    )
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}), encoding="utf-8"
    )
    (snapshot / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (snapshot / "LICENSE").write_text("fixture license\n", encoding="utf-8")
    model_id = "Qwen/Qwen3-0.6B" if "student" in root.name else "Qwen/Qwen3-1.7B"
    files = {
        path.relative_to(snapshot).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in snapshot.iterdir()
        if path.is_file()
    }
    (snapshot / "miniverl-snapshot.json").write_text(
        json.dumps({"model_id": model_id, "revision": revision, "files": files}),
        encoding="utf-8",
    )
    return snapshot


def _bundle(
    tmp_path: Path,
    *,
    teacher_adapter: bool = False,
    profile: str = "verl-opd-v0.8-single-gpu-v1",
) -> tuple[Path, str, str]:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.export import export_verl_bundle
    from tests.unit.test_verl_opd_export import _opd_run

    run, _, _ = _opd_run(tmp_path, teacher_adapter=teacher_adapter, profile=profile)
    bundle = tmp_path / "bundle"
    export_verl_bundle(run, target_verl=VERL_TAG, out=bundle)
    student_revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    teacher_revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    return bundle, student_revision, teacher_revision


def test_pg_k1_bundle_materializes_without_a_topk_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import materialize

    profile = "verl-opd-v0.8-single-gpu-pg-k1-v1"
    bundle, student_revision, teacher_revision = _bundle(tmp_path, profile=profile)
    student = _snapshot(tmp_path / "student-cache", student_revision)
    teacher = _snapshot(tmp_path / "teacher-cache", teacher_revision)

    def passed_pg(root: Path) -> dict[str, Any]:
        recipe = yaml.safe_load(
            (root / "recipe/verl-opd-overrides.yaml").read_text(encoding="utf-8")
        )
        loss = recipe["distillation"]["distillation_loss"]
        assert "topk" not in loss
        assert loss["loss_mode"] == "k1"
        assert loss["use_policy_gradient"] is True
        return {"status": "passed", "checks": {"sampled_k1_policy_gradient_contract": "passed"}}

    monkeypatch.setattr(materialize, "_validate_upstream_bundle", passed_pg)
    report = materialize.materialize_verl_bundle(
        bundle,
        student_snapshot=student,
        teacher_snapshot=teacher,
        offline=True,
    )

    assert report["profile"] == profile
    assert report["launchable"] is True
    assert report["materialization"]["upstream_validation"]["checks"] == {
        "sampled_k1_policy_gradient_contract": "passed"
    }


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _passed_upstream(root: Path) -> dict[str, Any]:
    resolved = root / "recipe" / "verl-opd-resolved.yaml"
    resolved.write_text("trainer:\n  total_training_steps: 1\n", encoding="utf-8")
    return {
        "status": "passed",
        "resolved_config": resolved.relative_to(root).as_posix(),
        "checks": {
            "config_parse": "passed",
            "parquet_schema": "passed",
            "student_peft": "passed",
            "student_snapshot": "passed",
            "teacher_snapshot": "passed",
            "topk_contract": "passed",
        },
    }


def test_materialization_accepts_a_padded_model_vocabulary() -> None:
    from miniverl.bridge.materialize import _validate_vocab_domain

    _validate_vocab_domain(model_vocab_size=151_936, tokenizer_max_token_id=151_668)
    with pytest.raises(ValueError, match="tokenizer ID domain"):
        _validate_vocab_domain(model_vocab_size=151_668, tokenizer_max_token_id=151_668)


def test_materialize_publishes_complete_launchable_bundle_transactionally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import materialize

    bundle, student_revision, teacher_revision = _bundle(tmp_path)
    student = _snapshot(tmp_path / "student-cache", student_revision)
    teacher = _snapshot(tmp_path / "teacher-cache", teacher_revision)
    monkeypatch.setattr(materialize, "_validate_upstream_bundle", _passed_upstream)

    report = materialize.materialize_verl_bundle(
        bundle,
        student_snapshot=student,
        teacher_snapshot=teacher,
        offline=True,
    )

    assert report["launchable"] is True
    assert report["student_artifact_loadable"] is True
    assert report["teacher_artifact_loadable"] is True
    assert report["upstream_parse_passed"] is True
    assert report["distributed_execution_tested"] is False
    assert (bundle / "recipe/launch.sh").is_file()
    launch = (bundle / "recipe/launch.sh").read_text(encoding="utf-8")
    assert "python -m verl.trainer.main_ppo" in launch
    assert "--config-path" in launch
    assert not (bundle / "recipe/launch.template.sh").exists()
    assert (bundle / "model/base/LICENSE").read_text() == "fixture license\n"
    assert (bundle / "teacher/base/LICENSE").read_text() == "fixture license\n"
    manifest = json.loads(
        (bundle / "provenance/materialization-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["student"]["revision"] == student_revision
    assert manifest["teacher"]["revision"] == teacher_revision
    assert "model/base/model.safetensors" in manifest["student"]["files"]
    assert "teacher/base/model.safetensors" in manifest["teacher"]["files"]
    assert report["launch_blockers"] == []

    from miniverl.bridge import doctor

    diagnosis = doctor.inspect_bridge_bundle(bundle)
    assert diagnosis["verdict"] == "ok"
    assert diagnosis["student_snapshot_loadability"]["status"] == "ok"
    assert diagnosis["teacher_snapshot_loadability"]["status"] == "ok"
    assert diagnosis["launchable"] is False
    assert diagnosis["bundle_declared_claims"]["launchable"] is True

    monkeypatch.setattr(
        doctor,
        "_installed_verl",
        lambda: {"status": "ok", "expected_commit": "fixture", "version": "fixture"},
    )
    monkeypatch.setattr(
        doctor,
        "_recompute_upstream_smoke",
        lambda root, installed, *, enabled: {"status": "passed"},
    )
    strict = doctor.inspect_bridge_bundle(bundle, require_verl=True)
    assert strict["verdict"] == "ok"
    assert strict["launchable"] is True


def test_materialize_failure_leaves_original_bundle_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import materialize

    bundle, student_revision, teacher_revision = _bundle(tmp_path)
    student = _snapshot(tmp_path / "student-cache", student_revision)
    teacher = _snapshot(tmp_path / "teacher-cache", teacher_revision)
    before = _tree_hashes(bundle)
    real_copy = materialize._copy_snapshot_tree
    calls = 0

    def fail_second_copy(source: Path, destination: Path, *, bundle_prefix: str) -> Any:
        nonlocal calls
        calls += 1
        result = real_copy(source, destination, bundle_prefix=bundle_prefix)
        if calls == 2:
            raise OSError("injected copy failure")
        return result

    monkeypatch.setattr(materialize, "_copy_snapshot_tree", fail_second_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        materialize.materialize_verl_bundle(
            bundle,
            student_snapshot=student,
            teacher_snapshot=teacher,
            offline=True,
        )

    assert _tree_hashes(bundle) == before
    assert not (bundle / "model/base").exists()


def test_local_snapshot_must_be_bound_to_the_expected_immutable_revision(tmp_path: Path) -> None:
    from miniverl.bridge.materialize import materialize_verl_bundle
    from miniverl.errors import ConfigError

    bundle, student_revision, teacher_revision = _bundle(tmp_path)
    student = _snapshot(tmp_path / "student-cache", "f" * 40)
    teacher = _snapshot(tmp_path / "teacher-cache", teacher_revision)

    with pytest.raises(ConfigError, match="immutable revision"):
        materialize_verl_bundle(
            bundle,
            student_snapshot=student,
            teacher_snapshot=teacher,
            offline=True,
        )
    assert not (bundle / "model/base").exists()
    assert student_revision not in str(student)


def test_portable_snapshot_manifest_must_bind_every_declared_file(tmp_path: Path) -> None:
    from miniverl.bridge.materialize import _validate_local_revision
    from miniverl.errors import ConfigError

    snapshot = tmp_path / "portable"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "miniverl-snapshot.json").write_text(
        json.dumps(
            {
                "model_id": "fixture/model",
                "revision": "a" * 40,
                "files": {"config.json": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="hash mismatch"):
        _validate_local_revision(
            snapshot,
            model_id="fixture/model",
            revision="a" * 40,
            downloaded=False,
        )


def test_portable_snapshot_manifest_must_enumerate_every_file(tmp_path: Path) -> None:
    from miniverl.bridge.materialize import _validate_local_revision
    from miniverl.errors import ConfigError

    snapshot = tmp_path / "portable"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "unlisted.bin").write_bytes(b"not identity-bound")
    (snapshot / "miniverl-snapshot.json").write_text(
        json.dumps(
            {
                "model_id": "fixture/model",
                "revision": "a" * 40,
                "files": {
                    "config.json": hashlib.sha256(b"{}").hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="enumerate every"):
        _validate_local_revision(
            snapshot,
            model_id="fixture/model",
            revision="a" * 40,
            downloaded=False,
        )


def test_model_shard_index_may_not_escape_snapshot(tmp_path: Path) -> None:
    from miniverl.bridge.materialize import _snapshot_model_files
    from miniverl.errors import ConfigError

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "outside.safetensors").write_bytes(_safetensors_bytes())
    with pytest.raises(ConfigError, match="unsafe paths"):
        _snapshot_model_files(snapshot)


def test_offline_download_dereferences_only_one_repository_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    huggingface_hub = pytest.importorskip("huggingface_hub")

    from miniverl.bridge.materialize import _download_snapshot

    cache = tmp_path / "models--fixture--model"
    source = cache / "snapshots" / ("a" * 40)
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "onnx").mkdir()
    (source / "onnx/model_q4.onnx").write_bytes(b"not part of a Transformers snapshot")
    received: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        received.update(kwargs)
        return str(source)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    destination = tmp_path / "regular"

    resolved = _download_snapshot(
        model_id="fixture/model",
        revision="a" * 40,
        destination=destination,
        offline=True,
    )

    assert resolved == destination
    assert (resolved / "config.json").read_text(encoding="utf-8") == "{}"
    assert not (resolved / "config.json").is_symlink()
    assert not (resolved / "onnx").exists()
    assert "model.safetensors" in received["allow_patterns"]
    assert "*.onnx" not in received["allow_patterns"]


def test_snapshot_symlink_is_rejected_before_publication(tmp_path: Path) -> None:
    from miniverl.bridge.materialize import materialize_verl_bundle
    from miniverl.errors import ConfigError

    bundle, student_revision, teacher_revision = _bundle(tmp_path)
    student = _snapshot(tmp_path / "student-cache", student_revision)
    teacher = _snapshot(tmp_path / "teacher-cache", teacher_revision)
    target = student / "outside.bin"
    target.write_bytes(b"outside")
    link = student / "linked.bin"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("this Windows account cannot create symlinks")
    manifest_path = student / "miniverl-snapshot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = {
        path.relative_to(student).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in student.iterdir()
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ConfigError, match=r"symlink|reparse"):
        materialize_verl_bundle(
            bundle,
            student_snapshot=student,
            teacher_snapshot=teacher,
            offline=True,
        )
    assert not (bundle / "model/base").exists()


def test_teacher_adapter_merge_requires_explicit_consent(tmp_path: Path) -> None:
    from miniverl.bridge.materialize import materialize_verl_bundle
    from miniverl.errors import ConfigError

    bundle, student_revision, teacher_revision = _bundle(tmp_path, teacher_adapter=True)
    student = _snapshot(tmp_path / "student-cache", student_revision)
    teacher = _snapshot(tmp_path / "teacher-cache", teacher_revision)

    with pytest.raises(ConfigError, match="--merge-teacher-adapter"):
        materialize_verl_bundle(
            bundle,
            student_snapshot=student,
            teacher_snapshot=teacher,
            offline=True,
        )


def test_teacher_adapter_merge_is_new_explicit_audited_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import materialize

    bundle, student_revision, teacher_revision = _bundle(tmp_path, teacher_adapter=True)
    student = _snapshot(tmp_path / "student-cache", student_revision)
    teacher = _snapshot(tmp_path / "teacher-cache", teacher_revision)
    original_teacher = _tree_hashes(teacher)

    def fake_merge(base: Path, adapter: Path, destination: Path) -> dict[str, str]:
        assert (adapter / "adapter_model.safetensors").is_file()
        shutil.copytree(base, destination)
        (destination / "merge-proof.json").write_text(
            json.dumps(
                {
                    "adapter_sha256": hashlib.sha256(
                        (adapter / "adapter_model.safetensors").read_bytes()
                    ).hexdigest()
                }
            ),
            encoding="utf-8",
        )
        return {"transformers": "fixture", "peft": "fixture", "torch": "fixture"}

    monkeypatch.setattr(materialize, "_merge_teacher_adapter", fake_merge)
    monkeypatch.setattr(materialize, "_validate_upstream_bundle", _passed_upstream)
    report = materialize.materialize_verl_bundle(
        bundle,
        student_snapshot=student,
        teacher_snapshot=teacher,
        offline=True,
        merge_teacher_adapter=True,
    )

    assert report["launchable"] is True
    assert _tree_hashes(teacher) == original_teacher
    manifest = json.loads(
        (bundle / "provenance/materialization-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["teacher"]["adapter_merged"] is True
    assert manifest["teacher"]["merge_software"]["peft"] == "fixture"
    assert "teacher/base/merge-proof.json" in manifest["teacher"]["files"]


def test_upstream_validation_failure_does_not_publish_partial_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import materialize
    from miniverl.errors import ConfigError

    bundle, student_revision, teacher_revision = _bundle(tmp_path)
    student = _snapshot(tmp_path / "student-cache", student_revision)
    teacher = _snapshot(tmp_path / "teacher-cache", teacher_revision)
    before = _tree_hashes(bundle)

    def fail_upstream(root: Path) -> dict[str, Any]:
        raise ConfigError("pinned upstream validation failed")

    monkeypatch.setattr(materialize, "_validate_upstream_bundle", fail_upstream)
    with pytest.raises(ConfigError, match="pinned upstream"):
        materialize.materialize_verl_bundle(
            bundle,
            student_snapshot=student,
            teacher_snapshot=teacher,
            offline=True,
        )
    assert _tree_hashes(bundle) == before


def test_directory_swap_rolls_back_when_new_bundle_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge.materialize import _replace_directory

    old = tmp_path / "bundle"
    new = tmp_path / "staged"
    old.mkdir()
    new.mkdir()
    (old / "value.txt").write_text("old", encoding="utf-8")
    (new / "value.txt").write_text("new", encoding="utf-8")
    real_replace = Path.replace

    def fail_new_once(path: Path, target: Path) -> Path:
        if path == new:
            raise OSError("injected publication rename failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_new_once)
    with pytest.raises(OSError, match="publication rename"):
        _replace_directory(new, old)

    assert (old / "value.txt").read_text(encoding="utf-8") == "old"
    assert (new / "value.txt").read_text(encoding="utf-8") == "new"
