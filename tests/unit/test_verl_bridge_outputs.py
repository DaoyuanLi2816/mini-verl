"""Stem-specific, collision-safe, transactional bridge output publication."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from miniverl.errors import ConfigError, RunLockedError

PROFILE = "single-gpu-online-distillation-v1"


def _source() -> dict[str, Any]:
    return {
        "data": {
            "train_files": ["train.parquet"],
            "val_files": ["val.parquet"],
            "prompt_key": "prompt",
            "max_prompt_length": 512,
            "max_response_length": 128,
            "seed": 77,
        },
        "actor_rollout_ref": {
            "model": {"path": "Qwen/Qwen3-0.6B", "enable_gradient_checkpointing": True},
            "actor": {"optim": {"lr": 2.0e-5}},
        },
        "trainer": {
            "save_freq": 2,
            "test_freq": 1,
            "project_name": "bridge-smoke",
            "experiment_name": "strict-profile",
            "total_epochs": 3,
        },
    }


def _write(tmp_path: Path) -> Path:
    source = tmp_path / "verl.yaml"
    source.write_text(yaml.safe_dump(_source(), sort_keys=False), encoding="utf-8")
    return source


def _choices(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "environment": "calculator",
        "teacher_model": "Qwen/Qwen3-1.7B",
        "loss_profile": "topk-tail-reverse-kl",
        "schedule_mapping": "epochs-as-cycles",
    }
    base.update(overrides)
    return base


def _import(source: Path, out: Path, **kwargs: Any) -> dict[str, Any]:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    payload = dict(_choices())
    payload.update(kwargs)
    return import_verl_config(source, profile=PROFILE, target_verl=VERL_TAG, out=out, **payload)


# ----------------------------------------------------------------- naming


def test_accepted_import_publishes_only_stem_specific_files(tmp_path: Path) -> None:
    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"
    report = _import(source, out)

    assert out.is_file()
    assert (out.parent / "foo.import-report.json").is_file()
    assert not (out.parent / "import-report.json").exists()
    assert not (out.parent / "imported.template.yaml").exists()
    assert not (out.parent / "foo.template.yaml").exists()
    assert report["generated_path"] == "foo.yaml"
    assert report["report_path"] == "foo.import-report.json"
    assert sorted(p.name for p in out.parent.iterdir() if p.is_file()) == [
        "foo.import-report.json",
        "foo.yaml",
    ]


def test_template_import_publishes_only_stem_specific_files(tmp_path: Path) -> None:
    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"
    report = _import(source, out, environment=None)

    assert not out.exists()
    assert (out.parent / "foo.template.yaml").is_file()
    assert (out.parent / "foo.import-report.json").is_file()
    assert not (out.parent / "imported.template.yaml").exists()
    assert not (out.parent / "import-report.json").exists()
    assert report["status"] == "needs_user_input"
    assert report["generated_path"] == "foo.template.yaml"


def test_two_distinct_stems_coexist_in_one_directory(tmp_path: Path) -> None:
    source = _write(tmp_path)
    recipes = tmp_path / "recipes"
    _import(source, recipes / "foo.yaml")
    _import(source, recipes / "bar.yaml", teacher_model="Qwen/Qwen3-4B")

    assert sorted(p.name for p in recipes.iterdir() if p.is_file()) == [
        "bar.import-report.json",
        "bar.yaml",
        "foo.import-report.json",
        "foo.yaml",
    ]
    foo = json.loads((recipes / "foo.import-report.json").read_text(encoding="utf-8"))
    bar = json.loads((recipes / "bar.import-report.json").read_text(encoding="utf-8"))
    assert foo["user_confirmations"]["teacher_model"] == "Qwen/Qwen3-1.7B"
    assert bar["user_confirmations"]["teacher_model"] == "Qwen/Qwen3-4B"


def test_one_invocation_never_leaves_both_recipe_and_template_current(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"
    _import(source, out, environment=None)
    assert (out.parent / "foo.template.yaml").is_file()

    _import(source, out, overwrite=True)
    assert out.is_file()
    assert not (out.parent / "foo.template.yaml").exists()


# ------------------------------------------------------------- collisions


def test_repeated_same_stem_import_fails_before_modifying_anything(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"
    _import(source, out)
    before = out.read_bytes()
    report_before = (out.parent / "foo.import-report.json").read_bytes()

    with pytest.raises(ConfigError) as excinfo:
        _import(source, out, teacher_model="Qwen/Qwen3-4B")

    message = str(excinfo.value)
    assert "foo.yaml" in message
    assert "foo.import-report.json" in message
    assert "--overwrite" in message
    assert out.read_bytes() == before
    assert (out.parent / "foo.import-report.json").read_bytes() == report_before


def test_collision_lists_every_conflicting_path(tmp_path: Path) -> None:
    from miniverl.bridge.publish import OutputCollisionError

    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"
    _import(source, out)

    with pytest.raises(OutputCollisionError) as excinfo:
        _import(source, out)
    assert set(excinfo.value.conflicts) == {
        out.resolve(),
        (out.parent / "foo.import-report.json").resolve(),
    }


def test_explicit_overwrite_replaces_the_whole_output_family(tmp_path: Path) -> None:
    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"
    _import(source, out)
    report = _import(source, out, teacher_model="Qwen/Qwen3-4B", overwrite=True)

    generated = yaml.safe_load(out.read_text(encoding="utf-8"))
    on_disk = json.loads((out.parent / "foo.import-report.json").read_text(encoding="utf-8"))
    assert generated["models"]["teacher"]["model_id"] == "Qwen/Qwen3-4B"
    assert on_disk == report
    assert on_disk["user_confirmations"]["teacher_model"] == "Qwen/Qwen3-4B"


def test_supplying_out_does_not_imply_overwrite(tmp_path: Path) -> None:
    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"
    _import(source, out)
    with pytest.raises(ConfigError, match="--overwrite"):
        _import(source, out)


# -------------------------------------------------------- fault injection


def test_report_write_failure_rolls_back_the_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import publish

    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"

    original = publish.OutputTransaction.write_json

    def explode(self: Any, name: str, payload: Any) -> None:
        raise OSError("simulated report write failure")

    monkeypatch.setattr(publish.OutputTransaction, "write_json", explode)
    with pytest.raises(OSError, match="simulated report write failure"):
        _import(source, out)
    monkeypatch.setattr(publish.OutputTransaction, "write_json", original)

    assert not out.exists()
    assert not (out.parent / "foo.import-report.json").exists()
    assert list(out.parent.glob("*.tmp")) == []
    assert list(out.parent.glob(".*.tmp*")) == []


def test_recipe_write_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import publish

    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"

    def explode(self: Any, name: str, data: bytes) -> None:
        raise OSError("simulated recipe write failure")

    monkeypatch.setattr(publish.OutputTransaction, "write_bytes", explode)
    with pytest.raises(OSError, match="simulated recipe write failure"):
        _import(source, out)

    assert not out.exists()
    assert not (out.parent / "foo.import-report.json").exists()
    assert not (out.parent / "foo.template.yaml").exists()


def test_final_rename_failure_restores_the_previous_output_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import publish

    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"
    _import(source, out)
    recipe_before = out.read_bytes()
    report_before = (out.parent / "foo.import-report.json").read_bytes()

    calls = {"count": 0}
    real_replace = publish._replace

    def flaky(src: Path, dst: Path) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated rename failure")
        real_replace(src, dst)

    monkeypatch.setattr(publish, "_replace", flaky)
    with pytest.raises(OSError, match="simulated rename failure"):
        _import(source, out, teacher_model="Qwen/Qwen3-4B", overwrite=True)

    assert out.read_bytes() == recipe_before
    assert (out.parent / "foo.import-report.json").read_bytes() == report_before


def test_a_failed_publication_leaves_no_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import publish

    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"

    def explode(self: Any, name: str, payload: Any) -> None:
        raise OSError("boom")

    monkeypatch.setattr(publish.OutputTransaction, "write_json", explode)
    with pytest.raises(OSError):
        _import(source, out)

    leftovers = [p for p in out.parent.iterdir() if p.name != ".miniverl-locks"]
    assert leftovers == []


# ------------------------------------------------------------ concurrency


def test_concurrent_same_stem_writers_cannot_mix_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministic interleave: both writers stage before either commits."""
    from miniverl.bridge import publish

    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"
    _import(source, out)

    entered = threading.Barrier(2, timeout=30)
    results: dict[str, Any] = {}
    errors: list[BaseException] = []
    original_commit = publish.OutputTransaction.commit
    seen = threading.Semaphore(0)

    def synchronized_commit(self: Any) -> None:
        # Both threads must have reached the lock before either may publish.
        seen.release()
        original_commit(self)

    monkeypatch.setattr(publish.OutputTransaction, "commit", synchronized_commit)

    def worker(teacher: str) -> None:
        try:
            entered.wait()
            results[teacher] = _import(
                source, out, teacher_model=teacher, overwrite=True, lock_timeout=30.0
            )
        except BaseException as exc:  # pragma: no cover - surfaced by assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("Qwen/Qwen3-1.7B",)),
        threading.Thread(target=worker, args=("Qwen/Qwen3-4B",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    assert len(results) == 2

    on_disk_report = json.loads((out.parent / "foo.import-report.json").read_text(encoding="utf-8"))
    on_disk_recipe = out.read_bytes()
    import hashlib

    assert on_disk_report["generated_miniverl_sha256"] == hashlib.sha256(on_disk_recipe).hexdigest()
    assert on_disk_report in results.values()


def test_concurrent_same_stem_writer_fails_fast_without_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import publish

    source = _write(tmp_path)
    out = tmp_path / "recipes" / "foo.yaml"
    holder = publish.OutputTransaction(
        targets={"recipe": out},
        stem="foo",
        lock_root=out.parent,
        overwrite=True,
    )
    holder.__enter__()
    try:
        with pytest.raises(RunLockedError):
            _import(source, out, overwrite=True, lock_timeout=0.0)
    finally:
        holder.rollback()


def test_concurrent_different_stem_writers_both_succeed(tmp_path: Path) -> None:
    source = _write(tmp_path)
    recipes = tmp_path / "recipes"
    start = threading.Barrier(2, timeout=30)
    errors: list[BaseException] = []

    def worker(stem: str) -> None:
        try:
            start.wait()
            _import(source, recipes / f"{stem}.yaml", lock_timeout=30.0)
        except BaseException as exc:  # pragma: no cover - surfaced by assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(stem,)) for stem in ("foo", "bar")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    assert (recipes / "foo.yaml").is_file()
    assert (recipes / "bar.yaml").is_file()
    assert (recipes / "foo.import-report.json").is_file()
    assert (recipes / "bar.import-report.json").is_file()


# ---------------------------------------------------------- path behavior


def test_output_family_is_derived_from_the_requested_stem_on_any_platform(
    tmp_path: Path,
) -> None:
    from miniverl.bridge.publish import import_output_targets

    targets = import_output_targets(tmp_path / "sub" / "my.recipe.yaml")
    assert targets["recipe"].name == "my.recipe.yaml"
    assert targets["report"].name == "my.recipe.import-report.json"
    assert targets["template"].name == "my.recipe.template.yaml"
    assert {path.parent for path in targets.values()} == {tmp_path / "sub"}


def test_relative_and_absolute_out_paths_agree(tmp_path: Path, monkeypatch: Any) -> None:
    from miniverl.bridge.publish import import_output_targets

    monkeypatch.chdir(tmp_path)
    relative = import_output_targets(Path("recipes/foo.yaml"))
    absolute = import_output_targets(tmp_path / "recipes" / "foo.yaml")
    assert {k: v.resolve() for k, v in relative.items()} == {
        k: v.resolve() for k, v in absolute.items()
    }
