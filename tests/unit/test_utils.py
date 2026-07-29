"""Contracts of the support layer: errors, lazy imports, seeding, env, runs, gpu, logging.

These modules are the ones every other subsystem depends on, so the invariants
protected here are cross-cutting:

* an error is only useful if its ``hint`` reaches the user, on its own line;
* a missing optional dependency must name the extra that provides it;
* seeds must be reproducible *across processes*, which rules out Python's
  salted ``hash``;
* a run directory is meant to be shareable, so the captured environment must
  contain no hostname, no username, no home directory and no environment
  variable outside :data:`TRACKED_ENV_VARS` (privacy regression test);
* run artifacts live inside the run root and never collide with each other;
* the CUDA helpers must degrade to honest zeros on a machine without CUDA;
* logging installs exactly one handler no matter how many call sites ask.
"""

from __future__ import annotations

import getpass
import hashlib
import importlib.util
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import HAS_TORCH, requires_torch

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
requires_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="requires numpy")

_SHA = "0123456789abcdef0123456789abcdef01234567"


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def preserve_rng() -> Iterator[None]:
    """Restore every global RNG this file perturbs, using stdlib calls only.

    The fixture deliberately does not use :mod:`miniverl.utils.seeding` so a bug
    in ``capture_rng``/``restore_rng`` cannot hide behind its own round trip.
    """
    python_state = random.getstate()
    numpy_state = None
    torch_state = None
    if _HAS_NUMPY:
        import numpy as np

        numpy_state = np.random.get_state()
    if HAS_TORCH:
        import torch

        torch_state = torch.get_rng_state().clone()
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            import numpy as np

            np.random.set_state(numpy_state)
        if torch_state is not None:
            import torch

            torch.set_rng_state(torch_state)


@pytest.fixture
def restore_torch_determinism() -> Iterator[None]:
    """Undo the process-wide determinism flags ``seed_everything`` sets."""
    import torch

    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)


@pytest.fixture
def miniverl_logger() -> Iterator[logging.Logger]:
    """The ``miniverl`` logger, with handlers/level/propagate restored after."""
    logger = logging.getLogger("miniverl")
    handlers = list(logger.handlers)
    level = logger.level
    propagate = logger.propagate
    try:
        yield logger
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate


# ----------------------------------------------------------------------- errors


def test_error_str_puts_the_hint_on_a_second_line() -> None:
    from miniverl.errors import ConfigError

    err = ConfigError("loss.chunk_size must be positive", hint="set loss.chunk_size: 256")
    assert str(err).splitlines() == [
        "loss.chunk_size must be positive",
        "  hint: set loss.chunk_size: 256",
    ]
    assert err.message == "loss.chunk_size must be positive"
    assert err.hint == "set loss.chunk_size: 256"
    # ``args`` stays single-element so ``raise ... from`` chains print cleanly.
    assert err.args == ("loss.chunk_size must be positive",)


def test_error_str_is_the_bare_message_without_a_hint() -> None:
    from miniverl.errors import MiniVerlError

    err = MiniVerlError("something broke")
    assert str(err) == "something broke"
    assert "hint" not in str(err)
    assert err.hint is None


def test_missing_dependency_error_builds_the_exact_pip_hint() -> None:
    from miniverl.errors import MissingDependencyError

    err = MissingDependencyError("bitsandbytes", "cuda", "4-bit quantization")
    assert err.package == "bitsandbytes"
    assert err.extra == "cuda"
    assert err.hint == 'pip install "miniverl[cuda]"'
    assert err.message == (
        "4-bit quantization requires the optional dependency 'bitsandbytes', "
        "which is not installed."
    )
    assert str(err).splitlines() == [err.message, '  hint: pip install "miniverl[cuda]"']


def test_every_exported_error_derives_from_the_base_and_is_documented() -> None:
    from miniverl import errors as errors_module

    for name in errors_module.__all__:
        cls = getattr(errors_module, name)
        assert issubclass(cls, errors_module.MiniVerlError), name
        assert cls.__doc__, f"{name} has no docstring"


def test_error_taxonomy_lets_callers_catch_families() -> None:
    from miniverl.errors import (
        AlignmentError,
        CacheCorruptionError,
        CacheError,
        MiniVerlError,
        StaleCacheError,
        TokenizerMismatchError,
    )

    assert issubclass(TokenizerMismatchError, AlignmentError)
    assert issubclass(StaleCacheError, CacheError)
    assert issubclass(CacheCorruptionError, CacheError)
    assert not issubclass(CacheError, AlignmentError)
    assert issubclass(AlignmentError, MiniVerlError)


# ------------------------------------------------------------------------- lazy


def test_have_module_is_true_for_installed_modules() -> None:
    from miniverl.utils.lazy import have_module

    assert have_module("json") is True
    assert have_module("miniverl") is True
    assert have_module("miniverl.utils.lazy") is True


def test_have_module_is_false_for_absent_modules() -> None:
    from miniverl.utils.lazy import have_module

    assert have_module("miniverl_definitely_not_installed") is False
    # A missing *parent* package makes find_spec raise; it must still be False.
    assert have_module("miniverl_definitely_not_installed.submodule") is False


def test_require_module_returns_the_imported_module() -> None:
    import json as json_module

    from miniverl.utils.lazy import require_module

    assert require_module("json", "train", "This operation") is json_module


def test_require_module_raises_missing_dependency_with_the_extra_and_purpose() -> None:
    from miniverl.errors import MissingDependencyError
    from miniverl.utils.lazy import require_module

    with pytest.raises(MissingDependencyError) as excinfo:
        require_module("miniverl_absent_backend", "train", "Training")
    err = excinfo.value
    assert err.package == "miniverl_absent_backend"
    assert err.extra == "train"
    assert err.hint == 'pip install "miniverl[train]"'
    assert "Training requires the optional dependency 'miniverl_absent_backend'" in err.message
    # The original ImportError is chained so tracebacks stay debuggable.
    assert isinstance(err.__cause__, ImportError)


@pytest.mark.torch
@requires_torch
def test_require_torch_returns_the_torch_module() -> None:
    import torch

    from miniverl.utils.lazy import require_torch

    assert require_torch("Training") is torch


# ---------------------------------------------------------------------- seeding


def test_derive_seed_is_a_stable_sha256_digest() -> None:
    """A pinned literal: this value must not move between releases or processes."""
    from miniverl.utils.seeding import derive_seed

    assert derive_seed("miniverl") == 5796897314664446566
    expected = int.from_bytes(hashlib.sha256(b"miniverl|7").digest()[:8], "big") >> 1
    assert derive_seed("miniverl", 7) == expected
    assert derive_seed("miniverl") == derive_seed("miniverl")
    assert derive_seed("miniverl").bit_length() <= 63


def test_derive_seed_separates_different_inputs() -> None:
    from miniverl.utils.seeding import derive_seed

    values = [
        derive_seed("run-a"),
        derive_seed("run-b"),
        derive_seed("run-a", 0),
        derive_seed("run-a", 1),
        derive_seed(0, "run-a"),
    ]
    assert len(set(values)) == len(values)


def test_derive_seed_ignores_pythonhashseed() -> None:
    """Two interpreters with different hash salts must agree on the seed."""
    from miniverl.utils.seeding import derive_seed

    code = "from miniverl.utils.seeding import derive_seed;print(derive_seed('miniverl', 7))"
    outputs = []
    for hash_seed in ("0", "424242"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=300,
        )
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1] == str(derive_seed("miniverl", 7))


def test_seed_everything_reproduces_python_draws(preserve_rng: None) -> None:
    from miniverl.utils.seeding import seed_everything

    seed_everything(1234, deterministic=False)
    first = [random.random(), random.randrange(10**6)]
    seed_everything(1234, deterministic=False)
    assert [random.random(), random.randrange(10**6)] == first
    seed_everything(4321, deterministic=False)
    assert [random.random(), random.randrange(10**6)] != first


@requires_numpy
def test_seed_everything_reproduces_numpy_draws(preserve_rng: None) -> None:
    import numpy as np

    from miniverl.utils.seeding import seed_everything

    seed_everything(2026, deterministic=False)
    first = np.random.rand(5).tolist()
    seed_everything(2026, deterministic=False)
    assert np.random.rand(5).tolist() == first


@pytest.mark.torch
@requires_torch
def test_seed_everything_reproduces_torch_draws(preserve_rng: None) -> None:
    import torch

    from miniverl.utils.seeding import seed_everything

    seed_everything(99, deterministic=False)
    first = torch.randn(4).tolist()
    seed_everything(99, deterministic=False)
    assert torch.randn(4).tolist() == first
    seed_everything(100, deterministic=False)
    assert torch.randn(4).tolist() != first


@pytest.mark.torch
@requires_torch
def test_deterministic_mode_sets_the_documented_flags(
    monkeypatch: pytest.MonkeyPatch,
    preserve_rng: None,
    restore_torch_determinism: None,
) -> None:
    import torch

    from miniverl.utils.seeding import seed_everything

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.backends.cudnn.benchmark = True
    seed_everything(7, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True
    # warn-only keeps a long run alive when a kernel has no deterministic path.
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True


@pytest.mark.torch
@requires_torch
def test_deterministic_mode_does_not_override_an_existing_cublas_config(
    monkeypatch: pytest.MonkeyPatch,
    preserve_rng: None,
    restore_torch_determinism: None,
) -> None:
    from miniverl.utils.seeding import seed_everything

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    seed_everything(7, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_capture_and_restore_round_trips_python_random(preserve_rng: None) -> None:
    from miniverl.utils.seeding import capture_rng, restore_rng

    random.seed(5)
    before = [random.random() for _ in range(3)]
    snapshot = capture_rng()
    expected = [random.random() for _ in range(3)]
    assert before != expected
    for _ in range(11):  # advance the stream well past the captured point
        random.random()
    restore_rng(snapshot)
    assert [random.random() for _ in range(3)] == expected


@requires_numpy
def test_capture_and_restore_round_trips_numpy(preserve_rng: None) -> None:
    import numpy as np

    from miniverl.utils.seeding import capture_rng, restore_rng

    np.random.seed(17)
    snapshot = capture_rng()
    expected = np.random.rand(4).tolist()
    np.random.rand(23)
    restore_rng(snapshot)
    assert np.random.rand(4).tolist() == expected


@pytest.mark.torch
@requires_torch
def test_capture_and_restore_round_trips_torch(preserve_rng: None) -> None:
    import torch

    from miniverl.utils.seeding import capture_rng, restore_rng

    torch.manual_seed(31)
    snapshot = capture_rng()
    assert snapshot.torch_state is not None
    expected = torch.randn(6).tolist()
    torch.randn(29)
    restore_rng(snapshot)
    assert torch.randn(6).tolist() == expected


def test_rng_snapshot_survives_a_json_round_trip(preserve_rng: None) -> None:
    """Checkpoints are data files, so the snapshot must be pickle-free JSON."""
    from miniverl.utils.seeding import RngSnapshot, capture_rng, restore_rng

    random.seed(11)
    snapshot = capture_rng()
    expected = [random.random() for _ in range(4)]
    payload = snapshot.to_dict()
    assert set(payload) == {"python_state", "torch_state", "cuda_states", "numpy_state"}
    rebuilt = RngSnapshot.from_dict(json.loads(json.dumps(payload)))
    assert rebuilt.to_dict() == payload
    restore_rng(rebuilt)
    assert [random.random() for _ in range(4)] == expected


def test_rng_snapshot_from_dict_tolerates_absent_optional_state() -> None:
    from miniverl.utils.seeding import RngSnapshot

    rebuilt = RngSnapshot.from_dict({"python_state": "[]", "cuda_states": None})
    assert rebuilt.torch_state is None
    assert rebuilt.numpy_state is None
    assert rebuilt.cuda_states == []


# -------------------------------------------------------------------------- env


def _personal_tokens() -> list[str]:
    tokens = [
        platform.node(),
        os.environ.get("USERNAME") or "",
        os.environ.get("USER") or "",
        getpass.getuser(),
        str(Path.home()),
    ]
    return [token for token in tokens if token]


def test_collect_environment_returns_the_documented_keys() -> None:
    from miniverl.utils.env import collect_environment

    env = collect_environment()
    assert set(env) == {
        "python_version",
        "python_implementation",
        "os",
        "os_release",
        "platform",
        "machine",
        "processor_family",
        "cpu_count",
        "packages",
        "gpu",
        "tracked_env_vars",
        "git_commit",
    }
    assert env["python_version"] == ".".join(str(p) for p in sys.version_info[:3])
    assert env["python_implementation"] == platform.python_implementation()
    assert isinstance(env["gpu"], dict)
    assert isinstance(env["gpu"]["available"], bool)
    assert isinstance(env["packages"], dict)


def test_collect_environment_contains_no_personal_data() -> None:
    """Privacy regression: a run directory is meant to be shareable as-is."""
    from miniverl.utils.env import collect_environment

    blob = json.dumps(collect_environment()).lower()
    for token in _personal_tokens():
        assert token.lower() not in blob, f"{token!r} leaked into the environment record"
        # json escapes backslashes, so the escaped spelling must be checked too.
        escaped = json.dumps(token)[1:-1].lower()
        assert escaped not in blob, f"{token!r} leaked (escaped) into the environment record"
    assert "hostname" not in blob
    assert "username" not in blob


def test_only_allowlisted_env_vars_are_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    from miniverl.utils.env import TRACKED_ENV_VARS, collect_environment

    monkeypatch.setenv("MINIVERL_TEST_SECRET", "hunter2-must-not-be-recorded")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    env = collect_environment()
    assert set(env["tracked_env_vars"]) <= set(TRACKED_ENV_VARS)
    assert env["tracked_env_vars"]["OMP_NUM_THREADS"] == "3"
    blob = json.dumps(env)
    assert "MINIVERL_TEST_SECRET" not in blob
    assert "hunter2-must-not-be-recorded" not in blob


def test_empty_tracked_env_vars_are_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    from miniverl.utils.env import collect_environment

    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "")
    assert "TOKENIZERS_PARALLELISM" not in collect_environment()["tracked_env_vars"]


def test_package_versions_covers_the_result_affecting_packages() -> None:
    from miniverl import __version__
    from miniverl.utils.env import package_versions

    versions = package_versions()
    assert versions["miniverl"] == __version__
    for name in ("torch", "transformers", "peft", "numpy", "pydantic", "rich", "typer"):
        assert name in versions
        assert versions[name] is None or isinstance(versions[name], str)


def test_gpu_info_is_honest_when_cuda_is_absent() -> None:
    from miniverl.utils.env import gpu_info

    info = gpu_info()
    assert isinstance(info["available"], bool)
    if info["available"]:
        assert info["total_memory_bytes"] > 0
        assert re.fullmatch(r"\d+\.\d+", info["capability"])
    else:
        assert info["reason"]


def test_git_commit_is_none_or_a_forty_char_sha() -> None:
    from miniverl.utils.env import git_commit

    value = git_commit()
    assert value is None or re.fullmatch(r"[0-9a-f]{40}", value), value


def test_git_commit_reads_a_branch_ref(tmp_path: Path) -> None:
    from miniverl.utils.env import git_commit

    git_dir = tmp_path / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text(f"{_SHA}\n", encoding="utf-8")
    assert git_commit(tmp_path) == _SHA


def test_git_commit_reads_a_detached_head(tmp_path: Path) -> None:
    from miniverl.utils.env import git_commit

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(f"{_SHA}\n", encoding="utf-8")
    assert git_commit(tmp_path) == _SHA


def test_git_commit_falls_back_to_packed_refs(tmp_path: Path) -> None:
    from miniverl.utils.env import git_commit

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{'f' * 40} refs/heads/other\n"
        f"{_SHA} refs/heads/main\n",
        encoding="utf-8",
    )
    assert git_commit(tmp_path) == _SHA


def test_git_commit_follows_a_gitdir_pointer_file(tmp_path: Path) -> None:
    from miniverl.utils.env import git_commit

    real = tmp_path / "real-git"
    (real / "refs" / "heads").mkdir(parents=True)
    (real / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (real / "refs" / "heads" / "main").write_text(_SHA, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").write_text(f"gitdir: {real}\n", encoding="utf-8")
    assert git_commit(work) == _SHA


def test_git_commit_returns_none_for_an_unresolvable_ref(tmp_path: Path) -> None:
    from miniverl.utils.env import git_commit

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/never-committed\n", encoding="utf-8")
    assert git_commit(tmp_path) is None


# ------------------------------------------------------------------------- runs


class _Dumpable:
    """Stands in for the dataclasses that reach the JSON writers."""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "dumpable"}


class _RecordingWriter:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)


_RUN_PATH_PROPERTIES = (
    "config_original",
    "config_resolved",
    "manifest",
    "manifest_start",
    "environment",
    "metrics",
    "events",
    "trajectories",
    "eval_trajectories",
    "teacher_cache",
    "offline_dataset",
    "offline_dataset_manifest",
    "offline_dataset_trajectories",
    "checkpoints",
    "eval_json",
    "report_html",
    "summary_md",
    "benchmark_json",
)


def test_make_run_id_sanitizes_and_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    from miniverl.utils.runs import make_run_id

    run_id = make_run_id("Qwen Calc! v2/beta")
    assert re.fullmatch(r"\d{8}-\d{6}-\d{6}-[0-9a-f]{8}-qwen-calc-v2-beta", run_id), run_id
    assert make_run_id("keep.dots_and-dashes").endswith("-keep.dots_and-dashes")
    assert re.fullmatch(r"\d{8}-\d{6}-\d{6}-[0-9a-f]{8}-run", make_run_id("///"))


def test_generated_run_ids_do_not_collide_within_one_second() -> None:
    from miniverl.utils.runs import make_run_id

    ids = {make_run_id("same-run") for _ in range(32)}
    assert len(ids) == 32


def test_make_run_id_honours_an_explicit_id() -> None:
    from miniverl.utils.runs import make_run_id

    assert make_run_id("ignored", explicit="  My Run/2 ") == "My-Run-2"
    assert make_run_id("ignored", explicit="already-safe") == "already-safe"
    assert make_run_id("ignored", explicit="///") == "run"
    # An explicit id is never given a timestamp prefix.
    assert not re.match(r"\d{8}-\d{6}-", make_run_id("name", explicit="mine"))


def test_utc_now_is_iso_8601_utc_at_second_resolution() -> None:
    from miniverl.utils.runs import utc_now

    value = utc_now()
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    assert parsed.microsecond == 0
    assert value.endswith("+00:00")
    assert "T" in value


def test_jsonl_writer_appends_and_read_jsonl_round_trips(tmp_path: Path) -> None:
    from miniverl.utils.runs import JsonlWriter, read_jsonl

    path = tmp_path / "nested" / "metrics.jsonl"
    writer = JsonlWriter(path)
    assert writer.count == 0
    writer.write({"step": 1, "loss": 0.5})
    writer.write({"step": 2, "note": "答案"})
    assert writer.count == 2
    assert read_jsonl(path) == [{"step": 1, "loss": 0.5}, {"step": 2, "note": "答案"}]

    second = JsonlWriter(path)
    second.write({"step": 3})
    assert second.count == 1, "count is per writer instance, not per file"
    assert [record["step"] for record in read_jsonl(path)] == [1, 2, 3]


def test_jsonl_writer_serializes_unsupported_objects(tmp_path: Path) -> None:
    from miniverl.utils.runs import JsonlWriter, read_jsonl

    path = tmp_path / "events.jsonl"
    JsonlWriter(path).write({"obj": _Dumpable(), "path": tmp_path / "a" / "b"})
    record = read_jsonl(path)[0]
    assert record["obj"] == {"kind": "dumpable"}
    assert record["path"] == str(tmp_path / "a" / "b")


def test_read_jsonl_returns_empty_for_a_missing_file(tmp_path: Path) -> None:
    from miniverl.utils.runs import read_jsonl

    assert read_jsonl(tmp_path / "absent.jsonl") == []
    assert read_jsonl(tmp_path) == [], "a directory is not a readable log either"


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    from miniverl.utils.runs import read_jsonl

    path = tmp_path / "metrics.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_names_the_corrupt_line_number(tmp_path: Path) -> None:
    from miniverl.utils.runs import read_jsonl

    path = tmp_path / "metrics.jsonl"
    path.write_text('{"a": 1}\n\n{"a": oops\n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{path}:3")) as excinfo:
        read_jsonl(path)
    assert "is not valid JSON" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_write_json_and_read_json_round_trip(tmp_path: Path) -> None:
    from miniverl.utils.runs import read_json, write_json

    payload = {"beta": [1, 2, {"x": None}], "alpha": "值", "obj": _Dumpable()}
    path = write_json(tmp_path / "deep" / "manifest.json", payload)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.index('"alpha"') < text.index('"beta"'), "keys are sorted for stable diffs"
    loaded = read_json(path)
    assert loaded["beta"] == [1, 2, {"x": None}]
    assert loaded["alpha"] == "值"
    assert loaded["obj"] == {"kind": "dumpable"}


def test_read_json_raises_run_not_found_for_a_missing_file(tmp_path: Path) -> None:
    from miniverl.errors import RunNotFoundError
    from miniverl.utils.runs import read_json

    with pytest.raises(RunNotFoundError, match="expected JSON file not found"):
        read_json(tmp_path / "manifest.json")


def test_atomic_json_failure_preserves_the_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import miniverl.utils.runs as runs_module
    from miniverl.utils.runs import write_json, write_json_atomic

    target = write_json(tmp_path / "manifest.json", {"status": "running"})
    before = target.read_bytes()

    def fail_replace(_source, _target):
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(runs_module, "_replace_file", fail_replace)
    with pytest.raises(OSError, match="injected"):
        write_json_atomic(target, {"status": "completed"})

    assert target.read_bytes() == before
    assert not list(tmp_path.glob(".manifest.json.tmp-*"))


def test_run_paths_create_makes_the_cache_and_checkpoint_directories(tmp_path: Path) -> None:
    from miniverl.errors import MiniVerlError
    from miniverl.utils.runs import RunPaths

    paths = RunPaths.create(tmp_path / "runs", "20260101-000000-demo")
    assert paths.root == tmp_path / "runs" / "20260101-000000-demo"
    assert paths.root.is_dir()
    assert paths.teacher_cache.is_dir()
    assert paths.checkpoints.is_dir()
    marker = paths.root / "old-artifact.txt"
    marker.write_text("must survive a refused collision", encoding="utf-8")

    with pytest.raises(MiniVerlError, match="already exists") as excinfo:
        RunPaths.create(tmp_path / "runs", "20260101-000000-demo")
    assert "--resume" in str(excinfo.value)
    assert "--overwrite" in str(excinfo.value)
    assert marker.read_text(encoding="utf-8") == "must survive a refused collision"


def test_run_paths_explicit_overwrite_replaces_the_whole_run(tmp_path: Path) -> None:
    from miniverl.utils.runs import RunPaths

    first = RunPaths.create(tmp_path / "runs", "replace-me")
    (first.root / "events.jsonl").write_text('{"old": true}\n', encoding="utf-8")
    (first.checkpoints / "stale").mkdir()

    second = RunPaths.create(tmp_path / "runs", "replace-me", overwrite=True)

    assert second == first
    assert not second.events.exists()
    assert not (second.checkpoints / "stale").exists()
    assert second.teacher_cache.is_dir()
    assert second.checkpoints.is_dir()
    assert not list((tmp_path / "runs").glob(".replace-me.overwrite-*"))


def test_run_paths_concurrent_creation_has_one_winner(tmp_path: Path) -> None:
    from miniverl.errors import MiniVerlError
    from miniverl.utils.runs import RunPaths

    def create() -> str:
        try:
            return str(RunPaths.create(tmp_path / "runs", "contended").root)
        except MiniVerlError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: create(), range(2)))

    assert outcomes.count(str(tmp_path / "runs" / "contended")) == 1
    assert outcomes.count("refused") == 1


def test_every_run_path_stays_inside_the_root_and_is_unique(tmp_path: Path) -> None:
    from miniverl.utils.runs import RunPaths

    paths = RunPaths.create(tmp_path / "runs", "run-1")
    declared = {name for name, value in vars(RunPaths).items() if isinstance(value, property)}
    assert declared == set(_RUN_PATH_PROPERTIES)
    values = [getattr(paths, name) for name in _RUN_PATH_PROPERTIES]
    for name, value in zip(_RUN_PATH_PROPERTIES, values, strict=True):
        assert isinstance(value, Path), name
        assert value != paths.root, name
        assert value.is_relative_to(paths.root), f"{name} escapes the run root: {value}"
        assert ".." not in value.parts, name
    assert len(set(values)) == len(values), "two run artifacts share a path"


def test_run_paths_open_requires_a_directory_with_a_manifest(tmp_path: Path) -> None:
    from miniverl.errors import RunNotFoundError
    from miniverl.utils.runs import RunPaths, write_json

    with pytest.raises(RunNotFoundError, match="run directory not found"):
        RunPaths.open(tmp_path / "nope")

    root = tmp_path / "runs" / "run-2"
    root.mkdir(parents=True)
    with pytest.raises(RunNotFoundError, match=re.escape("no manifest.json")):
        RunPaths.open(root)

    write_json(root / "manifest.json", {"run_id": "run-2"})
    assert RunPaths.open(root).root == root


# -------------------------------------------------------------------------- gpu


def test_cuda_available_returns_a_bool() -> None:
    from miniverl.utils.gpu import cuda_available

    assert isinstance(cuda_available(), bool)


def test_snapshot_reports_zeros_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA is simulated absent so the contract is checked on any machine."""
    from miniverl.utils import gpu

    monkeypatch.setattr(gpu, "cuda_available", lambda: False)
    snap = gpu.snapshot()
    assert snap.available is False
    assert (
        snap.allocated_bytes
        == snap.reserved_bytes
        == snap.peak_allocated_bytes
        == snap.peak_reserved_bytes
        == snap.total_bytes
        == 0
    )
    assert snap.peak_allocated_gib == 0.0
    assert snap.peak_reserved_gib == 0.0
    assert gpu.free_vram_gib() == 0.0
    # The mutating helpers must be no-ops rather than raising.
    assert gpu.reset_peak_stats() is None
    assert gpu.empty_cache() is None


def test_real_snapshot_matches_the_machine() -> None:
    from miniverl.utils.gpu import cuda_available, snapshot

    snap = snapshot()
    assert snap.available == cuda_available()
    if snap.available:
        assert snap.total_bytes > 0
        assert snap.reserved_bytes >= snap.allocated_bytes
    else:
        assert snap.total_bytes == 0


def test_memory_snapshot_to_dict_has_the_documented_keys() -> None:
    from miniverl.utils.gpu import MemorySnapshot

    payload = MemorySnapshot(
        available=True,
        allocated_bytes=1,
        reserved_bytes=2,
        peak_allocated_bytes=3 * 1024**3,
        peak_reserved_bytes=1024**3 // 2,
        total_bytes=4,
    ).to_dict()
    assert set(payload) == {
        "cuda_available",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "total_bytes",
    }
    assert payload["cuda_available"] is True
    assert payload["peak_allocated_gib"] == 3.0
    assert payload["peak_reserved_gib"] == 0.5
    assert json.loads(json.dumps(payload)) == payload


def test_is_oom_error_matches_the_allocator_messages() -> None:
    from miniverl.utils.gpu import is_oom_error

    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")) is True
    assert is_oom_error(RuntimeError("CUDA error: out of memory")) is True
    assert is_oom_error(MemoryError("DefaultCPUAllocator: not enough memory")) is False
    assert is_oom_error(ValueError("nope")) is False
    assert is_oom_error(RuntimeError("shape mismatch")) is False


@pytest.mark.torch
@requires_torch
def test_is_oom_error_recognizes_the_torch_exception_class() -> None:
    """The class check matters: the message alone need not mention memory."""
    import torch

    from miniverl.utils.gpu import is_oom_error

    assert is_oom_error(torch.cuda.OutOfMemoryError("allocation failed")) is True


# ---------------------------------------------------------------------- logging


def test_get_logger_namespaces_every_name(miniverl_logger: logging.Logger) -> None:
    from miniverl.utils.logging import get_logger

    assert get_logger().name == "miniverl"
    assert get_logger("rollout").name == "miniverl.rollout"
    assert get_logger("miniverl.trainer").name == "miniverl.trainer"
    assert get_logger("rollout") is get_logger("rollout")


def test_configure_logging_installs_exactly_one_handler(miniverl_logger: logging.Logger) -> None:
    from miniverl.utils.logging import configure_logging, get_logger

    miniverl_logger.handlers.clear()
    configure_logging()
    assert len(miniverl_logger.handlers) == 1
    handler = miniverl_logger.handlers[0]
    configure_logging()
    configure_logging("DEBUG")
    get_logger("run")
    assert miniverl_logger.handlers == [handler], "configure_logging must not stack handlers"
    assert miniverl_logger.propagate is False, "records must not reach the root logger twice"


def test_configure_logging_reads_the_env_var(
    monkeypatch: pytest.MonkeyPatch, miniverl_logger: logging.Logger
) -> None:
    from miniverl.utils.logging import configure_logging

    monkeypatch.setenv("MINIVERL_LOG_LEVEL", "warning")
    miniverl_logger.handlers.clear()
    configure_logging()
    assert miniverl_logger.level == logging.WARNING


def test_an_explicit_level_is_applied(
    monkeypatch: pytest.MonkeyPatch, miniverl_logger: logging.Logger
) -> None:
    from miniverl.utils.logging import configure_logging

    monkeypatch.delenv("MINIVERL_LOG_LEVEL", raising=False)
    miniverl_logger.handlers.clear()
    configure_logging("DEBUG")
    assert miniverl_logger.level == logging.DEBUG


def test_an_explicit_level_survives_a_later_get_logger(
    monkeypatch: pytest.MonkeyPatch, miniverl_logger: logging.Logger
) -> None:
    from miniverl.utils.logging import configure_logging, get_logger

    monkeypatch.delenv("MINIVERL_LOG_LEVEL", raising=False)
    miniverl_logger.handlers.clear()
    configure_logging("DEBUG")
    get_logger("run")
    assert miniverl_logger.level == logging.DEBUG


def test_get_console_is_shared_and_switches_streams() -> None:
    from miniverl.utils.logging import get_console

    stdout_console = get_console()
    assert get_console() is stdout_console
    assert stdout_console.stderr is False
    stderr_console = get_console(stderr=True)
    assert stderr_console.stderr is True
    assert get_console(stderr=True) is stderr_console


def test_event_log_mirrors_the_payload_to_the_writer(miniverl_logger: logging.Logger) -> None:
    from miniverl.utils.logging import EventLog

    writer = _RecordingWriter()
    payload = EventLog(writer, logger_name="miniverl.test").emit(
        "train_start", step=3, cfg={"a": 1}
    )
    assert payload["event"] == "train_start"
    assert payload["step"] == 3
    assert payload["cfg"] == {"a": 1}
    assert datetime.fromisoformat(payload["ts"]).utcoffset() == timedelta(0)
    assert writer.records == [payload]


def test_event_log_without_a_writer_still_logs(miniverl_logger: logging.Logger) -> None:
    from miniverl.utils.logging import EventLog

    payload = EventLog().emit("run_end", status="ok")
    assert payload["event"] == "run_end"
    assert payload["status"] == "ok"


# ----------------------------------------------------------------------- doctor


def test_run_doctor_marks_required_dependencies_ok(tmp_path: Path) -> None:
    from miniverl import __version__
    from miniverl.doctor import run_doctor

    report = run_doctor(tmp_path / "runs")
    assert report.miniverl_version == __version__
    by_name = {check.name: check for check in report.checks}
    for module in ("typer", "rich", "pydantic", "yaml", "jinja2", "platformdirs", "safetensors"):
        check = by_name[f"dependency:{module}"]
        assert check.status == "ok", f"{check.name}: {check.detail}"
        assert check.hint is None
    assert report.can_run_core is True
    assert by_name["output directory"].status == "ok"
    assert (tmp_path / "runs").is_dir()
    assert not list((tmp_path / "runs").iterdir()), "the write probe must be cleaned up"


def test_run_doctor_report_is_json_serializable(tmp_path: Path) -> None:
    from miniverl.doctor import run_doctor

    payload = run_doctor(tmp_path / "runs").to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert set(payload) == {"miniverl_version", "checks", "capabilities", "verdict"}
    assert set(payload["verdict"]) == {
        "core_commands",
        "cpu_training",
        "gpu_training",
        "qlora_4bit",
    }
    for check in payload["checks"]:
        assert set(check) == {"name", "status", "detail", "hint"}
        assert check["status"] in {"ok", "warn", "missing", "fail"}
    assert isinstance(payload["capabilities"]["environments"], list)
    assert payload["capabilities"]["environments"]


def test_run_doctor_flags_an_unwritable_output_directory(tmp_path: Path) -> None:
    from miniverl.doctor import run_doctor

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("i am a file", encoding="utf-8")
    report = run_doctor(blocker / "runs")
    check = {c.name: c for c in report.checks}["output directory"]
    assert check.status == "fail", check.detail
    assert "not writable" in check.detail
    assert check.hint == "pass --output to a writable location"
    # can_run_core only weighs dependency checks, so it stays True here.
    assert report.can_run_core is True


def test_run_doctor_reports_optional_dependencies_as_missing_not_fail(tmp_path: Path) -> None:
    from miniverl.doctor import run_doctor

    report = run_doctor(tmp_path / "runs")
    optional = [c for c in report.checks if c.name.startswith("optional:")]
    assert {c.name for c in optional} >= {
        "optional:torch",
        "optional:transformers",
        "optional:peft",
        "optional:accelerate",
        "optional:numpy",
        "optional:bitsandbytes",
    }
    for check in optional:
        assert check.status in {"ok", "missing"}, check.name
        module = check.name.split(":", 1)[1]
        if check.status == "missing":
            assert check.hint is not None
            assert check.hint.startswith('pip install "miniverl[')
            assert report.capabilities[module] is None
        else:
            assert check.hint is None
            assert report.capabilities[module]


def test_doctor_verdicts_derive_from_the_capabilities() -> None:
    from miniverl.doctor import Check, DoctorReport

    caps = {"torch": "2.3.0", "cuda_available": True, "bitsandbytes": "0.43.0", "peft": "0.12.0"}
    report = DoctorReport("0.1.0", capabilities=dict(caps))
    assert report.can_train_cpu is True
    assert report.can_train_gpu is True
    assert report.can_qlora is True
    for missing in ("cuda_available", "bitsandbytes", "peft"):
        degraded = dict(caps)
        degraded[missing] = None
        assert DoctorReport("0.1.0", capabilities=degraded).can_qlora is False
    assert DoctorReport("0.1.0", capabilities={}).can_train_cpu is False

    checks = [Check("dependency:yaml", "fail", "not installed"), Check("platform", "ok", "x")]
    assert DoctorReport("0.1.0", checks=checks).can_run_core is False
    ok_checks = [Check("dependency:yaml", "ok", "6.0"), Check("output directory", "fail", "x")]
    assert DoctorReport("0.1.0", checks=ok_checks).can_run_core is True
