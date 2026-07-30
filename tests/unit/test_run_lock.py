"""Real cross-process ownership tests for mutable run directories."""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pytest

from miniverl.errors import RunLockedError
from miniverl.utils.locking import RunLock


def _hold_lock(root: str, run_id: str, ready, release) -> None:  # type: ignore[no-untyped-def]
    with RunLock(Path(root), run_id):
        ready.set()
        release.wait(30)


def _acquire_once(root: str, run_id: str, ready, release) -> None:  # type: ignore[no-untyped-def]
    with RunLock(Path(root), run_id):
        ready.set()
        release.wait(30)


def _spawn_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context("spawn")


def test_same_run_fails_fast_and_changes_no_run_artifact(tmp_path) -> None:
    run = tmp_path / "same-run"
    run.mkdir()
    sentinel = run / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    ctx = _spawn_context()
    ready = ctx.Event()
    release = ctx.Event()
    owner = ctx.Process(target=_hold_lock, args=(str(tmp_path), run.name, ready, release))
    owner.start()
    assert ready.wait(15)
    before = sentinel.read_bytes()

    with pytest.raises(RunLockedError, match=r"same-run.*\.miniverl-locks"):
        RunLock(tmp_path, run.name).acquire()

    assert sentinel.read_bytes() == before
    release.set()
    owner.join(15)
    assert owner.exitcode == 0


def test_run_lock_timeout_is_bounded(tmp_path) -> None:
    ctx = _spawn_context()
    ready = ctx.Event()
    release = ctx.Event()
    owner = ctx.Process(target=_hold_lock, args=(str(tmp_path), "timeout", ready, release))
    owner.start()
    assert ready.wait(15)
    started = time.perf_counter()
    with pytest.raises(RunLockedError):
        RunLock(tmp_path, "timeout", timeout=0.2).acquire()
    elapsed = time.perf_counter() - started
    assert 0.15 <= elapsed < 2.0
    release.set()
    owner.join(15)
    assert owner.exitcode == 0


def test_process_termination_releases_the_actual_lock(tmp_path) -> None:
    ctx = _spawn_context()
    ready = ctx.Event()
    release = ctx.Event()
    owner = ctx.Process(target=_hold_lock, args=(str(tmp_path), "terminated", ready, release))
    owner.start()
    assert ready.wait(15)
    owner.terminate()
    owner.join(15)
    assert owner.exitcode is not None

    with RunLock(tmp_path, "terminated"):
        pass


def test_different_run_ids_can_be_owned_concurrently(tmp_path) -> None:
    ctx = _spawn_context()
    ready_a = ctx.Event()
    ready_b = ctx.Event()
    release = ctx.Event()
    process_a = ctx.Process(
        target=_acquire_once,
        args=(str(tmp_path), "run-a", ready_a, release),
    )
    process_b = ctx.Process(
        target=_acquire_once,
        args=(str(tmp_path), "run-b", ready_b, release),
    )
    process_a.start()
    process_b.start()
    assert ready_a.wait(15)
    assert ready_b.wait(15)
    release.set()
    process_a.join(15)
    process_b.join(15)
    assert process_a.exitcode == 0
    assert process_b.exitcode == 0
