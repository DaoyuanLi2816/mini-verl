"""Recovery never disguises corruption or loses the uncommitted diagnostic tail."""

import pytest

from miniverl.errors import CheckpointError
from miniverl.training.journal import capture_logs, restore_logs
from miniverl.utils.runs import write_text


def test_restore_logs_archives_tails_and_is_idempotent(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_text(path, '{"step":1}\n')
    cursors = capture_logs(tmp_path)
    write_text(path, '{"step":1}\n{"step":2}\n')
    restore_logs(tmp_path, cursors)
    restore_logs(tmp_path, cursors)
    assert path.read_text() == '{"step":1}\n'
    assert [p.read_text() for p in (tmp_path / "recovery-tails").glob("*.jsonl")] == [
        '{"step":2}\n'
    ]


def test_corrupt_prefix_prevents_any_log_replacement(tmp_path):
    first = tmp_path / "metrics.jsonl"
    last = tmp_path / "advantages.jsonl"
    write_text(first, "{}\n")
    write_text(last, "{}\n")
    cursors = capture_logs(tmp_path)
    write_text(first, "{}\n{}\n")
    write_text(last, "[]\n")
    with pytest.raises(CheckpointError, match="prefix mismatch"):
        restore_logs(tmp_path, cursors)
    assert first.read_text() == "{}\n{}\n"


def test_partial_atomic_restore_can_be_retried(tmp_path, monkeypatch):
    from miniverl.training import journal

    cursors = capture_logs(tmp_path)
    for name in ("metrics.jsonl", "rewards.jsonl"):
        write_text(tmp_path / name, "{}\n")
    original = journal._atomic_bytes

    def fail(path, content):
        if path.name == "rewards.jsonl":
            raise OSError("injected publication failure")
        original(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(journal, "_atomic_bytes", fail)
        with pytest.raises(OSError, match="injected"):
            restore_logs(tmp_path, cursors)
    restore_logs(tmp_path, cursors)
    assert (tmp_path / "metrics.jsonl").read_bytes() == b""
    assert (tmp_path / "rewards.jsonl").read_bytes() == b""
    assert len(list((tmp_path / "recovery-tails").glob("*.jsonl"))) == 2
