"""The repository's tracked text stays UTF-8 clean.

`docs/release-checklist.md` shipped `鈥?not applicable` and `CHANGELOG.md`
shipped `base 鈫?SFT checkpoint` -- UTF-8 punctuation written back through a
GBK console. The bytes are valid UTF-8 afterwards, so nothing raised.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_text_integrity import check_text_integrity

ROOT = Path(__file__).resolve().parents[2]


def test_the_repository_is_clean() -> None:
    assert check_text_integrity(ROOT) == []


def _repository(tmp_path: Path, name: str, content: bytes) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        (
            "checklist.md",
            "- `鈥?not applicable` rather than zero.\n".encode(),
            "mis-decoded",
        ),
        (
            "changelog.md",
            "base 鈫?SFT checkpoint\n".encode(),
            "mis-decoded",
        ),
        ("bom.md", b"\xef\xbb\xbf# Title\n", "byte-order mark"),
        ("broken.md", "text � here\n".encode(), "replacement character"),
    ],
)
def test_damage_is_reported(tmp_path: Path, name: str, content: bytes, expected: str) -> None:
    root = _repository(tmp_path, name, content)

    problems = check_text_integrity(root)

    assert len(problems) == 1
    assert expected in problems[0]
    assert name in problems[0]


def test_intentional_cjk_is_not_flagged(tmp_path: Path) -> None:
    """The Chinese README and the Unicode test fixtures must stay allowed."""
    root = _repository(
        tmp_path,
        "README.zh-CN.md",
        "# 中文文档\n\n这是一个单卡运行时。\n".encode(),
    )

    assert check_text_integrity(root) == []


def test_binary_files_are_not_read_as_text(tmp_path: Path) -> None:
    root = _repository(tmp_path, "model.safetensors", b"\x00\xff\xfe binary \x93")

    assert check_text_integrity(root) == []
