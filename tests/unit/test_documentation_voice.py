from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_product_entry_points_lead_with_workflows() -> None:
    entry_points = "\n".join(
        _read(path)
        for path in (
            "README.md",
            "README.zh-CN.md",
            "docs/index.md",
            "docs/comparisons.md",
        )
    )
    for old_framing in (
        "For most serious distillation work it is",
        "What miniVERL is not",
        "reasons not to trust miniVERL",
        "full verl compatibility are not claimed",
        "不执行任意 verl YAML",
    ):
        assert old_framing not in entry_points

    assert "## What a run gives you" in entry_points
    assert "## Choose your path" in entry_points
    assert "## 一次运行会得到什么" in entry_points
    assert "## 选择你的路径" in entry_points


def test_complete_boundaries_have_canonical_destinations() -> None:
    for path in ("README.md", "README.zh-CN.md", "docs/index.md", "docs/comparisons.md"):
        text = _read(path)
        assert "limitations" in text
        assert "compatibility" in text or "兼容" in text

    limitations = _read("docs/limitations.md")
    assert "one machine" in limitations
    assert "multi-GPU" in limitations
    assert "PPO/GRPO" in limitations
    assert "## Evidence scope" in limitations
    assert "### Execution boundary" in limitations


def test_main_runtime_visual_describes_the_product_flow() -> None:
    visuals = _read("docs/verl-local-runtime.svg") + _read("docs/verl-local-runtime-mobile.svg")
    assert "READINESS REPORT" in visuals
    assert "typed · inspectable · resumable" in visuals
    assert "OUTSIDE miniVERL" not in visuals
    assert "fail-closed" not in visuals
