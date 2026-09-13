from pathlib import Path

import pytest

from scripts.refresh_visual_assets import generated_assets, overlay_visual_assets


def test_all_frozen_visual_projections_match_committed_bytes():
    for path, rendered in generated_assets().items():
        assert path.read_bytes() == rendered.encode("utf-8"), path


def test_stable_overlay_changes_only_svg_after_evidence_verification(tmp_path: Path):
    root, stable, site = (tmp_path / name for name in ("main", "tag", "site"))
    for tree in (root, stable):
        (tree / "benchmarks/results").mkdir(parents=True)
        (tree / "benchmarks/results/result.json").write_text('{"measured":0}\n')
        (tree / "docs").mkdir()
        (tree / "docs/figure.svg").write_text("new" if tree == root else "old")
    site.mkdir()
    (site / "figure.svg").write_text("old")
    (site / "index.html").write_text("tag-pinned prose")
    overlay_visual_assets(stable, site, root=root)
    assert (site / "figure.svg").read_text() == "new"
    assert (site / "index.html").read_text() == "tag-pinned prose"
    assert (stable / "docs/figure.svg").read_text() == "old"

    (site / "figure.svg").write_text("old")
    (root / "benchmarks/results/result.json").write_text('{"measured":1}\n')
    with pytest.raises(ValueError, match="evidence differs"):
        overlay_visual_assets(stable, site, root=root)
    assert (site / "figure.svg").read_text() == "old"


def test_readme_gallery_covers_both_banner_and_runtime():
    from scripts.check_docs_visual import readme_figure_preview

    root = Path(__file__).resolve().parents[2]
    for language in ("README.md", "README.zh-CN.md"):
        preview = readme_figure_preview(root, "http://localhost:1234", language)
        for asset in (
            "banner.svg",
            "banner-mobile.svg",
            "verl-local-runtime.svg",
            "verl-local-runtime-mobile.svg",
        ):
            assert asset in preview
        assert "raw.githubusercontent.com" not in preview


def test_stable_overlay_requires_existing_source_and_site(tmp_path: Path):
    with pytest.raises(ValueError, match="missing directory"):
        overlay_visual_assets(tmp_path / "missing-tag", tmp_path / "missing-site")
