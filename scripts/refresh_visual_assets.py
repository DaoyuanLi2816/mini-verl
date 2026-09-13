"""Refresh only SVG projections of frozen evidence, never result JSON or JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import publish_alignment_external_artifacts as external  # noqa: E402
from scripts import publish_alignment_lab_artifacts as alignment  # noqa: E402
from scripts import publish_benchmark_artifacts as calculator  # noqa: E402
from scripts import publish_consumer_runtime_artifacts as consumer  # noqa: E402
from scripts import publish_recoverybench_artifacts as recovery  # noqa: E402
from scripts import publish_verl_bridge_diagrams as bridge  # noqa: E402
from scripts import publish_verl_opd_reference_artifacts as opd  # noqa: E402


def generated_assets() -> dict[Path, str]:
    def payload(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    results = ROOT / "benchmarks/results"
    calc_path = results / "gpu-calc-hard-equal-update-v2.json"
    recovery_path = results / "recoverybench-v1-equal-updates.json"
    calc_result = calculator.BenchmarkResult.model_validate(payload(calc_path))
    recovery_result = recovery.BenchmarkResult.model_validate(payload(recovery_path))
    assets = {
        ROOT / "docs/gpu-calc-hard-equal-update-v2.svg": calculator.render_svg(
            calc_result, digest(calc_path)
        ),
        consumer.OUTPUT: consumer.render_pareto(payload(consumer.RESULT), digest(consumer.RESULT)),
        consumer.MOBILE_OUTPUT: consumer.render_pareto_mobile(
            payload(consumer.RESULT), digest(consumer.RESULT)
        ),
        opd.FIGURE: opd.render(opd._load(opd.RESULT)),
        opd.MOBILE_FIGURE: opd.render_mobile(opd._load(opd.RESULT)),
        ROOT / "docs/verl-bridge-architecture.svg": bridge._desktop(),
        ROOT / "docs/verl-bridge-architecture-mobile.svg": bridge._mobile(),
    }
    assets.update(
        {
            alignment.DOCS / name: content
            for name, content in alignment.render_figures(
                payload(alignment.RESULT), digest(alignment.RESULT)
            ).items()
        }
    )
    assets.update(
        {
            recovery.DOCS / name: content
            for name, content in recovery.render_figures(
                recovery_result, payload(recovery.ANALYSIS), digest(recovery_path)
            ).items()
        }
    )
    for mobile in (False, True):
        suffix = "-mobile" if mobile else ""
        assets[external.DOCS / f"checkpoint-gate-matrix{suffix}.svg"] = external.render_gate_matrix(
            payload(external.RESULT), mobile=mobile
        )
        assets[external.DOCS / f"study-early-stop{suffix}.svg"] = external.render_flow(
            payload(external.RESULT), mobile=mobile
        )
    return assets


def overlay_visual_assets(stable_source: Path, site: Path, *, root: Path = ROOT) -> None:
    """Apply presentation fixes to built stable docs, keeping the release tree intact.

    Every stable scientific input must still exist byte-for-byte on main. This
    prevents a new result from silently replacing an older release's figure.
    The overlay copies only existing SVGs, never Markdown, JSON or runtime code.
    """
    for directory in (stable_source / "benchmarks/results", stable_source / "docs", site):
        if not directory.is_dir():
            raise ValueError(f"stable visual overlay refused: missing directory: {directory}")
    for folder in ("benchmarks/results", "benchmarks/evidence"):
        for frozen in (stable_source / folder).rglob("*"):
            if not frozen.is_file() or frozen.suffix not in {".json", ".jsonl"}:
                continue
            current = root / frozen.relative_to(stable_source)
            if not current.is_file() or frozen.read_bytes() != current.read_bytes():
                raise ValueError(f"stable visual overlay refused: evidence differs: {current}")
    for asset in (root / "docs").rglob("*.svg"):
        relative = asset.relative_to(root / "docs")
        destination = site / relative
        if destination.is_file() and (stable_source / "docs" / relative).is_file():
            shutil.copyfile(asset, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--overlay-site", type=Path)
    parser.add_argument("--stable-source", type=Path)
    args = parser.parse_args()
    for path, content in generated_assets().items():
        if args.check or args.overlay_site:
            if path.read_bytes() != content.encode("utf-8"):
                raise SystemExit(f"stale generated visual: {path.relative_to(ROOT)}")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    if args.overlay_site:
        if not args.stable_source:
            parser.error("--overlay-site requires --stable-source")
        overlay_visual_assets(args.stable_source, args.overlay_site)
    print("frozen-evidence SVG projections agree")


if __name__ == "__main__":
    main()
