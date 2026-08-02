#!/usr/bin/env python3
"""Build the data-bound RecoveryBench v1 technical report."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
VENV_SITE_PACKAGES = ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.is_dir():
    # The PDF runtime supplies ReportLab; the project environment supplies the
    # exact validation dependencies used by miniVERL.
    sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(ROOT / "src"))

from miniverl.evaluation.schema import BenchmarkResult  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"
OUTPUT = Path(__file__).with_name("recoverybench-v1.pdf")
EXPECTED_HASHES = {
    "recoverybench-v1-equal-updates.json": (
        "6ce2e6837e12b99ebc4fad6d27ce3e69c92e295ff3b9b60e0f68c2d308022384"
    ),
    "recoverybench-v1-equal-selected-tokens.json": (
        "fe4c9afc799724dfe7a32e631676a1e5177c44559a7374d2ea31da135354f137"
    ),
    "recoverybench-v1-equal-wall-time.json": (
        "425b0fa568f37b09e61af731d3da5009bd3833bddde6efaf2c66e9dba8355cbe"
    ),
    "recoverybench-v1-analysis.json": (
        "c0e7b8c9e8da9a0d0a5d64a17a688c45e3dbbd1c3b68074249b31fc10f0baeca"
    ),
}

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#4b5563")
CYAN = colors.HexColor("#0891b2")
VIOLET = colors.HexColor("#7c3aed")
LIGHT = colors.HexColor("#f1f5f9")
RED = colors.HexColor("#b91c1c")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> tuple[dict[str, BenchmarkResult], dict[str, Any]]:
    loaded: dict[str, BenchmarkResult] = {}
    for name in (
        "recoverybench-v1-equal-updates.json",
        "recoverybench-v1-equal-selected-tokens.json",
        "recoverybench-v1-equal-wall-time.json",
    ):
        path = RESULTS / name
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"unexpected RecoveryBench source hash for {name}: {actual}")
        result = BenchmarkResult.model_validate_json(path.read_text(encoding="utf-8"))
        if result.schema_version != 3 or result.invalidation_status != {
            "valid": True,
            "reasons": [],
        }:
            raise ValueError(f"{name} is not a valid schema-v3 result")
        loaded[str(result.budget_view)] = result
    analysis_path = RESULTS / "recoverybench-v1-analysis.json"
    if _sha256(analysis_path) != EXPECTED_HASHES[analysis_path.name]:
        raise ValueError("unexpected paired-analysis hash")
    return loaded, json.loads(analysis_path.read_text(encoding="utf-8"))


def _means(result: BenchmarkResult, arm_name: str) -> dict[str, float]:
    arms = result.by_arm()[arm_name]
    return {
        "strict": mean(float(arm.strict_task_success_rate or 0) for arm in arms),
        "recovery": mean(
            float((arm.recovery_metrics or {}).get("recovery_after_error_rate") or 0)
            for arm in arms
        ),
        "train": mean(float(arm.train_seconds or 0) for arm in arms),
        "wall": mean(float(arm.wall_seconds) for arm in arms),
        "vram": mean(float(arm.peak_reserved_bytes or 0) / 2**30 for arm in arms),
    }


class MetricBars(Flowable):
    """Compact two-metric horizontal bar chart."""

    def __init__(self, rows: list[tuple[str, float, float]], width: float = 6.8 * inch):
        super().__init__()
        self.rows = rows
        self.width = width
        self.height = 0.45 * inch * len(rows) + 0.22 * inch

    def draw(self) -> None:
        canvas = self.canv
        label_width = 1.75 * inch
        chart_width = self.width - label_width - 0.62 * inch
        maximum = max(max(first, second) for _, first, second in self.rows) or 1
        canvas.setFont("Helvetica", 7.4)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(self.width, self.height - 5, "strict / recovery")
        for index, (label, first, second) in enumerate(self.rows):
            y = self.height - 21 - index * 0.45 * inch
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica", 8)
            canvas.drawString(0, y + 4, label)
            canvas.setFillColor(CYAN)
            canvas.roundRect(
                label_width,
                y + 8,
                chart_width * first / maximum,
                6,
                3,
                fill=1,
                stroke=0,
            )
            canvas.setFillColor(VIOLET)
            canvas.roundRect(
                label_width,
                y - 1,
                chart_width * second / maximum,
                6,
                3,
                fill=1,
                stroke=0,
            )
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica-Bold", 7.5)
            canvas.drawRightString(self.width, y + 2, f"{first:.3f} / {second:.3f}")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=16,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            textColor=MUTED,
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=INK,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.2,
            textColor=INK,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=10,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=16,
            textColor=RED,
            borderColor=colors.HexColor("#fecaca"),
            borderWidth=0.8,
            borderPadding=9,
            backColor=colors.HexColor("#fef2f2"),
            spaceAfter=12,
        ),
        "center": ParagraphStyle(
            "Center",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.6,
            leading=8,
            textColor=INK,
        ),
        "hash": ParagraphStyle(
            "Hash",
            parent=base["Code"],
            fontName="Courier",
            fontSize=5.2,
            leading=6.5,
            textColor=INK,
        ),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _table(
    rows: list[list[Any]],
    widths: list[float],
    *,
    header: bool = True,
) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.6),
            ]
        )
    for row in range(1 if header else 0, len(rows)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT))
    table.setStyle(TableStyle(commands))
    return table


def _page(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(doc.leftMargin, 0.52 * inch, letter[0] - doc.rightMargin, 0.52 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(doc.leftMargin, 0.34 * inch, "miniVERL RecoveryBench v1")
    canvas.drawRightString(
        letter[0] - doc.rightMargin,
        0.34 * inch,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def _invariant_canvas(*args: Any, **kwargs: Any) -> Canvas:
    kwargs["invariant"] = 1
    return Canvas(*args, **kwargs)


def build(output: Path = OUTPUT) -> None:
    results, analysis = _load()
    primary = results["equal_optimizer_updates"]
    selected = results["equal_selected_training_tokens"]
    wall = results["equal_gpu_wall_time"]
    styles = _styles()
    story: list[Flowable] = []

    story.extend(
        [
            Spacer(1, 0.55 * inch),
            _p(
                "Fresh-State Distillation for Tool-Error Recovery on a Consumer GPU",
                styles["title"],
            ),
            _p(
                "RecoveryBench v1 technical report - miniVERL v0.3.0 - Daoyuan Li",
                styles["subtitle"],
            ),
            _p(
                "Primary finding: under eight equal continuation updates, strict fresh-state OPD "
                "underperformed frozen-student-state KD on both preregistered primary endpoints. "
                "Fresh-state supervision also required about 13.2x the continuation time.",
                styles["callout"],
            ),
            _p("Abstract", styles["h1"]),
            _p(
                "RecoveryBench isolates whether teacher supervision on states freshly visited by "
                "the current student improves recovery from structured SQLite tool errors, "
                "relative to distillation on a fixed state set collected from the cold-start "
                "student. We evaluate six equal-update arms and three-method selected-position "
                "and wall-time views over three prespecified seeds and 128 paired test tasks. "
                "A protocol-qualified Qwen3-1.7B teacher is selected on eval only and frozen at "
                "an immutable adapter revision. Fresh OPD does not improve the primary result: "
                "its strict-success mean is 10.9%, versus 23.2% for frozen KD, with a paired "
                "difference of -12.24 points. Recovery after error similarly falls by 13.79 "
                "points. The result is a scoped negative mechanism study, not a claim that OPD "
                "is universally ineffective or interchangeable with SFT.",
                styles["body"],
            ),
            Spacer(1, 0.08 * inch),
            _p(
                "Keywords: on-policy distillation, tool use, recovery, frozen states, consumer GPU",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("1. Question and hypotheses", styles["h1"]),
            _p(
                "The primary question is whether supervision on fresh states visited by the "
                "current student improves tool-error recovery over teacher scoring on states "
                "frozen from the shared cold-start student. H1 predicts a recovery advantage for "
                "strict fresh OPD under equal updates. H2 asks whether querying at most 50% of "
                "model-generated positions preserves most of any full-OPD gain. H3 requires "
                "quality to be interpreted with teacher queries, GPU time and peak VRAM.",
                styles["body"],
            ),
            _p(
                "RecoveryBench is not an alignment benchmark. SFT establishes basic capability "
                "and protocol competence; OPD is a teacher-student mechanism whose transferred "
                "behavior depends on the teacher. Ordinary task success is not a complete "
                "alignment objective.",
                styles["body"],
            ),
            _p("2. Environment and controlled intervention", styles["h1"]),
            _p(
                "The new sqlite_recovery environment retains in-memory, read-only SQLite safety "
                "constraints and uses 12 deterministic structural templates split disjointly "
                "across train, eval and test. Tasks cover schema inspection, filtering, joins, "
                "aggregation and correction. One subset receives a deterministic one-time "
                "SCHEMA_REFRESH_REQUIRED error; another exposes natural SQL errors; a third has "
                "no intervention. Structured error_code, retryable and intervention fields make "
                "recovery metrics independent of fragile string matching.",
                styles["body"],
            ),
            _p(
                "All methods use the same 256-task training schedule and the same 128 test task "
                "IDs within each seed. Oracle intervention traces execute the failed query, "
                "inspect the schema, retry correctly and emit a final answer; SFT therefore "
                "trains a real recovery sequence rather than bypassing the error.",
                styles["body"],
            ),
            _p("3. Methods", styles["h1"]),
            _table(
                [
                    ["Method", "State source", "Supervision", "Freshness"],
                    ["continued SFT", "oracle", "hard oracle tokens", "fixed demonstration"],
                    ["oracle offline KD", "oracle", "teacher top-k + tail", "fixed oracle"],
                    ["frozen-student KD", "cold student", "teacher top-k + tail", "fixed once"],
                    ["strict fresh OPD", "current student", "teacher top-k + tail", "each update"],
                    [
                        "fresh OPD 50%",
                        "current student",
                        "teacher on selected positions",
                        "each update",
                    ],
                ],
                [1.18 * inch, 1.18 * inch, 1.65 * inch, 1.45 * inch],
            ),
            Spacer(1, 0.12 * inch),
            _p(
                "Each seed starts from one shared 24-update SFT checkpoint. Frozen-student KD "
                "collects 64 cold-policy trajectories once at parameter version zero, scores "
                "them with the qualified teacher, and reuses the immutable dataset for all "
                "updates. Strict OPD recollects after every parameter update. Audit logs verify "
                "that rollout policy, current policy and current parameter versions match at all "
                "75 fresh-OPD updates across the three result views.",
                styles["body"],
            ),
        ]
    )

    story.extend(
        [
            PageBreak(),
            _p("4. Teacher qualification", styles["h1"]),
            _p(
                "The historical calculator protocol teacher was evaluated first on the 96-task "
                "RecoveryBench eval split and failed. Candidate A was trained by QLoRA on "
                "protocol-v2 oracle traces, reloaded on its preregistered NF4 base and became the "
                "first candidate to pass every gate. Candidate selection never used the test "
                "split.",
                styles["body"],
            ),
            _table(
                [
                    ["Candidate", "Strict", "Recovery", "Parse-valid", "Tool success", "Gate"],
                    ["calculator protocol teacher", "25.0%", "10.7%", "95.8%", "14.4%", "fail"],
                    ["SQLite candidate A (NF4)", "90.6%", "81.2%", "100.0%", "87.1%", "pass"],
                ],
                [1.65 * inch, 0.65 * inch, 0.72 * inch, 0.82 * inch, 0.82 * inch, 0.55 * inch],
            ),
            Spacer(1, 0.1 * inch),
            _p(
                "Selected adapter: DaoyuanLi/mini-verl-qwen3-1.7b-sqlite-recovery-teacher at "
                "eb2747895ec32dab47c5b50c2d8aa9c0d9701e0d; weights SHA-256 "
                "5355f7007efb904d1b45a1aeb9b73b479b6f52025ab92502ab7895706155b2ba.",
                styles["small"],
            ),
            _p("5. Budget matching", styles["h1"]),
            _p(
                "The primary view fixes eight continuation optimizer updates. The selected-token "
                "view stops at the first optimizer boundary at or beyond 6,224 selected positions "
                "and records overshoot. Every core method stopped after eight updates; overshoot "
                "was 0-646 positions.",
                styles["body"],
            ),
            _p(
                "The preregistered wall target is 50 continuation seconds. Fresh OPD crosses it "
                "in one indivisible step. SFT and frozen KD complete the eight-cycle ceiling "
                "before their internal continuation timer crosses 50 seconds, although complete "
                "train calls average 51.92 and 51.63 seconds because final checkpoint persistence "
                "is included there. The resulting artifact is a cycle-capped wall diagnostic, "
                "not exact equal-time evidence. It is preserved without a post-outcome rerun.",
                styles["callout"],
            ),
            _p(
                "Teacher preparation is excluded from arm continuation time and reported "
                "separately: 2,310.75 seconds once; 462.15 seconds amortized over five students; "
                "231.07 seconds over ten. An excluded 687.05-second diagnostic remains recorded.",
                styles["body"],
            ),
        ]
    )

    order = [arm.name for arm in primary.arms if arm.seed == primary.seeds[0]]
    labels = {
        "cold-start-only": "cold start",
        "continued-sft": "continued SFT",
        "offline-kd-oracle": "oracle offline KD",
        "offline-kd-frozen-student": "frozen-student KD",
        "strict-opd-fresh": "strict fresh OPD",
        "strict-opd-fresh-budget50": "fresh OPD 50%",
    }
    rows = []
    for name in order:
        metrics = _means(primary, name)
        arms = sorted(primary.by_arm()[name], key=lambda arm: arm.seed)
        seed_values = ", ".join(
            f"{100 * float(arm.strict_task_success_rate or 0):.1f}" for arm in arms
        )
        rows.append(
            [
                labels[name],
                seed_values,
                f"{100 * metrics['strict']:.1f}%",
                f"{100 * metrics['recovery']:.1f}%",
                f"{metrics['train']:.1f}",
                f"{metrics['vram']:.2f}",
            ]
        )

    story.extend(
        [
            PageBreak(),
            _p("6. Equal-update results", styles["h1"]),
            _table(
                [
                    [
                        "Method",
                        "Strict by seed (%)",
                        "Strict mean",
                        "Recovery",
                        "Train s",
                        "VRAM GiB",
                    ],
                    *rows,
                ],
                [1.35 * inch, 1.4 * inch, 0.72 * inch, 0.72 * inch, 0.6 * inch, 0.68 * inch],
            ),
            Spacer(1, 0.16 * inch),
            MetricBars(
                [
                    (
                        labels[name],
                        _means(primary, name)["strict"],
                        _means(primary, name)["recovery"],
                    )
                    for name in order
                ]
            ),
            Spacer(1, 0.08 * inch),
            _p(
                "Cyan is strict task success; violet is recovery after a structured tool error. "
                "Values are three-seed means. Seed order is 1234, 20260727, 20260801.",
                styles["center"],
            ),
            _p("Paired task analysis", styles["h2"]),
            _table(
                [
                    ["Endpoint", "Fresh - frozen", "95% paired bootstrap", "Pairs"],
                    [
                        "strict task success",
                        f"{100 * analysis['strict_task_success']['mean_difference']:.2f} pp",
                        f"[{100 * analysis['strict_task_success']['lower_95']:.2f}, "
                        f"{100 * analysis['strict_task_success']['upper_95']:.2f}]",
                        str(analysis["strict_task_success"]["pairs"]),
                    ],
                    [
                        "recovery after error",
                        f"{100 * analysis['recovery_after_error']['mean_difference']:.2f} pp",
                        f"[{100 * analysis['recovery_after_error']['lower_95']:.2f}, "
                        f"{100 * analysis['recovery_after_error']['upper_95']:.2f}]",
                        str(analysis["recovery_after_error"]["pairs"]),
                    ],
                ],
                [1.55 * inch, 1.05 * inch, 1.55 * inch, 0.6 * inch],
            ),
            Spacer(1, 0.1 * inch),
            _p(
                "H1 is not supported. Frozen-student KD has higher strict success and recovery "
                "than strict fresh OPD. Oracle-state offline KD has the highest mean. The 50% "
                "fresh arm reaches a 27.3% strict mean but ranges from 10.9% to 49.2%, so it does "
                "not establish a stable query-budget benefit.",
                styles["body"],
            ),
        ]
    )

    selected_rows = []
    for name in ("continued-sft", "offline-kd-frozen-student", "strict-opd-fresh"):
        arms = sorted(selected.by_arm()[name], key=lambda arm: arm.seed)
        selected_rows.append(
            [
                labels[name],
                ", ".join(str(arm.stop_criterion["actual"]) for arm in arms),
                f"{100 * _means(selected, name)['strict']:.1f}%",
                f"{100 * _means(selected, name)['recovery']:.1f}%",
                f"{_means(selected, name)['train']:.1f}",
            ]
        )
    wall_rows = []
    for name in ("continued-sft", "offline-kd-frozen-student", "strict-opd-fresh"):
        arms = sorted(wall.by_arm()[name], key=lambda arm: arm.seed)
        stop_kinds = [str(arm.stop_criterion["kind"]) for arm in arms]
        if len(set(stop_kinds)) != 1:
            raise ValueError(f"mixed wall-time stop kinds for {name}: {stop_kinds}")
        stop_label = {
            "configured_cycles": "cycle cap (3/3)",
            "wall_seconds": "wall target (3/3)",
        }.get(stop_kinds[0], f"{stop_kinds[0]} (3/3)")
        wall_rows.append(
            [
                labels[name],
                ", ".join(str(arm.optimizer_steps) for arm in arms),
                stop_label,
                f"{100 * _means(wall, name)['strict']:.1f}%",
                f"{_means(wall, name)['train']:.1f}",
            ]
        )
    story.extend(
        [
            PageBreak(),
            _p("7. Secondary budget views", styles["h1"]),
            _p("Equal selected positions (target 6,224)", styles["h2"]),
            _table(
                [["Method", "Actual by seed", "Strict", "Recovery", "Train s"], *selected_rows],
                [1.45 * inch, 1.45 * inch, 0.65 * inch, 0.68 * inch, 0.65 * inch],
            ),
            Spacer(1, 0.14 * inch),
            _p("Wall-time target diagnostic", styles["h2"]),
            _table(
                [
                    ["Method", "Steps by seed", "Recorded stop kind", "Strict", "Train s"],
                    *wall_rows,
                ],
                [1.42 * inch, 0.9 * inch, 2.0 * inch, 0.62 * inch, 0.65 * inch],
            ),
            Spacer(1, 0.12 * inch),
            _p(
                "Fresh OPD's one-step train calls average 136.65 seconds and reach 7.3% strict "
                "success. SFT and frozen KD remain eight-step outcomes because the configured "
                "cycle ceiling terminates them before the internal timer. No exact equal-time "
                "ranking is claimed.",
                styles["body"],
            ),
            _p("8. Cost and failure analysis", styles["h1"]),
            _p(
                "At equal updates, strict OPD averages 686.80 continuation seconds versus 52.10 "
                "for frozen KD, about 13.2x. Peak reserved VRAM is similar for the KD/OPD methods "
                "at roughly 3.47 GiB. The position-budget arm queries 49.77% of model-generated "
                "positions but is not faster: compressed target selection reduces stored or "
                "projected positions, not teacher backbone forward cost.",
                styles["body"],
            ),
            _p(
                "Seed variation is large. Oracle KD ranges from 0.0% to 54.7% strict success, "
                "frozen KD from 9.4% to 36.7%, full fresh OPD from 0.0% to 26.6%, and the 50% arm "
                "from 10.9% to 49.2%. The data therefore support a scoped negative comparison, "
                "not a universal ordering of algorithms.",
                styles["body"],
            ),
        ]
    )

    story.extend(
        [
            PageBreak(),
            _p("9. Limitations", styles["h1"]),
            _p(
                "The study uses one Qwen3 student/teacher pair, one read-only SQLite recovery "
                "environment, one RTX 4080 and three seeds. It evaluates mechanism-level task "
                "success and recovery, not safety, preference, over-refusal or general utility. "
                "The teacher is qualified on 96 eval tasks, not proven universally competent. "
                "Final-test generation is deterministic, but training remains sensitive to "
                "seed and finite task schedules.",
                styles["body"],
            ),
            _p(
                "The wall-time secondary view is cycle-capped for six runs and must not be read as "
                "an exact equal-wall experiment. Task-paired bootstrap intervals condition on this "
                "fixed task set and do not replace replication across models and environments. "
                "The 50% selector changes queried positions but not teacher forward count. No "
                "claim of statistical significance is made solely from three seeds.",
                styles["body"],
            ),
            _p("10. Reproducibility and artifact integrity", styles["h1"]),
            _p(
                "Preregistration commit: 7087b3a333463b88a62ffed73daee2c85d039145. "
                "Revision-1.3 digest: "
                "9c4c2ec19a56cebb2b2c1c0f3c7e504a9285467c99ae1590488251fbf2ff3934. "
                "Final execution commit recorded by the results: "
                "dba595f55ac634c7e0735db696a052126994bb26. The selected teacher adapter "
                "revision and weight digest are recorded in every KD/OPD arm and every frozen "
                "dataset manifest.",
                styles["body"],
            ),
            _table(
                [
                    ["Artifact", "SHA-256"],
                    *[
                        [_p(name, styles["table_cell"]), _p(digest, styles["hash"])]
                        for name, digest in EXPECTED_HASHES.items()
                    ],
                    [
                        _p("recoverybench-v1-task-results.jsonl", styles["table_cell"]),
                        _p(
                            "76ab53202f8ad1eb332b056c9c840eb34816986883a568813edc0e0f502d3086",
                            styles["hash"],
                        ),
                    ],
                    [
                        _p("legacy calculator result", styles["table_cell"]),
                        _p(
                            "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc",
                            styles["hash"],
                        ),
                    ],
                ],
                [2.25 * inch, 4.05 * inch],
            ),
            Spacer(1, 0.12 * inch),
            _p(
                "Audit coverage: 36 task artifacts, 4,608 task trajectories and 101,787,618 raw "
                "bytes; every embedded SHA-256 and byte count matched. Task IDs are paired across "
                "all methods within seed. Three cold checkpoints and three frozen-student dataset "
                "digests are shared across budget views. The legacy calculator JSON remains "
                "byte-identical.",
                styles["body"],
            ),
            _p("11. Conclusion", styles["h1"]),
            _p(
                "RecoveryBench finds no fresh-state advantage in this setting. Frozen-student KD "
                "is better and far cheaper than strict fresh OPD under the preregistered primary "
                "comparison. The result narrows miniVERL's product claim: it is an independent "
                "one-GPU companion for prototyping, diagnosing and validating online post-training "
                "workflows, not a claim that OPD replaces SFT or that local execution is equivalent "
                "to a distributed verl runtime.",
                styles["body"],
            ),
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.68 * inch,
        leftMargin=0.68 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.7 * inch,
        title="Fresh-State Distillation for Tool-Error Recovery on a Consumer GPU",
        author="Daoyuan Li",
        subject="miniVERL RecoveryBench v1",
    )
    document.build(
        story,
        onFirstPage=_page,
        onLaterPages=_page,
        canvasmaker=_invariant_canvas,
    )


if __name__ == "__main__":
    build()
    print(f"wrote {OUTPUT.relative_to(ROOT)} sha256={_sha256(OUTPUT)}")
