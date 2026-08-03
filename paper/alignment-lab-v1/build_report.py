#!/usr/bin/env python3
"""Build the deterministic, data-bound Alignment Lab v1 report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
RESULT = ROOT / "benchmarks" / "results" / "alignment-lab-v1.json"
TASKS = ROOT / "benchmarks" / "results" / "alignment-lab-v1-task-results.jsonl"
STATE = ROOT / "benchmarks" / "results" / "alignment-lab-v1-state-supervision.json"
PREREGISTRATION = ROOT / "benchmarks" / "preregistration" / "alignment-lab-v1.yaml"
CALCULATOR = ROOT / "benchmarks" / "results" / "gpu-calc-hard-equal-update-v2.json"
OUTPUT = Path(__file__).with_name("alignment-lab-v1.pdf")

EXPECTED_HASHES = {
    RESULT: "584752dccb91654109c357b8ebb12681a12a9c1476a9ba539dd35e4d860a22ef",
    TASKS: "8d7fc723436d7377d196fc44046d960e3cb7f0aa81e03d49ef05b627eb84630f",
    STATE: "9e08129ba4cd9e460c189b94b4e421d881ba69e3938f02eac95d251f50c88788",
    PREREGISTRATION: "71307dbfe9a5bb20c686307cafce8bd254c07af8b69c1bf1c6ec0dbf53a8cde0",
    CALCULATOR: "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc",
}

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#526079")
CYAN = colors.HexColor("#0891b2")
BLUE = colors.HexColor("#2563eb")
VIOLET = colors.HexColor("#7c3aed")
GREEN = colors.HexColor("#059669")
RED = colors.HexColor("#be123c")
LIGHT = colors.HexColor("#f1f5f9")
PALETTE = [
    colors.HexColor("#94a3b8"),
    colors.HexColor("#60a5fa"),
    colors.HexColor("#a78bfa"),
    colors.HexColor("#fbbf24"),
    colors.HexColor("#fb7185"),
    colors.HexColor("#34d399"),
]

LABELS = {
    "sft_checkpoint": "SFT checkpoint",
    "continued_sft": "continued SFT",
    "dpo": "DPO",
    "offline_distillation": "offline soft distillation",
    "standard_opd": "standard OPD",
    "verifier_gated_opd": "verifier-gated OPD",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in EXPECTED_HASHES.items():
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"unexpected Alignment Lab source hash for {path.name}: {actual}")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if result.get("measurement_status") != "measured_final":
        raise ValueError("Alignment Lab result is not a measured final artifact")
    if len(result.get("arms", [])) != 18:
        raise ValueError("Alignment Lab result must contain 18 arms")
    if any(arm.get("status") != "completed" for arm in result["arms"]):
        raise ValueError("Alignment Lab report refuses incomplete arms")
    if len(TASKS.read_text(encoding="utf-8").splitlines()) != 864:
        raise ValueError("Alignment Lab task evidence must contain 864 records")
    return result, state


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
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11.5,
            leading=16,
            textColor=MUTED,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15.5,
            leading=19,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=INK,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.1,
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
            fontSize=10.7,
            leading=15.4,
            textColor=RED,
            borderColor=colors.HexColor("#fecdd3"),
            borderWidth=0.8,
            borderPadding=9,
            backColor=colors.HexColor("#fff1f2"),
            spaceAfter=11,
        ),
        "center": ParagraphStyle(
            "Center",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=10.5,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.8,
            leading=8.3,
            textColor=INK,
        ),
        "hash": ParagraphStyle(
            "Hash",
            parent=base["Code"],
            fontName="Courier",
            fontSize=5.0,
            leading=6.2,
            textColor=INK,
        ),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _table(rows: list[list[Any]], widths: list[float], *, header: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.3),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.3),
            ]
        )
    for row in range(1 if header else 0, len(rows)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT))
    table.setStyle(TableStyle(commands))
    return table


class QualityUtilityPlot(Flowable):
    """Small vector quality-versus-utility plot bound to method means."""

    def __init__(self, summary: list[dict[str, Any]], width: float = 6.7 * inch):
        super().__init__()
        self.summary = summary
        self.width = width
        self.height = 3.0 * inch

    def draw(self) -> None:
        canvas = self.canv
        left, bottom = 0.62 * inch, 0.48 * inch
        chart_w, chart_h = 4.65 * inch, 2.18 * inch
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.rect(left, bottom, chart_w, chart_h, fill=0, stroke=1)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        for index in range(6):
            value = 0.8 + 0.04 * index
            x = left + chart_w * (value - 0.8) / 0.2
            y = bottom + chart_h * (value - 0.8) / 0.2
            canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
            canvas.line(x, bottom, x, bottom + chart_h)
            canvas.line(left, y, left + chart_w, y)
            canvas.setFillColor(MUTED)
            canvas.drawCentredString(x, bottom - 11, f"{value:.2f}")
            canvas.drawRightString(left - 5, y - 2, f"{value:.2f}")
        for index, row in enumerate(self.summary):
            utility = float(row["tool_utility_retention_mean"])
            quality = float(row["alignment_score_mean"])
            x = left + chart_w * (utility - 0.8) / 0.2
            y = bottom + chart_h * (quality - 0.8) / 0.2
            canvas.setStrokeColor(PALETTE[index])
            canvas.setLineWidth(2)
            canvas.circle(x, y, 4 + index * 1.4, fill=0, stroke=1)
            legend_y = bottom + chart_h - index * 20
            canvas.setFillColor(PALETTE[index])
            canvas.circle(left + chart_w + 26, legend_y, 3.4, fill=1, stroke=0)
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica", 7.2)
            canvas.drawString(left + chart_w + 35, legend_y - 2.5, LABELS[row["method"]])
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(left + chart_w / 2, 3, "tool utility retention")
        canvas.saveState()
        canvas.translate(8, bottom + chart_h / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, "alignment score")
        canvas.restoreState()


class QueryCostBars(Flowable):
    """Compact teacher-query and GPU-time comparison."""

    def __init__(self, summary: list[dict[str, Any]], width: float = 6.7 * inch):
        super().__init__()
        self.summary = summary
        self.width = width
        self.height = 2.25 * inch

    def draw(self) -> None:
        canvas = self.canv
        label_w = 1.55 * inch
        bar_w = 3.7 * inch
        max_time = max(float(row["gpu_seconds_mean"]) for row in self.summary) or 1
        for index, row in enumerate(self.summary):
            y = self.height - 17 - index * 24
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica", 7.2)
            canvas.drawString(0, y + 2, LABELS[row["method"]])
            canvas.setFillColor(PALETTE[index])
            time = float(row["gpu_seconds_mean"])
            canvas.roundRect(label_w, y, bar_w * time / max_time, 7, 3, fill=1, stroke=0)
            ratio = row.get("teacher_query_ratio_mean")
            ratio_text = "n/a" if ratio is None else f"{100 * float(ratio):.1f}%"
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica-Bold", 7.2)
            canvas.drawRightString(self.width, y + 1, f"{time:.1f}s · query {ratio_text}")
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(
            label_w,
            2,
            "continuation GPU time; query ratio is selected positions / generated positions",
        )


def _page(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(doc.leftMargin, 0.52 * inch, letter[0] - doc.rightMargin, 0.52 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(doc.leftMargin, 0.34 * inch, "miniVERL Alignment Lab v1")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.34 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _invariant_canvas(*args: Any, **kwargs: Any) -> Canvas:
    kwargs["invariant"] = 1
    return Canvas(*args, **kwargs)


def build(output: Path = OUTPUT) -> None:
    result, state = _load()
    summary = result["method_summary"]
    styles = _styles()
    story: list[Flowable] = []

    story.extend(
        [
            Spacer(1, 0.48 * inch),
            _p("Online Policy Distillation After SFT on One Consumer GPU", styles["title"]),
            _p(
                "Alignment Lab v1 technical report · miniVERL v0.5.0 · Daoyuan Li",
                styles["subtitle"],
            ),
            _p(
                "Primary finding: the common SFT checkpoint already scored 100% alignment and "
                "100% retained tool utility in every seed. No continuation method improved it. "
                "The evidence-based decision is to turn online teacher querying off for this "
                "recipe, while preserving every completed regression.",
                styles["callout"],
            ),
            _p("Abstract", styles["h1"]),
            _p(
                "Alignment Lab v1 compares a frozen SFT checkpoint, continued alignment SFT, "
                "DPO, offline soft teacher distillation, standard OPD and verifier-gated OPD "
                "from one checksummed Qwen3-0.6B SFT checkpoint. The final test uses 48 paired "
                "deterministic sandbox policy tasks and three preregistered seeds. The SFT "
                "checkpoint saturates the suite. DPO and offline distillation tie it; continued "
                "SFT, standard OPD and verifier-gated OPD each contain completed regressions. "
                "Harmful compliance and over-refusal remain zero for all methods, showing that "
                "those axes alone can miss a safe-recovery utility failure. The result supports "
                "a scoped no-continuation decision, not a broad claim that OPD is ineffective.",
                styles["body"],
            ),
            _p(
                "Keywords: alignment, on-policy distillation, DPO, retained utility, verifier gating, consumer GPU",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("1. Product and scientific question", styles["h1"]),
            _p(
                "SFT establishes instruction following, protocol competence and task capability. "
                "OPD is a later online teacher-student mechanism whose transferred behavior "
                "depends on the teacher. The experiment therefore starts every method from the "
                "same SFT checkpoint and asks whether any continuation improves policy quality "
                "without sacrificing retained utility, and at what query, time and memory cost.",
                styles["body"],
            ),
            _p("2. Controlled design", styles["h1"]),
            _table(
                [
                    ["Item", "Frozen value"],
                    ["Model", "Qwen/Qwen3-0.6B at c1899de2…"],
                    ["Starting checkpoint", result["starting_checkpoint_sha256"]],
                    ["Policy", f"Minipolicy v1 at {result['policy_sha256']}"],
                    ["Final test", "48 ordered paired tasks per arm; one read; greedy decoding"],
                    ["Seeds", "1234, 20260727, 20260801"],
                    ["Budget", "four continuation updates; frozen SFT checkpoint has none"],
                    ["Hardware", "NVIDIA GeForce RTX 4080; cross-GPU generalization not tested"],
                ],
                [1.35 * inch, 5.05 * inch],
            ),
            Spacer(1, 0.12 * inch),
            _p("3. Methods and roles", styles["h1"]),
            _table(
                [
                    ["Method", "State / signal", "Teacher or reference", "Purpose"],
                    ["SFT checkpoint", "none", "none", "common no-continuation baseline"],
                    ["continued SFT", "oracle hard tokens", "none", "continued alignment baseline"],
                    [
                        "DPO",
                        "frozen preferences",
                        "fixed reference",
                        "preference baseline; TRL 1.8.0",
                    ],
                    [
                        "offline distillation",
                        "frozen states; soft",
                        "qualified teacher",
                        "fixed-state distribution transfer",
                    ],
                    [
                        "standard OPD",
                        "fresh states; soft",
                        "qualified teacher",
                        "full online transfer",
                    ],
                    [
                        "verifier-gated OPD",
                        "fresh critical spans",
                        "qualified teacher + gate",
                        "localized transfer",
                    ],
                ],
                [1.12 * inch, 1.43 * inch, 1.42 * inch, 2.43 * inch],
            ),
            Spacer(1, 0.1 * inch),
            _p(
                "The policy-conditioned teacher sees a private deterministic rubric; the student "
                "does not. All tools are synthetic deterministic sandboxes. No real destructive "
                "action is executed. External IFEval, XSTest, HarmBench and RewardBench adapters "
                "are pinned metadata only and are not measured endpoints in this artifact.",
                styles["body"],
            ),
        ]
    )

    result_rows = []
    for row in summary:
        result_rows.append(
            [
                LABELS[row["method"]],
                f"{100 * row['alignment_score_mean']:.1f}%",
                f"{100 * row['harmful_compliance_rate_mean']:.1f}%",
                f"{100 * row['over_refusal_rate_mean']:.1f}%",
                f"{100 * row['tool_utility_retention_mean']:.1f}%",
                "n/a"
                if row["teacher_query_ratio_mean"] is None
                else f"{100 * row['teacher_query_ratio_mean']:.1f}%",
                f"{row['gpu_seconds_mean']:.1f}s",
            ]
        )
    story.extend(
        [
            PageBreak(),
            _p("4. Final three-seed result", styles["h1"]),
            _table(
                [["Method", "Align", "Harm", "Over", "Utility", "Query", "GPU"], *result_rows],
                [
                    1.44 * inch,
                    0.62 * inch,
                    0.58 * inch,
                    0.58 * inch,
                    0.66 * inch,
                    0.64 * inch,
                    0.62 * inch,
                ],
            ),
            Spacer(1, 0.15 * inch),
            QualityUtilityPlot(summary),
            _p(
                "Concentric points at 1.0 / 1.0 denote exact mean overlap, not algorithmic "
                "equivalence. Every non-overlapping regression remains in the primary table.",
                styles["center"],
            ),
            Spacer(1, 0.1 * inch),
            _p("Cost and query accounting", styles["h2"]),
            QueryCostBars(summary),
            _p(
                "DPO time includes its external pinned TRL training. Evaluation is excluded from "
                "the continuation-GPU-time axis. Query ratio counts selected positions and does "
                "not imply proportional teacher-backbone FLOP savings.",
                styles["small"],
            ),
        ]
    )

    comparisons = state["matched_comparisons"]
    story.extend(
        [
            PageBreak(),
            _p("5. State × Supervision diagnostic", styles["h1"]),
            _p(
                "The diagnostic separates state source from target type while remaining explicit "
                "that it measures teacher signal, not a separately trained hard-target outcome.",
                styles["body"],
            ),
            _table(
                [
                    ["Matched comparison", "Observed signal", "Interpretation"],
                    [
                        "frozen hard vs fresh hard",
                        "argmax/token match 1.0000 vs 1.0000",
                        "no observed argmax disagreement",
                    ],
                    [
                        "frozen soft vs fresh soft",
                        "entropy 0.00235 vs 0.00216 nats",
                        "fresh minus frozen −0.00019 nats",
                    ],
                    [
                        "fresh hard vs fresh soft",
                        f"{100 * comparisons['fresh_hard_vs_fresh_soft']['soft_probability_mass_beyond_argmax_mean']:.4f}% mass beyond argmax",
                        "same states, teacher, budget, checkpoint and seeds",
                    ],
                ],
                [1.5 * inch, 2.18 * inch, 2.72 * inch],
            ),
            Spacer(1, 0.12 * inch),
            _p(
                "The fresh soft target contains almost no probability mass beyond its argmax in "
                "this saturated recipe. No soft-target quality advantage is claimed. The result "
                "does not generalize to teachers or states with higher entropy or disagreement.",
                styles["callout"],
            ),
            _p("6. Verifier-Gated OPD", styles["h1"]),
            _p(
                "The policy-critical-span-v1 gate was calibrated on eval, frozen before test, and "
                "records a decision per example and span. Mean queried positions fall from 100% "
                "for standard OPD to 46.8% for gated OPD. Mean GPU time falls from 76.7 to 66.0 "
                "seconds, but alignment and retained utility do not improve. Localized or "
                "verifier-qualified distillation is not claimed as novel.",
                styles["body"],
            ),
            _p("7. Why two safety axes were insufficient", styles["h1"]),
            _p(
                "All methods score 0% harmful compliance and 0% over-refusal. Nevertheless, "
                "continued SFT, standard OPD and verifier-gated OPD regress on safe error "
                "recovery. Alignment evaluation must include benign completion and retained "
                "utility rather than treating zero harmful compliance as a sufficient endpoint.",
                styles["body"],
            ),
        ]
    )

    failures = [arm for arm in result["arms"] if float(arm["metrics"]["alignment_score"]) < 1.0]
    failure_rows = [
        [
            LABELS[arm["method"]],
            str(arm["seed"]),
            f"{100 * arm['metrics']['alignment_score']:.1f}%",
            str(arm["provenance"]["final_failure_audit"]["failed_tasks"]),
            ", ".join(arm["provenance"]["final_failure_audit"]["policy_categories"]),
        ]
        for arm in failures
    ]
    story.extend(
        [
            PageBreak(),
            _p("8. Preserved negative evidence", styles["h1"]),
            _table(
                [["Method", "Seed", "Alignment", "Failed", "Category"], *failure_rows],
                [1.55 * inch, 0.8 * inch, 0.85 * inch, 0.65 * inch, 2.35 * inch],
            ),
            Spacer(1, 0.12 * inch),
            _p(
                "No completed arm was rerun after final-test inspection. The seed-1234 SFT "
                "baseline is the checksummed union of two disjoint 24-task segments under the "
                "public preregistration recovery rule; it contains 48 unique tasks and no repeat. "
                "An interrupted continued-SFT construction run was never evaluated and remains "
                "outside the headline result.",
                styles["body"],
            ),
            _p("9. Pilot decision", styles["h1"]),
            _p(
                "alignment-pilot-v1 returns recommendation: insufficient_evidence. Operational "
                "decision: do not spend online teacher-query cost on this already-saturated "
                "recipe. The starting policy is at the measured ceiling, no continuation method "
                "improves it, and the matched teacher signal exposes almost no hard/soft gap. A "
                "more discriminating policy suite is required before choosing DPO, offline "
                "distillation or either OPD variant. The pilot binds measured time, VRAM and "
                "teacher-query fraction. Free-running teacher competence, distribution-level "
                "student/teacher top-k overlap, independent gate precision and a population "
                "uncertainty interval were not measured and remain null rather than zero.",
                styles["callout"],
            ),
            _p("10. Alignment Cards", styles["h1"]),
            _p(
                "Each of the 18 completed arms has JSON and Markdown cards recording its starting "
                "SFT checkpoint, teacher/reference identity, method, policy revision, alignment "
                "and over-alignment metrics, retained utility, query ratio, cost, VRAM, limits "
                "and artifact hashes. Public cards contain no machine-local path or credential.",
                styles["body"],
            ),
        ]
    )

    story.extend(
        [
            PageBreak(),
            _p("11. Scope and limitations", styles["h1"]),
            _p(
                "One small model family, one deterministic synthetic policy suite, three seeds "
                "and one RTX 4080 do not support broad safety, capability, population or "
                "cross-hardware claims. The suite is too easy for the common SFT checkpoint. "
                "External safety, preference, instruction-following and general-capability suites "
                "were not executed. GPU time and VRAM are observations for this machine and "
                "software stack, not forecasts for another card.",
                styles["body"],
            ),
            _p(
                "The teacher-query ratio is a selected-position ratio, not a teacher-forward or "
                "FLOP ratio. DPO uses a pinned external TRL run and is accounted with that cost. "
                "The deterministic validators improve auditability but cannot establish general "
                "safety. No method novelty is claimed.",
                styles["body"],
            ),
            _p("12. Reproducibility and integrity", styles["h1"]),
            _table(
                [
                    ["Artifact", "SHA-256"],
                    *[
                        [_p(path.name, styles["cell"]), _p(digest, styles["hash"])]
                        for path, digest in EXPECTED_HASHES.items()
                    ],
                ],
                [2.3 * inch, 4.1 * inch],
            ),
            Spacer(1, 0.11 * inch),
            _p(
                "The machine-readable result binds all 18 arms, DPO provenance, the disjoint "
                "baseline recovery and 864 task-level records. Exact generation tests compare "
                "every public figure, Markdown report and 36 Alignment Card files byte for byte. "
                "The immutable calculator result remains unchanged.",
                styles["body"],
            ),
            _p("13. Conclusion", styles["h1"]),
            _p(
                "Alignment Lab v1 does not find an incremental alignment benefit because its SFT "
                "starting point already saturates the deterministic suite. The useful outcome is "
                "a calibrated no-continuation decision and a complete cost/utility record. "
                "miniVERL should be used to justify when online teacher-student training is worth "
                "running—not to presume that OPD should replace SFT.",
                styles["body"],
            ),
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.66 * inch,
        leftMargin=0.66 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.7 * inch,
        title="Online Policy Distillation After SFT on One Consumer GPU",
        author="Daoyuan Li",
        subject="miniVERL Alignment Lab v1",
    )
    document.build(story, onFirstPage=_page, onLaterPages=_page, canvasmaker=_invariant_canvas)


if __name__ == "__main__":
    build()
    print(f"wrote {OUTPUT.relative_to(ROOT)} sha256={_sha256(OUTPUT)}")
