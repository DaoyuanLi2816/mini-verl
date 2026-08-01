"""Torch-free inspection and reporting invariants.

This file protects the promises miniVERL makes about *reading* run artifacts on
a machine with nothing but the base install:

* ``miniverl inspect`` re-derives the provenance masks from the span partition,
  so a hand-edited trajectory file cannot smuggle ``tool_result`` (or
  ``system``/``user``) tokens into the training loss -- it raises
  :class:`~miniverl.errors.SchemaValidationError` instead of printing a
  reassuring summary.  The same holds for a file written by an incompatible
  schema version.
* The provenance summary never lists a context span type as trainable.
* ``miniverl report`` reads a run directory built entirely of JSON/JSONL/YAML,
  and reports quantities that were *not measured* as ``None``/``n/a`` rather
  than as a comforting zero -- in particular a CPU run on a GPU box must not
  claim ``0.000 GiB`` peak VRAM.
* The HTML report is self-contained and inert: no ``<script>``, no remote
  ``src``/``href``, and every value that came from run data is escaped.

Everything here is built by hand -- no training, no torch, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from miniverl.errors import RunNotFoundError, SchemaValidationError
from miniverl.inspection import iter_spans_for_display, summarize_file
from miniverl.reporting.charts import bar_chart, line_chart, token_strip
from miniverl.reporting.data import ReportData
from miniverl.reporting.html import render_html
from miniverl.reporting.markdown import render_markdown, render_summary_json
from miniverl.schemas.trajectory import (
    MODEL_GENERATED_SPAN_TYPES,
    Span,
    SpanType,
    TerminationReason,
    ToolCallRecord,
    ToolResultRecord,
    Trajectory,
    Turn,
    VerificationRecord,
)
from miniverl.trajectory.io import write_trajectories
from miniverl.trajectory.masks import build_masks
from miniverl.utils.runs import RunPaths, write_json

RUN_ID = "20260727-010203-inspect-demo"
FINGERPRINT = "fingerprint-c0ffee-0123456789abcdef-0123456789abcdef"
MODEL_ID = "toy/student-4m"

#: One hand-built layout shared by every trajectory below, so the tests can
#: reason about absolute token offsets without re-deriving them.
SPAN_LAYOUT: tuple[tuple[SpanType, int, int, int], ...] = (
    (SpanType.SYSTEM, 0, 3, 0),
    (SpanType.USER, 3, 7, 0),
    (SpanType.ASSISTANT_TOOL_CALL, 7, 11, 0),
    (SpanType.TOOL_RESULT, 11, 15, 0),
    (SpanType.ASSISTANT_FINAL, 15, 20, 1),
)
SEQ_LEN = SPAN_LAYOUT[-1][2]
MODEL_TOKENS_PER_TRAJ = 4 + 5
CONTEXT_TOKENS_PER_TRAJ = SEQ_LEN - MODEL_TOKENS_PER_TRAJ
TOKENS_BY_SPAN_TYPE_PER_TRAJ = {
    "system": 3,
    "user": 4,
    "assistant_tool_call": 4,
    "tool_result": 4,
    "assistant_final": 5,
}

#: Deliberately hostile text: it must survive into the HTML only as escaped
#: entities, never as a live tag.
HOSTILE_TEXT = "<script>alert(1)</script>"


# -- hand-built artifacts -------------------------------------------------


def _build_trajectory(
    *,
    trajectory_id: str,
    task_id: str,
    policy_version: int,
    termination_reason: TerminationReason,
    tool_name: str,
    tool_call_valid: bool,
    verification: VerificationRecord | None,
    user_text: str = "compute 6 * 7",
) -> Trajectory:
    """Build one structurally real trajectory with consistent masks."""
    texts = {
        SpanType.SYSTEM: "you may call tools",
        SpanType.USER: user_text,
        SpanType.ASSISTANT_TOOL_CALL: f'{{"tool": "{tool_name}", "args": {{"expr": "6*7"}}}}',
        SpanType.TOOL_RESULT: "42" if tool_call_valid else "parse error",
        SpanType.ASSISTANT_FINAL: "the answer is 42",
    }
    spans = [
        Span(
            span_type=span_type,
            start=start,
            end=end,
            turn_id=turn_id,
            text=texts[span_type],
            tool_name=(tool_name if span_type is SpanType.ASSISTANT_TOOL_CALL else None),
            tool_call_id=(
                f"{trajectory_id}-call-0"
                if span_type in (SpanType.ASSISTANT_TOOL_CALL, SpanType.TOOL_RESULT)
                else None
            ),
        )
        for span_type, start, end, turn_id in SPAN_LAYOUT
    ]
    model_mask, critical_mask = build_masks(spans, SEQ_LEN)
    call = ToolCallRecord(
        call_id=f"{trajectory_id}-call-0",
        name=tool_name,
        arguments={"expr": "6*7"},
        raw_text=texts[SpanType.ASSISTANT_TOOL_CALL],
        valid=tool_call_valid,
        parse_error=None if tool_call_valid else "arguments are not valid JSON",
    )
    result = ToolResultRecord(
        call_id=f"{trajectory_id}-call-0",
        ok=tool_call_valid,
        result="42" if tool_call_valid else "",
        error=None if tool_call_valid else "tool call was not parseable",
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        environment="calc_tools",
        token_ids=list(range(100, 100 + SEQ_LEN)),
        attention_mask=[1] * SEQ_LEN,
        model_generated_mask=model_mask,
        critical_mask=critical_mask,
        spans=spans,
        turns=[
            Turn(turn_id=0, tool_call=call, tool_result=result),
            Turn(turn_id=1, is_final=True),
        ],
        policy_version=policy_version,
        tokenizer_fingerprint=FINGERPRINT,
        model_id=MODEL_ID,
        model_revision="main",
        verification=verification,
        termination_reason=termination_reason,
        generated_token_count=MODEL_TOKENS_PER_TRAJ,
        invalid_tool_calls=0 if tool_call_valid else 1,
        metadata={"source": "hand-built"},
    )


def _trajectories() -> list[Trajectory]:
    """Three trajectories: solved, unsolved-with-invalid-call, and ungraded."""
    return [
        _build_trajectory(
            trajectory_id="traj-0",
            task_id="task-a",
            policy_version=0,
            termination_reason=TerminationReason.FINAL_ANSWER,
            tool_name="calculator",
            tool_call_valid=True,
            verification=VerificationRecord(solved=True, reward=1.0, predicted="42", expected="42"),
            # The report must escape this instead of emitting a live tag.
            user_text=f"compute 6 * 7 {HOSTILE_TEXT}",
        ),
        _build_trajectory(
            trajectory_id="traj-1",
            task_id="task-b",
            policy_version=1,
            termination_reason=TerminationReason.MAX_TURNS,
            tool_name="search",
            tool_call_valid=False,
            verification=VerificationRecord(
                solved=False,
                reward=0.0,
                predicted="41",
                expected="42",
                failure_category="wrong_answer",
            ),
        ),
        _build_trajectory(
            trajectory_id="traj-2",
            task_id="task-c",
            policy_version=1,
            termination_reason=TerminationReason.FINAL_ANSWER,
            tool_name="calculator",
            tool_call_valid=True,
            verification=None,
        ),
    ]


def _manifest() -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "mode": "opd",
        "miniverl_version": "0.1.0",
        "git_commit": "0123456789abcdef",
        "created_at": "2026-07-27T01:02:03+00:00",
        "seed": 1234,
        "deterministic": True,
        "python_version": "3.11.9",
        "os": "Windows",
        "os_release": "11",
        "platform": "win-amd64",
        # The *run* saw no CUDA device even though the machine has one.
        "gpu": {"available": False, "reason": "device was forced to cpu"},
        "packages": {
            "torch": "2.5.1",
            "transformers": "4.46.0",
            "peft": "0.13.2",
            "bitsandbytes": None,
        },
        "models": {
            "device": "cpu",
            "student": {
                "model_id": MODEL_ID,
                "revision": "main",
                "precision": "float32",
                "quantization": "none",
                "capabilities": {"num_trainable_parameters": 4096},
            },
            "teacher": {
                "model_id": "toy/teacher-16m",
                "revision": "main",
                "precision": "float32",
                "quantization": "none",
                "context_mode": "full",
            },
            "tokenizer_fingerprint": FINGERPRINT,
            "tokenizer_vocab_size": 512,
        },
        "objective": {
            "divergence": "reverse_kl",
            "loss_mode": "exact",
            "top_k": 64,
            "temperature": 1.0,
            "selector": "critical_tokens",
        },
        "memory": {"strategy": "chunked_projection", "projection_chunk_size": 64},
        "environment": {
            "name": "calc_tools",
            "difficulty": "easy",
            "split_sizes": {"train": 8, "eval": 4},
        },
        "measurement_status": {"peak_vram": "not measured (no CUDA in this run)"},
    }


def _environment() -> dict[str, Any]:
    """A machine description that *does* advertise a GPU."""
    return {
        "gpu": {
            "available": True,
            "name": "NVIDIA GeForce RTX 4080",
            "total_memory_gib": 15.99,
            "capability": "8.9",
            "driver_version": "560.94",
            "torch_cuda_version": "12.4",
        },
        "packages": {"torch": "2.5.1", "transformers": "4.46.0"},
        "platform": {"system": "Windows", "release": "11"},
    }


def _metrics() -> list[dict[str, Any]]:
    memory = {"peak_allocated_bytes": 2 * 1024**3, "peak_reserved_bytes": 3 * 1024**3}
    return [
        {
            "phase": "sft_warmup",
            "step": 0,
            "loss": 2.5,
            "train_selected_tokens_per_second": 100.0,
            "memory": memory,
        },
        {
            "phase": "sft_warmup",
            "step": 1,
            "loss": 2.0,
            "train_selected_tokens_per_second": 120.0,
            "memory": memory,
        },
        {
            "phase": "opd",
            "step": 2,
            "loss": 1.5,
            "train_selected_tokens_per_second": 140.0,
            "memory": memory,
        },
        {
            "phase": "opd",
            "step": 3,
            "loss": 1.0,
            "train_selected_tokens_per_second": 160.0,
            "memory": memory,
        },
        {
            "phase": "opd_cycle",
            "cycle": 0,
            "selection": {
                "selected_by_span_type": {"assistant_final": 10, "assistant_tool_call": 8}
            },
        },
        {
            "phase": "eval",
            "global_step": 0,
            "success_rate": 0.25,
            "failure_categories": {"wrong_answer": 2, "no_final_answer": 1},
            "termination_reasons": {"final_answer": 3, "max_turns": 1},
        },
        {
            "phase": "eval",
            "global_step": 4,
            "success_rate": 0.5,
            "failure_categories": {"wrong_answer": 2},
            "termination_reasons": {"final_answer": 3, "max_turns": 1},
        },
    ]


def _events() -> list[dict[str, Any]]:
    return [
        {"ts": "2026-07-27T01:02:03+00:00", "event": "run_started", "mode": "opd"},
        {
            "ts": "2026-07-27T01:02:10+00:00",
            "event": "rollouts_collected",
            "trajectories": 3,
            "rollout_tokens_per_second": 90.0,
        },
        {
            "ts": "2026-07-27T01:02:40+00:00",
            "event": "rollouts_collected",
            "trajectories": 3,
            "rollout_tokens_per_second": 110.0,
        },
        {"ts": "2026-07-27T01:03:00+00:00", "event": "run_finished", "ok": True},
    ]


def _eval_json() -> dict[str, Any]:
    return {
        "duration_seconds": 123.5,
        "policy_version": 1,
        "baseline_eval": {
            "tag": "baseline",
            "policy_version": 0,
            "global_step": 0,
            "tasks": 4,
            "success_rate": 0.25,
            "avg_turns": 2.5,
            "invalid_tool_call_rate": 0.125,
            "parse_valid_tool_call_rate": 0.875,
            "tool_execution_success_rate": 0.8,
            "generated_tokens_per_task": 40.0,
            "rollout_tokens_per_second": 90.0,
        },
        "eval": {
            "tag": "final",
            "policy_version": 1,
            "global_step": 4,
            "tasks": 4,
            "success_rate": 0.5,
            "avg_turns": 2.0,
            "invalid_tool_call_rate": 0.0,
            "parse_valid_tool_call_rate": 1.0,
            "tool_execution_success_rate": 1.0,
            "generated_tokens_per_task": 32.0,
            "rollout_tokens_per_second": 110.0,
        },
    }


def _token_analysis() -> list[dict[str, Any]]:
    """Per-token records for traj-0 and traj-1 only (traj-2 has none)."""
    rows: list[dict[str, Any]] = []
    for traj_id, base in (("traj-0", 0.1), ("traj-1", 0.4)):
        for offset, position in enumerate(range(8, 11)):
            rows.append(
                {
                    "trajectory_id": traj_id,
                    "target_position": position,
                    "span_type": "assistant_tool_call",
                    "token_id": 100 + position,
                    "token_piece": HOSTILE_TEXT if offset == 0 else f"tok{offset}",
                    "token_loss": base + offset * 0.05,
                    "weight": 1.0,
                    "teacher_entropy": 0.5 + offset * 0.1,
                    "teacher_top_token": "42",
                    "student_top_token": "41" if offset else "42",
                }
            )
    return rows


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


@pytest.fixture
def synthetic_run(tmp_path: Path) -> Path:
    """A complete, hand-written run directory -- no training involved."""
    root = tmp_path / "runs" / RUN_ID
    root.mkdir(parents=True)
    write_json(root / "manifest.json", _manifest())
    write_json(root / "environment.json", _environment())
    write_json(root / "eval.json", _eval_json())
    (root / "config.original.yaml").write_text(
        "mode: opd\nmodels:\n  device: auto\n", encoding="utf-8"
    )
    (root / "config.resolved.yaml").write_text(
        "mode: opd\nmodels:\n  device: cpu\n", encoding="utf-8"
    )
    _write_jsonl(root / "metrics.jsonl", _metrics())
    _write_jsonl(root / "events.jsonl", _events())
    _write_jsonl(root / "token_analysis.jsonl", _token_analysis())
    written = write_trajectories(root / "trajectories.jsonl", _trajectories())
    assert written == 3
    return root


@pytest.fixture
def trajectory_file(synthetic_run: Path) -> Path:
    return synthetic_run / "trajectories.jsonl"


@pytest.fixture
def report_data(synthetic_run: Path) -> ReportData:
    return ReportData.from_run(synthetic_run)


def _tamper(path: Path, mutate: Any) -> Path:
    """Rewrite ``path`` after applying ``mutate`` to the first record."""
    lines = path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines if line.strip()]
    mutate(payloads[0])
    _write_jsonl(path, payloads)
    return path


# -- summarize_file -------------------------------------------------------


def test_summarize_file_counts_tokens_and_provenance(trajectory_file: Path) -> None:
    summary = summarize_file(trajectory_file)

    assert summary.trajectories == 3
    assert summary.tokens == 3 * SEQ_LEN
    assert summary.model_tokens == 3 * MODEL_TOKENS_PER_TRAJ
    # Both model spans in this layout are critical spans.
    assert summary.critical_tokens == 3 * MODEL_TOKENS_PER_TRAJ
    assert summary.context_tokens == 3 * CONTEXT_TOKENS_PER_TRAJ
    assert summary.tokens == summary.model_tokens + summary.context_tokens
    assert summary.model_token_fraction == pytest.approx(MODEL_TOKENS_PER_TRAJ / SEQ_LEN)
    assert summary.policy_versions == [0, 1]
    assert summary.tokenizer_fingerprints == [FINGERPRINT]
    assert summary.model_ids == [MODEL_ID]
    assert len(summary.samples) == 3


def test_summarize_file_success_rate_only_counts_graded(trajectory_file: Path) -> None:
    summary = summarize_file(trajectory_file)

    # traj-2 carries no verification record, so it is not graded at all.
    assert summary.graded == 2
    assert summary.solved == 1
    assert summary.success_rate == pytest.approx(0.5)


def test_summarize_file_termination_reasons(trajectory_file: Path) -> None:
    summary = summarize_file(trajectory_file)

    assert summary.termination_reasons == {"final_answer": 2, "max_turns": 1}


def test_summarize_file_tokens_by_span_type(trajectory_file: Path) -> None:
    summary = summarize_file(trajectory_file)

    assert summary.tokens_by_span_type == {
        name: 3 * count for name, count in TOKENS_BY_SPAN_TYPE_PER_TRAJ.items()
    }


def test_summarize_file_tools_used_excludes_invalid_calls(trajectory_file: Path) -> None:
    summary = summarize_file(trajectory_file)

    # traj-1's only call is a parse failure, so "search" was never really used.
    assert summary.tools_used == {"calculator": 2}
    invalid = [s for s in summary.samples if s.invalid_tool_calls]
    assert [s.trajectory_id for s in invalid] == ["traj-1"]
    assert all(
        s.valid_tool_calls == (0 if s.trajectory_id == "traj-1" else 1) for s in summary.samples
    )


def test_provenance_check_never_calls_context_spans_trainable(trajectory_file: Path) -> None:
    """The headline provenance claim, read straight out of the summary payload."""
    payload = summarize_file(trajectory_file).to_dict()
    check = payload["provenance_check"]

    trainable = check["trainable_span_types"]
    excluded = check["context_span_types_excluded_from_loss"]

    assert trainable == sorted(s.value for s in MODEL_GENERATED_SPAN_TYPES)
    for context_type in ("tool_result", "system", "user"):
        assert context_type in excluded
        assert context_type not in trainable
    assert set(excluded).isdisjoint(trainable)
    assert check["critical_span_types"] == ["assistant_final", "assistant_tool_call"]


def test_summarize_file_filters_by_trajectory_id(trajectory_file: Path) -> None:
    summary = summarize_file(trajectory_file, trajectory_id="traj-1")

    assert summary.trajectories == 1
    assert [s.trajectory_id for s in summary.samples] == ["traj-1"]
    assert summary.tokens == SEQ_LEN
    assert summary.termination_reasons == {"max_turns": 1}
    assert summary.tools_used == {}
    assert summary.policy_versions == [1]
    assert summary.graded == 1
    assert summary.solved == 0
    assert summary.success_rate == pytest.approx(0.0)


def test_summarize_file_unknown_trajectory_id_yields_empty_summary(
    trajectory_file: Path,
) -> None:
    summary = summarize_file(trajectory_file, trajectory_id="does-not-exist")

    assert summary.trajectories == 0
    assert summary.tokens == 0
    assert summary.success_rate is None
    assert summary.model_token_fraction == 0.0


def test_summarize_file_honours_the_sample_limit(trajectory_file: Path) -> None:
    summary = summarize_file(trajectory_file, limit=1)

    assert summary.trajectories == 3
    assert len(summary.samples) == 1


# -- tamper detection (the point of the whole schema) ---------------------


def test_tampered_mask_cannot_smuggle_tool_output_into_the_loss(
    trajectory_file: Path,
) -> None:
    """Flipping one ``tool_result`` token to trainable must be rejected."""
    baseline = summarize_file(trajectory_file)
    assert baseline.model_tokens == 3 * MODEL_TOKENS_PER_TRAJ

    def flip_first_tool_result_token(payload: dict[str, Any]) -> None:
        tool_span = next(span for span in payload["spans"] if span["span_type"] == "tool_result")
        position = tool_span["start"]
        assert payload["model_generated_mask"][position] is False
        payload["model_generated_mask"][position] = True

    _tamper(trajectory_file, flip_first_tool_result_token)

    with pytest.raises(SchemaValidationError) as excinfo:
        summarize_file(trajectory_file)
    assert "model_generated_mask" in str(excinfo.value)


def test_tampered_critical_mask_is_rejected(trajectory_file: Path) -> None:
    def flip_first_system_token(payload: dict[str, Any]) -> None:
        payload["critical_mask"][0] = True

    _tamper(trajectory_file, flip_first_system_token)

    with pytest.raises(SchemaValidationError, match="critical_mask"):
        summarize_file(trajectory_file)


def test_wrong_schema_version_is_rejected(trajectory_file: Path) -> None:
    def bump_version(payload: dict[str, Any]) -> None:
        payload["schema_version"] = 99

    _tamper(trajectory_file, bump_version)

    with pytest.raises(SchemaValidationError) as excinfo:
        summarize_file(trajectory_file)
    message = str(excinfo.value)
    assert "schema_version" in message
    assert "99" in message


def test_missing_trajectory_file_is_reported_as_schema_error(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match="not found"):
        summarize_file(tmp_path / "nope.jsonl")


# -- span display rows ----------------------------------------------------


def test_iter_spans_for_display_marks_only_assistant_spans_in_loss(
    trajectory_file: Path,
) -> None:
    rows = list(iter_spans_for_display(trajectory_file, "traj-0"))

    assert [row["span_type"] for row in rows] == [t.value for t, _, _, _ in SPAN_LAYOUT]
    assert [row["in_loss"] for row in rows] == [False, False, True, False, True]
    assert [row["critical"] for row in rows] == [False, False, True, False, True]
    assert [(row["start"], row["end"]) for row in rows] == [
        (start, end) for _, start, end, _ in SPAN_LAYOUT
    ]
    assert [row["tokens"] for row in rows] == [end - start for _, start, end, _ in SPAN_LAYOUT]
    assert sum(row["tokens"] for row in rows) == SEQ_LEN
    assert [row["turn_id"] for row in rows] == [turn for _, _, _, turn in SPAN_LAYOUT]
    tool_names = {row["span_type"]: row["tool_name"] for row in rows}
    assert tool_names["assistant_tool_call"] == "calculator"
    assert tool_names["tool_result"] is None


def test_iter_spans_for_display_unknown_id_yields_nothing(trajectory_file: Path) -> None:
    assert list(iter_spans_for_display(trajectory_file, "traj-404")) == []


# -- ReportData -----------------------------------------------------------


def test_report_data_reads_the_run_directory(report_data: ReportData, synthetic_run: Path) -> None:
    assert report_data.run_id == RUN_ID
    assert report_data.run_dir == synthetic_run
    assert report_data.mode == "opd"
    assert report_data.is_on_policy is True
    assert report_data.manifest["seed"] == 1234
    assert report_data.environment["gpu"]["available"] is True
    assert "device: cpu" in report_data.resolved_config
    assert "device: auto" in report_data.original_config
    assert report_data.cache_stats is None
    assert report_data.benchmark is None


def test_report_data_partitions_metrics_by_phase(report_data: ReportData) -> None:
    assert [m["phase"] for m in report_data.step_metrics] == [
        "sft_warmup",
        "sft_warmup",
        "opd",
        "opd",
    ]
    assert [m["phase"] for m in report_data.cycle_metrics] == ["opd_cycle"]
    assert [m["global_step"] for m in report_data.eval_metrics] == [0, 4]
    assert len(report_data.events) == 4


def test_report_data_is_on_policy_is_false_for_non_opd_modes(synthetic_run: Path) -> None:
    manifest = _manifest()
    manifest["mode"] = "offline_kd"
    write_json(synthetic_run / "manifest.json", manifest)

    data = ReportData.from_run(synthetic_run)
    assert data.mode == "offline_kd"
    assert data.is_on_policy is False


def test_loss_series_is_grouped_by_phase(report_data: ReportData) -> None:
    series = report_data.loss_series()

    assert [name for name, _, _ in series] == ["opd", "sft_warmup"]
    by_name = {name: (xs, ys) for name, xs, ys in series}
    assert by_name["opd"] == ([2.0, 3.0], [1.5, 1.0])
    assert by_name["sft_warmup"] == ([0.0, 1.0], [2.5, 2.0])


def test_eval_series_tracks_success_rate(report_data: ReportData) -> None:
    series = report_data.eval_series()

    assert len(series) == 1
    name, xs, ys = series[0]
    assert name == "task success rate"
    assert xs == [0.0, 4.0]
    assert ys == [0.25, 0.5]


def test_failure_and_termination_counts_come_from_the_last_eval(
    report_data: ReportData,
) -> None:
    # The first eval record has an extra "no_final_answer" category; only the
    # most recent evaluation may be reported.
    assert report_data.failure_counts() == [("wrong_answer", 2.0)]
    assert report_data.termination_counts() == [("final_answer", 3.0), ("max_turns", 1.0)]


def test_selection_counts_come_from_the_last_cycle(report_data: ReportData) -> None:
    assert report_data.selection_counts() == [
        ("assistant_final", 10.0),
        ("assistant_tool_call", 8.0),
    ]
    # Context span types can never appear here.
    assert {name for name, _ in report_data.selection_counts()}.issubset(
        {s.value for s in MODEL_GENERATED_SPAN_TYPES}
    )


def test_baseline_comparison_orders_before_then_after(report_data: ReportData) -> None:
    rows = report_data.baseline_comparison()

    assert [row["label"] for row in rows] == ["before training", "after training"]
    assert [row["global_step"] for row in rows] == [0, 4]
    assert [row["success_rate"] for row in rows] == [0.25, 0.5]
    assert [row["policy_version"] for row in rows] == [0, 1]


def test_baseline_comparison_is_empty_without_eval_json(synthetic_run: Path) -> None:
    (synthetic_run / "eval.json").unlink()

    data = ReportData.from_run(synthetic_run)
    assert data.baseline_comparison() == []
    assert data.throughput()["wall_clock_seconds"] is None


def test_report_data_prefers_trajectories_with_token_analysis(
    report_data: ReportData,
) -> None:
    ids = [view.trajectory_id for view in report_data.trajectories]

    # The two trajectories with token analysis come first; the remaining display
    # budget is then filled rather than wasted.
    assert ids[:2] == ["traj-0", "traj-1"]
    assert set(ids) == {"traj-0", "traj-1", "traj-2"}
    assert set(report_data.token_analysis) == {"traj-0", "traj-1"}
    assert len(report_data.token_analysis["traj-0"]) == 3

    first = report_data.trajectories[0]
    assert first.tokens == SEQ_LEN
    assert first.model_tokens == MODEL_TOKENS_PER_TRAJ
    assert first.critical_tokens == MODEL_TOKENS_PER_TRAJ
    assert first.tokens_by_span_type == TOKENS_BY_SPAN_TYPE_PER_TRAJ
    assert first.solved is True
    assert first.termination_reason == "final_answer"
    assert [span["model_generated"] for span in first.spans] == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert report_data.trajectories[1].failure_category == "wrong_answer"


def test_token_analysis_respects_max_tokens(synthetic_run: Path) -> None:
    data = ReportData.from_run(synthetic_run, max_tokens=2)

    assert all(len(rows) == 2 for rows in data.token_analysis.values())


def test_max_trajectories_zero_loads_nothing(synthetic_run: Path) -> None:
    data = ReportData.from_run(synthetic_run, max_trajectories=0)

    assert data.trajectories == []


def test_stored_rollouts_are_displayed_without_token_analysis(synthetic_run: Path) -> None:
    (synthetic_run / "token_analysis.jsonl").unlink()
    assert not (synthetic_run / "eval_trajectories.jsonl").exists()

    data = ReportData.from_run(synthetic_run, max_trajectories=5)
    assert [view.trajectory_id for view in data.trajectories] == ["traj-0", "traj-1", "traj-2"]


def test_view_of_annotations_resolve() -> None:
    from typing import get_type_hints

    hints = get_type_hints(ReportData._view_of)
    assert "traj" in hints


# -- throughput: unmeasured must not look like zero -----------------------


def test_throughput_reports_rates_and_step_count(report_data: ReportData) -> None:
    throughput = report_data.throughput()

    assert throughput["optimizer_steps"] == 4
    assert throughput["train_selected_tokens_per_second_mean"] == pytest.approx(130.0)
    assert throughput["rollout_tokens_per_second_mean"] == pytest.approx(100.0)
    assert throughput["wall_clock_seconds"] == pytest.approx(123.5)


def test_cpu_run_on_a_gpu_box_reports_vram_as_not_measured(
    report_data: ReportData,
) -> None:
    """The manifest device decides, not whether the machine owns a GPU."""
    assert report_data.manifest["models"]["device"] == "cpu"
    assert report_data.environment["gpu"]["available"] is True
    # The metrics file does carry non-zero byte counts...
    assert report_data.step_metrics[0]["memory"]["peak_allocated_bytes"] > 0

    throughput = report_data.throughput()
    assert throughput["cuda_available"] is False
    assert throughput["peak_allocated_gib"] is None
    assert throughput["peak_reserved_gib"] is None


def test_cuda_run_on_a_gpu_box_reports_measured_vram(synthetic_run: Path) -> None:
    """The mirror image: with a cuda device the same bytes become GiB."""
    manifest = _manifest()
    manifest["models"]["device"] = "cuda:0"
    write_json(synthetic_run / "manifest.json", manifest)

    throughput = ReportData.from_run(synthetic_run).throughput()
    assert throughput["cuda_available"] is True
    assert throughput["peak_allocated_gib"] == pytest.approx(2.0)
    assert throughput["peak_reserved_gib"] == pytest.approx(3.0)


def test_cuda_device_without_a_visible_gpu_is_still_not_measured(
    synthetic_run: Path,
) -> None:
    manifest = _manifest()
    manifest["models"]["device"] = "cuda:0"
    write_json(synthetic_run / "manifest.json", manifest)
    environment = _environment()
    environment["gpu"]["available"] = False
    write_json(synthetic_run / "environment.json", environment)

    throughput = ReportData.from_run(synthetic_run).throughput()
    assert throughput["cuda_available"] is False
    assert throughput["peak_allocated_gib"] is None


# -- HTML report ----------------------------------------------------------


def test_render_html_is_self_contained_and_inert(report_data: ReportData) -> None:
    html = render_html(report_data)

    lowered = html.lower()
    assert "<script" not in lowered
    assert "javascript:" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert 'src="http' not in lowered
    assert 'href="http' not in lowered
    # The hostile span text and the hostile token piece both round-tripped, but
    # only as escaped entities -- one occurrence from the Jinja span table, two
    # from the hand-built token strips of traj-0 and traj-1.
    assert HOSTILE_TEXT not in html
    assert html.count("&lt;script&gt;") == 3


def test_render_html_contains_the_run_and_the_comparison(report_data: ReportData) -> None:
    html = render_html(report_data)

    assert RUN_ID in html
    assert "Baseline comparison" in html
    assert "No evaluation was recorded for this run." not in html
    assert "before training" in html
    assert "after training" in html
    # 0.5 success rate rendered as a percentage.
    assert "50.0%" in html


def test_render_html_labels_unmeasured_vram(report_data: ReportData) -> None:
    html = render_html(report_data)

    assert "n/a (no CUDA)" in html
    assert "0.000 GiB" not in html
    assert "No CUDA device was visible during this run" in html


def test_render_html_shows_context_spans_as_not_in_loss(report_data: ReportData) -> None:
    html = render_html(report_data)

    assert "no (context)" in html
    assert "tool_result" in html
    assert "traj-0" in html


def test_render_html_rejects_an_empty_manifest(synthetic_run: Path) -> None:
    from miniverl.errors import ReportError

    write_json(synthetic_run / "manifest.json", {})
    data = ReportData.from_run(synthetic_run)

    with pytest.raises(ReportError, match="empty manifest"):
        render_html(data)


# -- Markdown and JSON summaries -----------------------------------------


def test_render_markdown_has_the_run_and_a_results_row(report_data: ReportData) -> None:
    text = render_markdown(report_data)

    assert f"# miniVERL run `{RUN_ID}`" in text
    assert "| before training | 0 | 4 | 25.0% | 2.50 | 87.5% | 80.0% | 40.0 |" in text
    assert "| after training | 4 | 4 | 50.0% | 2.00 | 100.0% | 100.0% | 32.0 |" in text
    assert "_no evaluation recorded_" not in text
    assert "- mode: **opd**  (genuine on-policy distillation)" in text


def test_render_markdown_says_not_measured_for_vram(report_data: ReportData) -> None:
    text = render_markdown(report_data)

    assert "- peak CUDA allocated: not measured (no CUDA)" in text
    assert "- peak CUDA reserved: not measured (no CUDA)" in text
    assert "CPU only (no CUDA device visible)" in text
    assert "`wrong_answer`: 2" in text


def test_render_markdown_placeholder_row_without_eval(synthetic_run: Path) -> None:
    (synthetic_run / "eval.json").unlink()
    text = render_markdown(ReportData.from_run(synthetic_run))

    assert "| _no evaluation recorded_ | | | | | | | |" in text


def test_render_summary_json_top_level_keys(report_data: ReportData) -> None:
    payload = render_summary_json(report_data)

    assert set(payload) == {
        "run_id",
        "mode",
        "is_on_policy",
        "manifest",
        "comparison",
        "throughput",
        "cache",
        "failure_categories",
        "termination_reasons",
        "selected_by_span_type",
        "benchmark",
    }
    assert payload["run_id"] == RUN_ID
    assert payload["mode"] == "opd"
    assert payload["is_on_policy"] is True
    assert payload["failure_categories"] == {"wrong_answer": 2.0}
    assert payload["termination_reasons"] == {"final_answer": 3.0, "max_turns": 1.0}
    assert payload["selected_by_span_type"] == {
        "assistant_final": 10.0,
        "assistant_tool_call": 8.0,
    }
    assert payload["throughput"]["peak_allocated_gib"] is None
    assert payload["cache"] is None
    assert payload["benchmark"] is None
    # Must survive json.dumps with no custom encoder.
    assert json.loads(json.dumps(payload))["run_id"] == RUN_ID


@pytest.mark.parametrize(
    "private_path",
    [
        r"C:\Users\Alice\OneDrive\private\adapter",
        "/home/alice/private/adapter",
        "/Users/alice/private/adapter",
    ],
)
def test_shareable_report_formats_redact_private_paths_and_secrets(
    synthetic_run: Path,
    private_path: str,
) -> None:
    config = f"models:\n  teacher:\n    adapter:\n      path: {private_path!r}\n"
    (synthetic_run / "config.original.yaml").write_text(config, encoding="utf-8")
    (synthetic_run / "config.resolved.yaml").write_text(config, encoding="utf-8")
    manifest = _manifest()
    manifest["private_adapter_path"] = private_path
    manifest["api_token"] = "secret-token-value"
    manifest["diagnostic"] = (
        f"path={private_path} api_key=plain-text-secret Authorization: Bearer abcdefghijklmnop"
    )
    write_json(synthetic_run / "manifest.json", manifest)
    environment = _environment()
    environment["hostname"] = "alice-workstation"
    write_json(synthetic_run / "environment.json", environment)

    data = ReportData.from_run(synthetic_run)
    rendered = "\n".join(
        [
            render_html(data),
            render_markdown(data),
            json.dumps(render_summary_json(data), sort_keys=True),
        ]
    )
    variants = {
        private_path,
        private_path.replace("\\", "/"),
        private_path.replace("\\", "&#92;"),
        "secret-token-value",
        "plain-text-secret",
        "abcdefghijklmnop",
        "alice-workstation",
        str(synthetic_run),
    }
    lowered = rendered.lower()
    for value in variants:
        assert value.lower() not in lowered
    assert "alice" not in lowered
    assert "<local>/adapter" in rendered or "&lt;local&gt;/adapter" in rendered


def test_shareable_formats_redact_composite_credentials_and_paths(synthetic_run: Path) -> None:
    private_values = {
        "github_token": "github-unique-secret",
        "authorization": "Basic unique-auth-secret",
        "cookie": "session=unique-cookie-secret",
        "session": "unique-session-secret",
        "diagnostic": (
            r"failed at C:\Users\Alice Smith\OneDrive\private\adapter; "
            r"fallback \\server\share\private\adapter; "
            "source https://user:password@example.com/public/model"
        ),
    }
    manifest = _manifest()
    manifest["privacy_regression"] = private_values
    write_json(synthetic_run / "manifest.json", manifest)
    serialized = json.dumps({"privacy_regression": private_values}, indent=2)
    (synthetic_run / "config.original.yaml").write_text(serialized, encoding="utf-8")
    (synthetic_run / "config.resolved.yaml").write_text(serialized, encoding="utf-8")

    data = ReportData.from_run(synthetic_run)
    payloads = [
        render_html(data),
        render_markdown(data),
        json.dumps(render_summary_json(data), sort_keys=True),
    ]

    for rendered in payloads:
        for private in (
            "github-unique-secret",
            "unique-auth-secret",
            "unique-cookie-secret",
            "unique-session-secret",
            "Alice Smith",
            "user:password",
            "server",
            "share",
        ):
            assert private.lower() not in rendered.lower()


def test_every_shareable_report_view_redacts_remaining_cross_platform_sentinels(
    synthetic_run: Path,
) -> None:
    private_values = {
        "authorization_header": "auth-header-private-sentinel",
        "proxyAuthorization": "proxy-auth-private-sentinel",
        "set_cookie": "cookie-private-sentinel",
        "session_id": "session-private-sentinel",
        "client_secret_key": "client-private-sentinel",
        "auth_token_value": "token-private-sentinel",
        "diagnostic": (
            "linux=/mnt/data/Alice/private/model; workspace=/workspace/Alice/private/model; "
            "opt=/opt/project/private/file; "
            "ssh=ssh://user:password@example.com/repo; "
            "db=postgresql://user:password@example.com/database; "
            r"windows=C:\Users\Alice Smith\private\model; "
            r"unc=\\server\share\Alice\private\model"
        ),
        "tokenizer_id": "public/tokenizer",
        "token_count": 17,
        "session_count": 3,
    }
    manifest = _manifest()
    manifest["adversarial_privacy"] = private_values
    write_json(synthetic_run / "manifest.json", manifest)
    serialized = json.dumps({"adversarial_privacy": private_values}, indent=2)
    (synthetic_run / "config.original.yaml").write_text(serialized, encoding="utf-8")
    (synthetic_run / "config.resolved.yaml").write_text(serialized, encoding="utf-8")

    data = ReportData.from_run(synthetic_run)
    payloads = [
        render_html(data),
        render_markdown(data),
        json.dumps(render_summary_json(data), sort_keys=True),
        data.original_config,
        data.resolved_config,
        json.dumps(data.manifest, sort_keys=True),
    ]

    for rendered in payloads:
        for sentinel in (
            "auth-header-private-sentinel",
            "proxy-auth-private-sentinel",
            "cookie-private-sentinel",
            "session-private-sentinel",
            "client-private-sentinel",
            "token-private-sentinel",
            "Alice",
            "user:password",
            "server\\share",
        ):
            assert sentinel.lower() not in rendered.lower()
    assert data.manifest["adversarial_privacy"]["tokenizer_id"] == "public/tokenizer"
    assert data.manifest["adversarial_privacy"]["token_count"] == 17
    assert data.manifest["adversarial_privacy"]["session_count"] == 3


# -- charts ---------------------------------------------------------------


def test_line_chart_with_no_series_says_no_data() -> None:
    assert line_chart([]) == '<p class="muted">no data</p>'
    assert line_chart([("empty", [], [])]) == '<p class="muted">no data</p>'
    # A length mismatch is data corruption, not something to plot.
    assert line_chart([("ragged", [0.0, 1.0], [0.0])]) == '<p class="muted">no data</p>'


def test_line_chart_with_one_point_draws_a_circle() -> None:
    svg = line_chart([("single", [3.0], [0.25])], title="objective")

    assert "<circle" in svg
    assert "<polyline" not in svg
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")


def test_line_chart_with_several_points_draws_a_polyline() -> None:
    svg = line_chart([("many", [0.0, 1.0, 2.0], [1.0, 0.5, 0.25])])

    assert "<polyline" in svg
    assert "<circle" not in svg


def test_line_chart_escapes_series_names() -> None:
    svg = line_chart([(HOSTILE_TEXT, [0.0, 1.0], [1.0, 2.0])], title=HOSTILE_TEXT)

    assert HOSTILE_TEXT not in svg
    assert "&lt;script&gt;" in svg


def test_bar_chart_with_no_items_says_no_data() -> None:
    assert bar_chart([]) == '<p class="muted">no data</p>'
    # Entries whose value is None are dropped, which can empty the chart.
    assert bar_chart([("missing", None)]) == '<p class="muted">no data</p>'


def test_bar_chart_draws_one_rect_per_category() -> None:
    svg = bar_chart([("wrong_answer", 3.0), ("no_final_answer", 1.0)], title="failures")

    assert svg.count("<rect") == 2
    assert "wrong_answer" in svg
    assert "no_final_answer" in svg


def test_bar_chart_escapes_category_names() -> None:
    svg = bar_chart([(HOSTILE_TEXT, 1.0)])

    assert HOSTILE_TEXT not in svg
    assert "&lt;script&gt;" in svg


def test_token_strip_without_records_explains_itself() -> None:
    out = token_strip([])

    assert out == '<p class="muted">no token analysis recorded for this run</p>'
    assert "no token analysis" in out


def test_token_strip_escapes_token_pieces() -> None:
    out = token_strip(
        [
            {
                "trajectory_id": "traj-0",
                "target_position": 8,
                "span_type": "assistant_tool_call",
                "token_id": 108,
                "token_piece": HOSTILE_TEXT,
                "token_loss": 0.5,
                "weight": 1.0,
                "teacher_entropy": 0.25,
                "teacher_top_token": "42",
                "student_top_token": "41",
            }
        ]
    )

    assert HOSTILE_TEXT not in out
    assert "<script" not in out.lower()
    assert "&lt;script&gt;" in out
    # A teacher/student argmax disagreement is flagged.
    assert "tok-mismatch" in out


def test_token_strip_honours_max_tokens_and_span_classes() -> None:
    records = [
        {
            "span_type": "assistant_final",
            "token_piece": f"t{i}",
            "token_loss": 0.1 * i,
            "target_position": i,
            "teacher_top_token": "42",
            "student_top_token": "42",
        }
        for i in range(5)
    ]
    out = token_strip(records, max_tokens=2)

    assert out.count("<span") == 2
    assert "tok-final" in out
    assert "tok-mismatch" not in out


# -- run directory discovery ---------------------------------------------


def test_run_paths_open_rejects_a_directory_without_a_manifest(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-run"
    plain.mkdir()
    (plain / "metrics.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RunNotFoundError, match="does not look like a miniVERL run"):
        RunPaths.open(plain)


def test_run_paths_open_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(RunNotFoundError, match="run directory not found"):
        RunPaths.open(tmp_path / "absent")


def test_report_data_from_run_propagates_run_not_found(tmp_path: Path) -> None:
    plain = tmp_path / "empty"
    plain.mkdir()

    with pytest.raises(RunNotFoundError):
        ReportData.from_run(plain)


def test_run_paths_open_accepts_the_synthetic_run(synthetic_run: Path) -> None:
    paths = RunPaths.open(synthetic_run)

    assert paths.root == synthetic_run
    assert paths.manifest.is_file()
    assert paths.trajectories.is_file()
    assert paths.eval_json.is_file()
    # Never written by this fixture, so the report must degrade gracefully.
    assert not paths.eval_trajectories.exists()
    assert not paths.teacher_cache.exists()
    assert not paths.benchmark_json.exists()
