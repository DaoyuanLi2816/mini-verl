"""Self-contained offline reports built from run artifacts."""

from __future__ import annotations

from miniverl.reporting.data import ReportData, TrajectoryView
from miniverl.reporting.html import render_html, write_report
from miniverl.reporting.markdown import render_markdown, render_summary_json, write_markdown

__all__ = [
    "ReportData",
    "TrajectoryView",
    "render_html",
    "write_report",
    "render_markdown",
    "write_markdown",
    "render_summary_json",
]
