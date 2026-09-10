"""Prompt reward validation is not a measured task-success endpoint."""

from pathlib import Path

from miniverl.reporting.data import ReportData


def test_unmeasured_success_is_neither_zero_nor_a_report_crash():
    data = ReportData("test", Path("run"), {}, {}, "", "", {})
    data.eval_metrics = [{"global_step": 0, "success_rate": None, "reward_mean": 0.3}]
    assert data.eval_series() == []
    data.eval_metrics += [{"global_step": 2, "success_rate": 0.0}]
    assert data.eval_series() == [("task success rate", [2.0], [0.0])]


def test_rl_report_is_on_policy():
    data = ReportData("test", Path("run"), {"mode": "rl"}, {}, "", "", {})
    assert data.is_on_policy is True
