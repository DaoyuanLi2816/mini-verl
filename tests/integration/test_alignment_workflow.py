"""The alignment command publishes a complete, self-describing run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import requires_torch
from tests.integration.test_resume_and_swap import _config

pytestmark = [requires_torch, pytest.mark.torch]

pytest.importorskip("torch")


def test_alignment_workflow_writes_stages_metrics_card_and_manifest(tmp_path: Path) -> None:
    from miniverl.alignment import run_alignment

    config = _config(
        tmp_path,
        models={
            "student": {"toy": {"max_position_embeddings": 1024}},
            "teacher": {
                "mode": "privileged_context",
                "toy_pretrain_steps": 1,
                "toy": {"max_position_embeddings": 1024},
            },
        },
        environment={
            "name": "tool_policy",
            "params": {"prompt_style": "compact", "protocol_version": "v2"},
            "train_tasks": 12,
            "eval_tasks": 2,
            "test_tasks": 2,
        },
        train={"cycles": 1, "sft_warmup_cycles": 1},
        rollout={"max_total_tokens": 768},
        eval={"enabled": True, "baseline_enabled": True, "tasks": 2},
        alignment={
            "method": "standard_opd",
            "teacher_mode": "policy_conditioned",
            "policy": {"id": "minipolicy", "revision": "v1", "sha256": "a" * 64},
            "limitations": ["toy backend is a machinery harness"],
        },
    )
    payload = run_alignment(config, run_id="alignment-workflow")
    run = Path(payload["run_dir"])
    assert (run / "alignment-card.md").is_file()
    assert (run / "alignment-card.json").is_file()
    alignment = json.loads((run / "alignment.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert alignment["final_metrics"]["tasks"] == 2
    assert alignment["workflow"]["stages"][1]["source"] == "embedded_sft_warmup"
    assert manifest["status"] == "completed"
    assert manifest["alignment_result"]["sha256"]
    assert manifest["alignment_workflow"]["method"] == "standard_opd"


def test_verifier_gated_workflow_records_per_example_span_decisions(tmp_path: Path) -> None:
    from miniverl.alignment import run_alignment

    gate = {
        "version": "policy-span-v1",
        "signal": "policy_critical_span",
        "decision_scope": "span",
        "threshold": 1.0,
        "calibrated_on": "eval",
        "frozen_before_test": True,
    }
    config = _config(
        tmp_path,
        models={
            "student": {"toy": {"max_position_embeddings": 1024}},
            "teacher": {
                "mode": "privileged_context",
                "toy_pretrain_steps": 1,
                "toy": {"max_position_embeddings": 1024},
            },
        },
        environment={
            "name": "tool_policy",
            "params": {"prompt_style": "compact", "protocol_version": "v2"},
            "train_tasks": 12,
            "eval_tasks": 2,
            "test_tasks": 2,
        },
        selection={"selector": "verifier_gated", "gate": gate},
        train={"cycles": 1, "sft_warmup_cycles": 1},
        rollout={"max_total_tokens": 768},
        eval={"enabled": True, "baseline_enabled": True, "tasks": 2},
        alignment={
            "method": "verifier_gated_opd",
            "teacher_mode": "policy_conditioned",
            "policy": {"id": "minipolicy", "revision": "v1", "sha256": "a" * 64},
            "gate": gate,
        },
    )
    payload = run_alignment(config, run_id="alignment-gated")
    events = [
        json.loads(line)
        for line in (Path(payload["run_dir"]) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    decisions = [row for row in events if row["event"] == "alignment_gate_decision"]
    assert decisions
    assert all(row["trajectory_id"] and row["task_id"] for row in decisions)
    assert all(row["gate_version"] == "policy-span-v1" for row in decisions)
    assert all(isinstance(row["spans"], list) for row in decisions)
