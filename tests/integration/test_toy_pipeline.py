"""End-to-end toy pipeline: SFT, offline KD and genuine OPD.

These tests run the *real* trainer -- rollouts, tool execution, teacher scoring,
cache writes, masked updates -- at a size that finishes in seconds.  They are
what makes "the mode label means what it says" checkable: an OPD run must
increment its policy version every cycle and must never consume a target
produced under a different one, while an offline-KD run must reuse one frozen
cache and say so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]


def _config(tmp_path: Path, **overrides: Any):
    from miniverl.config import RunConfig

    payload: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "name": "toy-pipeline-test",
            "mode": "opd",
            "seed": 7,
            "output_dir": str(tmp_path),
            "deterministic": True,
        },
        "models": {
            "backend": "toy",
            "device": "cpu",
            "student": {
                "model_id": "toy-student",
                "lora": {"enabled": False},
                "toy": {
                    "hidden_size": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "intermediate_size": 64,
                    "max_position_embeddings": 512,
                },
            },
            "teacher": {
                "model_id": "toy-teacher",
                "toy_pretrain_steps": 3,
                "toy": {
                    "hidden_size": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "intermediate_size": 64,
                    "max_position_embeddings": 512,
                },
            },
        },
        "environment": {
            "name": "calculator",
            "difficulty": "easy",
            "params": {"prompt_style": "compact"},
            "train_tasks": 8,
            "eval_tasks": 2,
            "test_tasks": 2,
            "split_seed": 3,
        },
        "rollout": {
            "max_turns": 2,
            "max_new_tokens_per_turn": 12,
            "max_total_tokens": 420,
        },
        "selection": {"selector": "all_model_tokens"},
        "loss": {
            "mode": "bucketed_topk_tail",
            "divergence": "reverse_kl",
            "top_k": 8,
            "chunk_size": 32,
        },
        "train": {
            "cycles": 2,
            "rollouts_per_cycle": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.003,
            "sft_warmup_cycles": 0,
        },
        "memory": {"strategy": "resident"},
        "cache": {"entries_per_shard": 2, "dtype": "float32"},
        "eval": {"enabled": True, "tasks": 2, "temperature": 0.0},
        "report": {"enabled": True, "max_trajectories": 2, "max_tokens_per_trajectory": 64},
    }
    for section, values in overrides.items():
        if isinstance(values, dict):
            payload.setdefault(section, {})
            _deep_update(payload[section], values)
        else:
            payload[section] = values
    return RunConfig.model_validate(payload)


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _train(config, run_id: str):
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(config, run_id=run_id)
    try:
        result = trainer.train()
    finally:
        trainer.close()
    return trainer, result


# ------------------------------------------------------------- run modes


def test_opd_completes_and_produces_every_documented_artifact(tmp_path: Path):
    trainer, result = _train(_config(tmp_path), "opd-run")
    root = trainer.paths.root
    for name in (
        "config.original.yaml",
        "config.resolved.yaml",
        "manifest.json",
        "environment.json",
        "metrics.jsonl",
        "events.jsonl",
        "trajectories.jsonl",
        "eval.json",
    ):
        assert (root / name).is_file(), name
    assert (root / "teacher-cache" / "index.json").is_file()
    assert (root / "checkpoints" / "final").is_dir()
    assert result.mode == "opd"
    assert result.global_step == 2
    # One policy version per cycle: OPD resamples after every update.
    assert result.policy_version == 2
    assert result.eval is not None and result.baseline_eval is not None


def test_sft_completes_without_loading_a_teacher(tmp_path: Path):
    trainer, result = _train(_config(tmp_path, run={"mode": "sft"}), "sft-run")
    assert result.mode == "sft"
    assert trainer.teacher is None, "SFT must not load a teacher at all"
    assert trainer.scorer is None
    assert result.global_step == 2
    # No teacher means no teacher cache.
    index = trainer.paths.teacher_cache / "index.json"
    assert not index.is_file() or json.loads(index.read_text(encoding="utf-8"))["entries"] == {}


def test_offline_kd_reuses_one_frozen_cache_and_labels_itself(tmp_path: Path):
    config = _config(
        tmp_path,
        run={"mode": "offline_kd"},
        cache={"reuse_across_policy_versions": True, "strict_policy_version": False},
        train={"cycles": 3},
    )
    trainer, result = _train(config, "offline-run")
    assert result.mode == "offline_kd"
    events = [
        json.loads(line) for line in trainer.paths.events.read_text(encoding="utf-8").splitlines()
    ]
    reuse = [e for e in events if e["event"] == "offline_kd_reuse"]
    assert reuse, "offline KD must announce that it is reusing fixed targets"
    assert "not on-policy" in reuse[0]["note"]
    # Targets were scored once, for one policy version only.
    from miniverl.cache.store import TeacherCache

    cache = TeacherCache.open(trainer.paths.teacher_cache)
    assert len(cache.index.policy_versions()) == 1


def test_opd_and_offline_kd_differ_in_their_cache_policy_versions(tmp_path: Path):
    from miniverl.cache.store import TeacherCache

    opd_trainer, _ = _train(
        _config(tmp_path, train={"cycles": 3}, cache={"keep_cycles": 10}), "opd-v"
    )
    opd_versions = TeacherCache.open(opd_trainer.paths.teacher_cache).index.policy_versions()
    assert len(opd_versions) == 3, "each OPD cycle must score under a fresh policy version"


def test_exact_full_vocab_mode_runs_and_writes_no_cache(tmp_path: Path):
    """A resident exact teacher rebuilds distributions per chunk instead of storing them."""
    config = _config(
        tmp_path,
        loss={"mode": "exact_full_vocab", "top_k": 1, "chunk_size": 32},
        memory={"strategy": "resident"},
    )
    trainer, result = _train(config, "exact-run")
    assert result.global_step == 2
    index = trainer.paths.teacher_cache / "index.json"
    if index.is_file():
        assert json.loads(index.read_text(encoding="utf-8"))["entries"] == {}


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "jsd"])
def test_every_divergence_trains(tmp_path: Path, divergence: str):
    _, result = _train(_config(tmp_path, loss={"divergence": divergence}), f"div-{divergence}")
    assert result.global_step == 2


@pytest.mark.parametrize("selector", ["all_model_tokens", "uniform_ratio", "hybrid"])
def test_every_selector_trains(tmp_path: Path, selector: str):
    _, result = _train(
        _config(tmp_path, selection={"selector": selector, "ratio": 0.5}), f"sel-{selector}"
    )
    assert result.global_step >= 1


def test_a_selector_that_finds_nothing_says_so_instead_of_doing_nothing(tmp_path: Path):
    """`tool_and_final` on an untrained policy selects zero tokens.

    A random toy student emits no parseable tool call or final answer, so there
    are no *critical* tokens to supervise. The run must complete with zero
    optimizer steps AND record why, rather than looking like a normal cycle.
    """
    trainer, result = _train(
        _config(tmp_path, selection={"selector": "tool_and_final"}), "sel-empty"
    )
    assert result.global_step == 0
    events = [
        json.loads(line) for line in trainer.paths.events.read_text(encoding="utf-8").splitlines()
    ]
    skipped = [e for e in events if e["event"] == "cycle_skipped_no_selected_positions"]
    assert skipped, "an empty selection must be announced"
    assert skipped[0]["selector"] == "tool_and_final"
    assert "zero optimizer steps" in skipped[0]["note"]


def test_sft_warmup_then_opd(tmp_path: Path):
    trainer, result = _train(
        _config(tmp_path, train={"sft_warmup_cycles": 2, "cycles": 2}), "warmup-run"
    )
    metrics = [
        json.loads(line) for line in trainer.paths.metrics.read_text(encoding="utf-8").splitlines()
    ]
    phases = {m.get("phase") for m in metrics}
    assert "sft_warmup" in phases
    assert "opd" in phases
    assert result.global_step == 4


# --------------------------------------------------- provenance in practice


def test_no_stored_trajectory_ever_marks_tool_output_trainable(tmp_path: Path):
    """The end-to-end version of the provenance guarantee."""
    from miniverl.schemas.trajectory import MODEL_GENERATED_SPAN_TYPES
    from miniverl.trajectory.io import read_trajectories

    trainer, _ = _train(_config(tmp_path), "provenance-run")
    trajectories = read_trajectories(trainer.paths.trajectories)
    assert trajectories
    saw_tool_result = False
    for traj in trajectories:
        for span in traj.spans:
            trainable = any(traj.model_generated_mask[span.start : span.end])
            if span.span_type in MODEL_GENERATED_SPAN_TYPES:
                assert trainable
            else:
                assert not trainable, f"{span.span_type} was marked trainable"
            if span.span_type.value == "tool_result":
                saw_tool_result = True
    assert saw_tool_result, "the fixture must actually contain tool output to be meaningful"


def test_selected_positions_are_always_model_tokens(tmp_path: Path):
    from miniverl.config.models import SelectionConfig
    from miniverl.selection.selectors import select_positions
    from miniverl.trajectory.io import read_trajectories

    trainer, _ = _train(_config(tmp_path), "selection-run")
    for traj in read_trajectories(trainer.paths.trajectories):
        for selector in ("all_model_tokens", "tool_and_final", "uniform_ratio", "hybrid"):
            result = select_positions(
                traj, SelectionConfig(selector=selector, ratio=0.5), run_seed=7
            )
            for position in result.positions:
                assert traj.model_generated_mask[position]


def test_cache_entries_carry_full_provenance(tmp_path: Path):
    from miniverl.cache.store import TeacherCache

    trainer, _ = _train(_config(tmp_path), "cache-provenance")
    cache = TeacherCache.open(trainer.paths.teacher_cache)
    assert cache.index.tokenizer_fingerprint == trainer.tokenizer.fingerprint
    assert cache.index.teacher_model_id == "toy-teacher"
    assert cache.index.top_k == 8
    assert cache.index.loss_mode == "bucketed_topk_tail"
    assert cache.validate() == []
    stats = cache.stats()
    assert stats.num_selected_positions > 0
    assert stats.theoretical_full_logit_bytes > 0


def test_manifest_records_provenance_and_no_personal_data(tmp_path: Path):
    import getpass
    import platform

    trainer, _ = _train(_config(tmp_path), "manifest-run")
    manifest = json.loads(trainer.paths.manifest.read_text(encoding="utf-8"))
    for key in (
        "miniverl_version",
        "run_id",
        "created_at",
        "python_version",
        "os",
        "packages",
        "gpu",
        "mode",
        "seed",
        "models",
        "objective",
        "memory",
        "measurement_status",
    ):
        assert key in manifest, key
    assert manifest["models"]["tokenizer_fingerprint"]
    blob = json.dumps(manifest)
    assert platform.node() not in blob
    assert getpass.getuser() not in blob
    assert str(Path.home()) not in blob


def test_report_renders_from_a_real_run(tmp_path: Path):
    from miniverl.reporting import ReportData, render_markdown, write_report

    trainer, _ = _train(_config(tmp_path), "report-run")
    html_path = write_report(trainer.paths.root, trainer.paths.report_html)
    html = html_path.read_text(encoding="utf-8")
    assert "miniVERL run" in html
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html.lower()
    data = ReportData.from_run(trainer.paths.root)
    assert data.is_on_policy
    markdown = render_markdown(data)
    assert "genuine on-policy distillation" in markdown
    assert "not measured (no CUDA)" in markdown  # the toy run is CPU-only


def test_eval_is_deterministic_under_greedy_decoding(tmp_path: Path):
    trainer, _ = _train(_config(tmp_path), "determinism-run")
    first = trainer.evaluate(tag="repeat-a")
    second = trainer.evaluate(tag="repeat-b")
    assert first["success_rate"] == second["success_rate"]
    assert first["generated_tokens"] == second["generated_tokens"]
    assert first["termination_reasons"] == second["termination_reasons"]
    trainer.close()
