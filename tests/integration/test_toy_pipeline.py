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
        "config.validated.yaml",
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


def test_eval_disabled_suppresses_periodic_evaluations(tmp_path: Path):
    trainer, result = _train(
        _config(
            tmp_path,
            run={"mode": "sft"},
            train={"cycles": 2, "eval_every_cycles": 1},
            eval={"enabled": False},
        ),
        "no-periodic-eval",
    )
    metrics = [
        json.loads(line) for line in trainer.paths.metrics.read_text(encoding="utf-8").splitlines()
    ]
    assert result.baseline_eval is None
    assert result.eval is None
    assert not [record for record in metrics if record.get("phase") == "eval"]


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


def test_frozen_student_offline_kd_collects_once_without_parameter_updates(
    tmp_path: Path,
) -> None:
    from miniverl.trainer import OPDTrainer
    from miniverl.trajectory.io import read_trajectories

    config = _config(
        tmp_path,
        run={"mode": "offline_kd"},
        cache={"reuse_across_policy_versions": True, "strict_policy_version": False},
        offline_kd={
            "trajectory_source": "frozen_student",
            "collection_seed": 20260801,
            "collection_tasks": 4,
        },
        train={"cycles": 2, "rollouts_per_cycle": 2, "gradient_accumulation_steps": 2},
        eval={"enabled": False},
    )
    trainer = OPDTrainer.from_config(config, run_id="frozen-student-offline")
    trainer.set_offline_collection_checkpoint_digest("a" * 64)
    result = trainer.train()

    manifest = json.loads(trainer.paths.offline_dataset_manifest.read_text(encoding="utf-8"))
    trajectories = read_trajectories(trainer.paths.offline_dataset_trajectories)
    assert result.global_step == 2
    assert manifest["schema_version"] == 2
    assert manifest["source"] == "frozen_student"
    assert manifest["cold_start_checkpoint_digest"] == "a" * 64
    assert manifest["parameter_version"] == 0
    assert len(trajectories) == 4
    assert all(":oracle:" not in trajectory.trajectory_id for trajectory in trajectories)
    assert [trajectory.task_id for trajectory in trajectories] == manifest["task_ids"]
    assert manifest["task_schedule_digest"]
    assert manifest["generation_seeds"] == [20260801, 20260802, 20260803, 20260804]
    trainer.close()


def test_persisted_offline_kd_reuses_the_prepared_bundle_without_mutating_it(
    tmp_path: Path,
) -> None:
    from miniverl.trainer import OPDTrainer

    common = {
        "run": {"mode": "offline_kd"},
        "cache": {"reuse_across_policy_versions": True, "strict_policy_version": False},
        "train": {"cycles": 2, "rollouts_per_cycle": 2, "gradient_accumulation_steps": 2},
        "eval": {"enabled": False},
    }
    prepared = OPDTrainer.from_config(
        _config(
            tmp_path,
            **common,
            offline_kd={
                "trajectory_source": "frozen_student",
                "collection_seed": 17,
                "collection_tasks": 4,
            },
        ),
        run_id="prepared-bundle",
    )
    prepared.set_offline_collection_checkpoint_digest("b" * 64)
    summary = prepared.prepare_offline_dataset()
    source_root = prepared.paths.root
    prepared.close()
    source_bytes = {
        path.relative_to(source_root): path.read_bytes()
        for artifact in (source_root / "offline-dataset", source_root / "teacher-cache")
        for path in artifact.rglob("*")
        if path.is_file()
    }

    consumer = OPDTrainer.from_config(
        _config(
            tmp_path,
            **common,
            offline_kd={
                "trajectory_source": "persisted",
                "dataset_path": str(source_root),
                "task_schedule_digest": summary["manifest"]["task_schedule_digest"],
            },
        ),
        run_id="persisted-consumer",
    )
    result = consumer.train()

    assert result.global_step == 2
    assert consumer.offline_dataset_digest == summary["dataset_digest"]
    assert consumer._offline_samples is not None
    assert len(consumer._offline_samples) == 4
    assert all(
        (source_root / path).read_bytes() == contents for path, contents in source_bytes.items()
    )
    consumer.close()


def test_selected_token_budget_stops_only_after_an_optimizer_step_and_records_overshoot(
    tmp_path: Path,
) -> None:
    trainer, result = _train(
        _config(
            tmp_path,
            run={"mode": "sft"},
            train={"cycles": 10, "max_selected_training_tokens": 1},
            eval={"enabled": False},
        ),
        "selected-token-stop",
    )

    assert result.global_step == 1
    assert result.cycles_completed == 1
    assert result.stop_criterion == {
        "kind": "selected_training_tokens",
        "target": 1,
        "actual": result.overshoot["actual"],
    }
    assert result.overshoot["value"] == result.overshoot["actual"] - 1
    assert result.overshoot["value"] >= 0
    manifest = json.loads(trainer.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["result"]["stop_criterion"] == result.stop_criterion


def test_wall_budget_stops_after_a_complete_optimizer_step(tmp_path: Path) -> None:
    _trainer, result = _train(
        _config(
            tmp_path,
            run={"mode": "sft"},
            train={"cycles": 10, "max_wall_seconds": 1.0e-9},
            eval={"enabled": False},
        ),
        "wall-stop",
    )
    assert result.global_step == 1
    assert result.cycles_completed == 1
    assert result.stop_criterion["kind"] == "wall_seconds"
    assert result.overshoot["value"] >= 0.0


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
    from miniverl.trainer import OPDTrainer

    config = _config(tmp_path, selection={"selector": "tool_and_final"})
    trainer, result = _train(config, "sel-empty")
    assert result.global_step == 0
    events = [
        json.loads(line) for line in trainer.paths.events.read_text(encoding="utf-8").splitlines()
    ]
    skipped = [e for e in events if e["event"] == "cycle_skipped_no_selected_positions"]
    assert skipped, "an empty selection must be announced"
    assert skipped[0]["selector"] == "tool_and_final"
    assert "zero optimizer steps" in skipped[0]["note"]
    assert result.parameter_version == 0
    assert result.policy_version == 0
    assert result.global_step == 0

    resumed = OPDTrainer.from_config(config, run_id="sel-empty-resumed")
    state = resumed.load_from_checkpoint(trainer.paths.checkpoints / "final")
    assert state.global_step == 0
    assert state.parameter_version == 0
    resumed_result = resumed.train()
    assert resumed_result.global_step == 0
    assert resumed_result.parameter_version == 0
    resumed.close()


def test_replay_records_one_rollout_version_and_each_successful_parameter_version(
    tmp_path: Path,
):
    """Two optimizer groups consume one rollout batch without relabelling it."""
    trainer, result = _train(
        _config(
            tmp_path,
            train={
                "cycles": 1,
                "rollouts_per_cycle": 4,
                "gradient_accumulation_steps": 2,
                "opd_freshness": "replay",
            },
            eval={"enabled": False},
            report={"enabled": False},
        ),
        "version-replay",
    )
    metrics = [
        json.loads(line) for line in trainer.paths.metrics.read_text(encoding="utf-8").splitlines()
    ]
    updates = [record for record in metrics if record.get("phase") == "opd"]

    assert result.global_step == 2
    assert result.parameter_version == 2
    assert result.policy_version == 2
    assert [record["global_optimizer_step"] for record in updates] == [1, 2]
    assert [record["parameter_version"] for record in updates] == [1, 2]
    assert {record["rollout_policy_version"] for record in updates} == {0}
    assert result.rollout_policy_version == 0


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
    from miniverl.trainer import OPDTrainer

    with OPDTrainer.from_config(_config(tmp_path), run_id="determinism-run") as trainer:
        trainer.train()
        first = trainer.evaluate(tag="repeat-a")
        second = trainer.evaluate(tag="repeat-b")
        assert first["success_rate"] == second["success_rate"]
        assert first["generated_tokens"] == second["generated_tokens"]
        assert first["termination_reasons"] == second["termination_reasons"]


def test_a_raising_tool_ends_the_episode_with_its_own_termination_reason(tmp_path: Path):
    """An environment that raises is an environment defect, not a policy mistake.

    ``TerminationReason.ENVIRONMENT_ERROR`` exists for exactly this case. Without
    it a broken tool would either crash a training run or be recorded as the
    model's failure, which would poison the failure taxonomy.

    The *oracle* path deliberately does not catch this: an oracle that cannot run
    its own reference actions is a bug in the environment and must fail loudly.
    """
    from miniverl.agent.loop import RolloutRunner
    from miniverl.agent.protocol import render_tool_call
    from miniverl.config.models import RolloutConfig
    from miniverl.environments import make_environment, make_splits
    from miniverl.environments.base import StepResult, ToolCall
    from miniverl.errors import ToolEnvironmentError
    from miniverl.models.base import GenerationOutput
    from miniverl.models.tokenizers import ToyTokenizer
    from miniverl.models.toy import ToyBackend
    from miniverl.schemas.trajectory import TerminationReason

    calculator_cls = type(make_environment("calculator"))

    class ExplodingCalculator(calculator_cls):  # type: ignore[misc, valid-type]
        """A calculator whose tool raises instead of returning a StepResult."""

        def step(self, call: ToolCall) -> StepResult:
            raise ToolEnvironmentError("the tool backend fell over")

    tokenizer = ToyTokenizer()

    class ScriptedBackend(ToyBackend):
        """A policy that always emits one well-formed tool call."""

        def generate(self, prefix_token_ids, **kwargs) -> GenerationOutput:
            text = render_tool_call("calculator", {"expression": "1 + 1"})
            ids = tokenizer.encode(text)
            return GenerationOutput(
                token_ids=ids,
                text=text,
                stop_reason="stop_sequence",
                matched_stop="</tool_call>",
            )

    backend = ScriptedBackend(
        tokenizer=tokenizer, model_id="scripted", seed=0, hidden_size=32, num_layers=2
    )
    environment = ExplodingCalculator(prompt_style="compact")
    task = make_splits(
        environment, counts={"train": 1, "eval": 0, "test": 0}, seed=1, difficulty="easy"
    )["train"][0]
    runner = RolloutRunner(
        backend=backend,
        environment=environment,
        config=RolloutConfig(max_turns=3, max_new_tokens_per_turn=8, max_total_tokens=400),
    )

    traj = runner.rollout(task, policy_version=0, seed=1)
    assert traj.termination_reason is TerminationReason.ENVIRONMENT_ERROR
    assert "fell over" in str(traj.metadata.get("environment_error", ""))
    # The trajectory is still structurally valid and still masks tool output.
    assert traj.length > 0
    for span in traj.spans:
        if span.span_type.value == "tool_result":
            assert not any(traj.model_generated_mask[span.start : span.end])

    # The oracle path fails loudly instead.
    with pytest.raises(ToolEnvironmentError, match="fell over"):
        runner.oracle_rollout(task)


def test_every_json_artifact_of_a_run_is_strictly_valid_json(tmp_path: Path):
    """Artifacts are data files, so a non-Python reader must be able to load them.

    Python's ``json`` module both writes and accepts the non-standard ``NaN``
    token, so a run could -- and did -- emit a ``metrics.jsonl`` that
    ``JSON.parse`` and most non-Python parsers reject. Any quantity that can be
    undefined has to be ``null`` instead, which is what
    :func:`miniverl.evaluation.schema.finite_or_none` is for.

    The toy student solves nothing at this budget, so the divide-by-zero in
    ``tokens_per_solved_task`` is exercised rather than hypothetical.
    """
    import json

    from miniverl.training.trainer import OPDTrainer

    def strict(text: str, where: str) -> object:
        def reject(constant: str) -> float:
            raise AssertionError(f"{where} contains the non-standard JSON token {constant!r}")

        return json.loads(text, parse_constant=reject)

    config = _config(
        tmp_path,
        train={"cycles": 1, "sft_warmup_cycles": 1, "rollouts_per_cycle": 2},
        eval={"enabled": True, "tasks": 4},
    )
    result = OPDTrainer.from_config(config, run_id="strict-json").train()

    checked = 0
    for path in sorted(Path(result.run_dir).rglob("*.json")):
        strict(path.read_text(encoding="utf-8"), path.name)
        checked += 1
    metrics = []
    for path in sorted(Path(result.run_dir).rglob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                record = strict(line, f"{path.name}:{number}")
                if path.name == "metrics.jsonl":
                    metrics.append(record)
        checked += 1
    assert checked >= 4, f"expected several JSON artifacts, found {checked}"

    # And the undefined case really did occur, so the test is not vacuous.
    def rollout_blocks():
        for record in metrics:
            block = record.get("rollouts") if isinstance(record, dict) else None
            if isinstance(block, dict) and "tokens_per_solved_task" in block:
                yield block

    blocks = list(rollout_blocks())
    assert blocks, "no rollout metrics were written, so this test would prove nothing"
    assert any(b["tokens_per_solved_task"] is None for b in blocks), (
        "expected a cycle that solved nothing, which is the null path this test guards"
    )


def test_a_run_writes_lf_line_endings_on_every_platform(tmp_path: Path):
    """A run directory must be a function of the computation, not the OS.

    ``Path.write_text`` translates ``\n`` to the platform separator, so the same
    run produced CRLF artifacts on Windows and LF ones on Linux. That is not
    cosmetic: the teacher-cache index and the checkpoint state carry checksums,
    and CI regenerates the benchmark JSON Schema and byte-diffs it against the
    committed copy -- a check that can never pass if the two platforms disagree
    about newlines. Every artifact now goes through
    :func:`miniverl.utils.runs.write_text`.

    This assertion only has teeth on Windows, where the translation happens.
    """
    from miniverl.training.trainer import OPDTrainer

    config = _config(
        tmp_path,
        train={"cycles": 1, "sft_warmup_cycles": 1, "rollouts_per_cycle": 2},
        eval={"enabled": True, "tasks": 4},
    )
    result = OPDTrainer.from_config(config, run_id="lf-endings").train()

    checked, offenders = 0, []
    for path in sorted(Path(result.run_dir).rglob("*")):
        if not path.is_file() or path.suffix == ".safetensors":
            continue
        blob = path.read_bytes()
        if b"\x00" in blob[:1024]:
            continue
        checked += 1
        if b"\r\n" in blob:
            offenders.append(
                f"{path.relative_to(result.run_dir)} ({blob.count(bytes([13, 10]))} CRLF)"
            )
    assert checked >= 6, f"expected several text artifacts, found {checked}"
    assert not offenders, f"artifacts written with CRLF: {offenders}"


def test_both_schema_output_paths_produce_identical_bytes(tmp_path: Path):
    """``miniverl schema`` and ``miniverl schema --out`` must agree exactly.

    They did not: ``--out`` went through the canonical writer (sorted keys, LF,
    trailing newline) while stdout went through Rich, which preserves insertion
    order, and on Windows text-mode stdout re-expanded newlines to CRLF. CI
    regenerates the schema and byte-diffs it against the committed copy, so a
    contributor who used the other path could not reproduce the committed file.
    """
    import subprocess
    import sys

    out_path = tmp_path / "via-out.json"
    subprocess.run(
        [sys.executable, "-m", "miniverl.cli", "schema", "--out", str(out_path)],
        check=True,
        capture_output=True,
    )
    piped = subprocess.run(
        [sys.executable, "-m", "miniverl.cli", "schema"],
        check=True,
        capture_output=True,
    )
    assert out_path.read_bytes() == piped.stdout, (
        "the two schema output paths disagree; CI byte-diffs this file"
    )
    assert b"\r\n" not in piped.stdout, "stdout re-expanded newlines to CRLF"

    committed = (
        Path(__file__).resolve().parents[2] / "benchmarks/schema/benchmark-result.schema.json"
    )
    assert committed.read_bytes() == out_path.read_bytes(), (
        "the committed schema is stale: run `miniverl schema --out benchmarks/schema/"
        "benchmark-result.schema.json`"
    )
