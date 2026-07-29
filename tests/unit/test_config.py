"""The recipe contract: a :class:`RunConfig` either describes a runnable experiment or refuses
to exist.

This file protects three things that everything downstream assumes:

* **Every shipped recipe parses.** ``recipes/*.yaml`` is the documentation, so a recipe that no
  longer validates -- or that validates but silently drops a key -- is a broken promise. The
  leaf-by-leaf comparison against the raw YAML is what catches a renamed field.
* **Malformed input fails as a :class:`~miniverl.errors.ConfigError`, not a traceback.** Missing
  file, unparseable YAML, empty file, top-level list.
* **Contradictions are rejected at parse time.** Every rule in
  ``RunConfig._validate_combination`` is exercised, each asserting on the message the user will
  actually read, plus the neighbouring accepted case so the test cannot pass by rejecting
  everything.

``recipes/`` holds two kinds of file: run recipes (:class:`RunConfig`) and matched-budget
benchmark specifications (:class:`~miniverl.evaluation.schema.BenchmarkConfig`, identified by an
``arms`` key). Both kinds are covered, and the partition is asserted to be exhaustive so a new
file cannot escape coverage.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from miniverl.config.models import (
    CONFIG_SCHEMA_VERSION,
    AdapterSource,
    Divergence,
    LossMode,
    LRSchedule,
    MemoryStrategy,
    ModelBackend,
    OPDFreshness,
    OptimizerName,
    Precision,
    Quantization,
    RunConfig,
    SelectorName,
    TeacherContextMode,
    ToyModelConfig,
    TrainingMode,
)
from miniverl.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES_DIR = REPO_ROOT / "recipes"


def _read_raw(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


ALL_RECIPES = sorted(RECIPES_DIR.glob("*.yaml"))
RUN_RECIPES = [p for p in ALL_RECIPES if "arms" not in _read_raw(p)]
BENCHMARK_RECIPES = [p for p in ALL_RECIPES if "arms" in _read_raw(p)]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _payload(**sections: Any) -> dict[str, Any]:
    """A minimal *valid* OPD payload with ``sections`` deep-merged on top."""
    minimal: dict[str, Any] = {
        "models": {
            "student": {"model_id": "toy-student"},
            "teacher": {"model_id": "toy-teacher"},
        },
        "environment": {"name": "calculator"},
    }
    return _deep_merge(minimal, sections)


def _rejects(payload: dict[str, Any], expected: str) -> None:
    """Assert ``payload`` is rejected and that the user-visible text explains why."""
    with pytest.raises(ValidationError) as excinfo:
        RunConfig.from_mapping(payload)
    rendered = str(excinfo.value)
    assert expected in rendered, f"expected {expected!r} in:\n{rendered}"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    target = tmp_path / name
    target.write_text(text, encoding="utf-8")
    return target


def _assert_same_leaves(raw: dict[str, Any], dumped: dict[str, Any], prefix: str = "") -> None:
    """Every leaf written in the recipe must survive parsing unchanged."""
    for key, expected in raw.items():
        assert key in dumped, f"{prefix}{key} is absent from the parsed config"
        actual = dumped[key]
        if isinstance(expected, dict) and isinstance(actual, dict):
            _assert_same_leaves(expected, actual, f"{prefix}{key}.")
        else:
            assert actual == expected, f"{prefix}{key}: parsed {actual!r}, recipe says {expected!r}"


# -- the shipped recipes -------------------------------------------------------


def test_recipes_directory_is_partitioned_into_run_and_benchmark_files() -> None:
    """The parametrized suites below must jointly cover every file in recipes/."""
    assert ALL_RECIPES, f"no recipes found under {RECIPES_DIR}"
    assert RUN_RECIPES, "expected at least one RunConfig recipe"
    assert BENCHMARK_RECIPES, "expected at least one benchmark specification"
    assert sorted(RUN_RECIPES + BENCHMARK_RECIPES) == ALL_RECIPES


@pytest.mark.parametrize("path", RUN_RECIPES, ids=lambda p: p.name)
def test_every_run_recipe_validates(path: Path) -> None:
    config = RunConfig.from_yaml(path)
    assert isinstance(config, RunConfig)
    assert config.schema_version == CONFIG_SCHEMA_VERSION
    assert config.models.student.model_id
    assert config.models.teacher.model_id
    assert config.is_on_policy is (config.run.mode is TrainingMode.OPD)


@pytest.mark.parametrize("path", RUN_RECIPES, ids=lambda p: p.name)
def test_run_recipe_fields_match_the_yaml_leaf_for_leaf(path: Path) -> None:
    config = RunConfig.from_yaml(path)
    _assert_same_leaves(_read_raw(path), config.model_dump(mode="json"))


def test_single_gpu_quickstart_is_portable_and_uses_protocol_teacher() -> None:
    supported = RunConfig.from_yaml(RECIPES_DIR / "qwen_consumer_gpu_calc.yaml")
    raw_control = RunConfig.from_yaml(RECIPES_DIR / "qwen_consumer_gpu_calc_raw_teacher.yaml")

    adapter = supported.models.teacher.adapter
    assert adapter is not None
    assert adapter.source is AdapterSource.HUB
    assert adapter.path == "DaoyuanLi/mini-verl-qwen3-1.7b-protocol-teacher"
    assert adapter.revision == "23323751318135484c06c043b1f9b9e7016dd89f"
    assert adapter.require_policy_evaluation is True
    assert adapter.minimum_strict_success_rate == pytest.approx(0.5)
    assert supported.models.device == "auto"
    assert supported.models.student.dtype is Precision.AUTO
    assert supported.models.teacher.dtype is Precision.AUTO
    assert supported.run.name == "qwen3-calc-opd-single-gpu"
    assert "single-gpu" in supported.run.tags
    assert not {"rtx4080", "16gb"} & set(supported.run.tags)

    assert supported.models.student.model_id == raw_control.models.student.model_id
    assert supported.models.teacher.model_id == raw_control.models.teacher.model_id
    assert raw_control.models.teacher.adapter is None


def test_toy_cpu_recipe_is_parsed_into_the_expected_typed_values() -> None:
    # Budgets are tuning knobs; assert them against the YAML rather than
    # hard-coding, so retuning a recipe does not require editing this test.
    import yaml as _yaml

    raw = _yaml.safe_load((RECIPES_DIR / "toy_cpu.yaml").read_text(encoding="utf-8"))
    config = RunConfig.from_yaml(RECIPES_DIR / "toy_cpu.yaml")

    assert config.schema_version == 1
    assert config.run.name == "toy-cpu-opd"
    assert config.run.mode is TrainingMode.OPD
    assert config.run.seed == 1234
    assert config.run.output_dir == "runs"
    assert config.run.deterministic is True
    assert config.run.run_id is None
    assert config.run.tags == []

    assert config.models.backend is ModelBackend.TOY
    assert config.models.device == "cpu"
    student = config.models.student
    assert student.model_id == "toy-student"
    assert student.dtype is Precision.FLOAT32
    assert student.quantization is Quantization.NONE
    assert student.lora.enabled is False
    assert (student.toy.hidden_size, student.toy.num_layers, student.toy.num_heads) == (96, 3, 4)
    assert student.toy.intermediate_size == 192
    assert student.toy.max_position_embeddings == 1024
    teacher = config.models.teacher
    assert teacher.model_id == "toy-teacher"
    assert teacher.mode is TeacherContextMode.STANDARD
    assert teacher.toy_teacher_seed == 99
    assert teacher.toy_pretrain_steps == raw["models"]["teacher"]["toy_pretrain_steps"]
    assert teacher.toy_pretrain_lr == pytest.approx(3e-3)
    assert (teacher.toy.hidden_size, teacher.toy.num_layers) == (160, 4)

    assert config.environment.name == "calculator"
    assert config.environment.difficulty == "easy"
    assert config.environment.params == {
        "prompt_style": "compact",
        "protocol_version": "v1",
    }
    assert config.environment.train_tasks == raw["environment"]["train_tasks"]
    assert config.environment.eval_tasks == 24
    assert config.environment.test_tasks == 24
    assert config.environment.split_seed == 7

    assert config.rollout.max_turns == 3
    assert config.rollout.max_new_tokens_per_turn == 40
    assert config.rollout.max_total_tokens == 512
    assert config.rollout.temperature == pytest.approx(1.0)
    assert config.rollout.top_p == pytest.approx(1.0)
    assert config.rollout.max_parse_errors == 2
    assert config.rollout.max_repeated_calls == 2

    assert config.selection.selector is SelectorName.ALL_MODEL_TOKENS
    assert config.selection.critical_weight == pytest.approx(1.0)
    assert config.selection.other_weight == pytest.approx(1.0)

    assert config.loss.mode is LossMode.BUCKETED_TOPK_TAIL
    assert config.loss.divergence is Divergence.REVERSE_KL
    assert config.loss.temperature == pytest.approx(1.0)
    assert config.loss.scale_by_temperature_squared is True
    assert config.loss.top_k == 16
    assert config.loss.tail_epsilon == pytest.approx(1e-9)
    assert config.loss.chunk_size == 128
    assert config.loss.sampled_token_nll_weight == pytest.approx(0.0)

    assert config.train.cycles == raw["train"]["cycles"]
    assert config.train.rollouts_per_cycle == 8
    assert config.train.gradient_accumulation_steps == 8
    assert config.train.learning_rate == pytest.approx(3e-3)
    assert config.train.lr_schedule is LRSchedule.COSINE
    assert config.train.optimizer is OptimizerName.ADAMW
    assert config.train.sft_warmup_cycles == raw["train"]["sft_warmup_cycles"]
    assert config.train.save_every_cycles == 0
    assert config.train.eval_every_cycles == 20

    assert config.memory.strategy is MemoryStrategy.RESIDENT
    assert config.memory.oom_retries == 2
    assert config.memory.min_chunk_size == 8

    assert config.cache.dir is None
    assert config.cache.entries_per_shard == 16
    assert config.cache.strict_policy_version is True
    assert config.cache.reuse_across_policy_versions is False
    assert config.cache.dtype == "float32"
    assert config.cache.keep_cycles == 2

    assert config.eval.enabled is True
    assert config.eval.split == "eval"
    assert config.eval.tasks is None
    assert config.eval.temperature == pytest.approx(0.0)
    assert config.eval.seed == 0

    assert config.report.enabled is True
    assert config.report.max_trajectories == 4
    assert config.report.max_tokens_per_trajectory == 240


def test_toy_cpu_recipe_leaves_unspecified_fields_at_their_defaults() -> None:
    """Keys absent from the recipe must come from the model, not from nowhere."""
    config = RunConfig.from_yaml(RECIPES_DIR / "toy_cpu.yaml")
    raw = _read_raw(RECIPES_DIR / "toy_cpu.yaml")

    assert "warmup_steps" not in raw["train"]
    assert config.train.warmup_steps == 0
    assert config.train.log_every_steps == 1
    assert config.train.adam_beta1 == pytest.approx(0.9)
    assert config.train.adam_beta2 == pytest.approx(0.95)

    assert "exact_max_vocab" not in raw["loss"]
    assert config.loss.exact_max_vocab == 8192
    assert config.loss.allow_large_exact is False
    assert config.loss.jsd_beta == pytest.approx(0.5)

    assert "ratio" not in raw["selection"]
    assert config.selection.ratio == pytest.approx(0.35)
    assert config.selection.max_positions_per_trajectory is None

    assert config.models.student.attn_implementation == "sdpa"
    assert config.models.student.trust_remote_code is False
    assert config.memory.empty_cache_between_phases is True
    assert config.memory.reset_peak_stats_each_cycle is True
    assert config.cache.verify_checksums_on_load is True
    assert config.eval.max_turns is None


@pytest.mark.parametrize("path", BENCHMARK_RECIPES, ids=lambda p: p.name)
def test_every_benchmark_arm_deep_merges_into_a_valid_run_config(path: Path) -> None:
    """A benchmark file is only useful if every arm it declares is itself runnable."""
    from miniverl.evaluation.benchmark import deep_merge
    from miniverl.evaluation.schema import BenchmarkConfig

    config = BenchmarkConfig.from_yaml(path)
    assert isinstance(config.base, str), "this test assumes a recipe path, not an inline base"
    base = RunConfig.from_yaml(config.base).model_dump(mode="json")

    modes = set()
    for arm in config.arms:
        merged = RunConfig.model_validate(deep_merge(base, arm.overrides))
        modes.add(merged.run.mode)
    assert TrainingMode.OPD in modes, "a matched-budget benchmark must contain an OPD arm"


# -- from_yaml failure modes ---------------------------------------------------


def test_missing_recipe_raises_config_error_with_an_actionable_hint(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        RunConfig.from_yaml(tmp_path / "does_not_exist.yaml")
    error = excinfo.value
    assert "recipe not found" in error.message
    assert "does_not_exist.yaml" in error.message
    assert error.hint is not None
    assert "recipes/" in error.hint
    assert error.hint in str(error)


def test_directory_instead_of_a_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="recipe not found"):
        RunConfig.from_yaml(tmp_path)


@pytest.mark.parametrize(
    "text",
    ["run: {name: unterminated\n", "a: b: c\n", "run:\n  name: x\n   indent: y\n"],
    ids=["unterminated-flow", "double-colon", "bad-indent"],
)
def test_unparseable_yaml_raises_config_error(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path, "broken.yaml", text)
    with pytest.raises(ConfigError) as excinfo:
        RunConfig.from_yaml(path)
    assert "is not valid YAML" in excinfo.value.message


@pytest.mark.parametrize(
    "text", ["", "\n", "# only a comment\n"], ids=["empty", "blank", "comment"]
)
def test_empty_recipe_raises_config_error(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path, "empty.yaml", text)
    with pytest.raises(ConfigError) as excinfo:
        RunConfig.from_yaml(path)
    assert excinfo.value.message.endswith("is empty")


@pytest.mark.parametrize(
    "text", ["- one\n- two\n", "just a string\n", "42\n"], ids=["list", "scalar", "int"]
)
def test_non_mapping_recipe_raises_config_error(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path, "not_a_mapping.yaml", text)
    with pytest.raises(ConfigError) as excinfo:
        RunConfig.from_yaml(path)
    assert "must contain a YAML mapping at the top level" in excinfo.value.message


def test_a_valid_recipe_on_disk_is_not_confused_for_a_broken_one(tmp_path: Path) -> None:
    """Guards the failure tests above: the same code path accepts a good file."""
    path = _write(tmp_path, "good.yaml", yaml.safe_dump(_payload()))
    assert RunConfig.from_yaml(path).run.mode is TrainingMode.OPD


# -- extra="forbid" ------------------------------------------------------------


@pytest.mark.parametrize(
    ("sections", "location"),
    [
        ({"unexpected_top_level": 1}, ("unexpected_top_level",)),
        ({"run": {"nmae": "typo"}}, ("run", "nmae")),
        ({"loss": {"topk": 16}}, ("loss", "topk")),
        ({"cache": {"reuse": True}}, ("cache", "reuse")),
        ({"environment": {"tasks": 8}}, ("environment", "tasks")),
        ({"models": {"student": {"lora": {"rank": 8}}}}, ("models", "student", "lora", "rank")),
        (
            {"models": {"teacher": {"toy": {"n_heads": 4}}}},
            ("models", "teacher", "toy", "n_heads"),
        ),
    ],
    ids=["top", "run", "loss", "cache", "environment", "lora", "teacher-toy"],
)
def test_unknown_keys_are_rejected_at_every_depth(
    sections: dict[str, Any], location: tuple[str, ...]
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        RunConfig.from_mapping(_payload(**sections))
    errors = excinfo.value.errors()
    assert [e["type"] for e in errors] == ["extra_forbidden"]
    assert errors[0]["loc"] == location


def test_environment_params_is_the_one_free_form_mapping() -> None:
    """``extra=forbid`` must not leak into the per-environment parameter bag."""
    config = RunConfig.from_mapping(
        _payload(environment={"params": {"prompt_style": "compact", "nested": {"depth": 2}}})
    )
    assert config.environment.params == {"prompt_style": "compact", "nested": {"depth": 2}}


# -- cross-field rules ---------------------------------------------------------


def test_schema_version_from_the_future_is_rejected() -> None:
    _rejects(
        _payload(schema_version=CONFIG_SCHEMA_VERSION + 1),
        f"is not supported by this miniVERL build (expected {CONFIG_SCHEMA_VERSION})",
    )


def test_opd_rejects_cache_reuse_across_policy_versions() -> None:
    _rejects(
        _payload(cache={"reuse_across_policy_versions": True}),
        "cache.reuse_across_policy_versions=true contradicts run.mode=opd",
    )


def test_opd_requires_strict_policy_version() -> None:
    _rejects(
        _payload(cache={"strict_policy_version": False}),
        "run.mode=opd requires cache.strict_policy_version=true",
    )


def test_offline_kd_requires_cache_reuse_across_policy_versions() -> None:
    _rejects(
        _payload(run={"mode": "offline_kd"}),
        "run.mode=offline_kd needs cache.reuse_across_policy_versions=true",
    )


def test_offline_kd_with_reuse_enabled_is_accepted() -> None:
    config = RunConfig.from_mapping(
        _payload(run={"mode": "offline_kd"}, cache={"reuse_across_policy_versions": True})
    )
    assert config.run.mode is TrainingMode.OFFLINE_KD
    assert config.is_on_policy is False


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_jsd_rejects_degenerate_beta(beta: float) -> None:
    _rejects(
        _payload(loss={"divergence": "jsd", "jsd_beta": beta}),
        f"loss.jsd_beta must be strictly inside (0, 1) for divergence=jsd, got {beta}",
    )


@pytest.mark.parametrize("beta", [1e-6, 0.5, 0.999])
def test_jsd_accepts_beta_strictly_inside_the_unit_interval(beta: float) -> None:
    config = RunConfig.from_mapping(_payload(loss={"divergence": "jsd", "jsd_beta": beta}))
    assert config.loss.jsd_beta == pytest.approx(beta)


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_degenerate_beta_is_only_rejected_for_jsd(beta: float) -> None:
    """The endpoint check must be scoped to the divergence that actually mixes."""
    config = RunConfig.from_mapping(_payload(loss={"divergence": "reverse_kl", "jsd_beta": beta}))
    assert config.loss.divergence is Divergence.REVERSE_KL
    assert config.loss.jsd_beta == pytest.approx(beta)


def test_exact_full_vocab_rejects_a_top_k() -> None:
    _rejects(
        _payload(loss={"mode": "exact_full_vocab", "top_k": 64}),
        "loss.mode=exact_full_vocab ignores loss.top_k",
    )


def test_exact_full_vocab_accepts_top_k_one() -> None:
    config = RunConfig.from_mapping(_payload(loss={"mode": "exact_full_vocab", "top_k": 1}))
    assert config.loss.mode is LossMode.EXACT_FULL_VOCAB
    assert config.loss.top_k == 1


def test_bucketed_mode_still_accepts_a_real_top_k() -> None:
    config = RunConfig.from_mapping(_payload(loss={"mode": "bucketed_topk_tail", "top_k": 64}))
    assert config.loss.top_k == 64


def test_exact_full_vocab_accepts_a_recipe_that_omits_top_k() -> None:
    """The remedy the rejection message suggests -- removing the key -- has to work.

    ``loss.top_k`` defaults to 64, so the check consults ``model_fields_set`` rather than
    the resolved value, then normalizes the resolved value to 1 so the dumped
    ``config.resolved.yaml`` loads again.
    """
    config = RunConfig.from_mapping(_payload(loss={"mode": "exact_full_vocab"}))
    assert config.loss.mode is LossMode.EXACT_FULL_VOCAB
    assert config.loss.top_k == 1


def test_sft_rejects_a_partial_ce_weight() -> None:
    _rejects(
        _payload(run={"mode": "sft"}, loss={"ce_weight": 0.5}),
        "run.mode=sft trains with oracle cross-entropy only",
    )


@pytest.mark.parametrize("ce_weight", [0.0, 1.0])
def test_sft_accepts_implicit_and_explicit_cross_entropy(ce_weight: float) -> None:
    config = RunConfig.from_mapping(_payload(run={"mode": "sft"}, loss={"ce_weight": ce_weight}))
    assert config.run.mode is TrainingMode.SFT
    assert config.loss.ce_weight == pytest.approx(ce_weight)


def test_a_partial_ce_weight_is_allowed_outside_sft() -> None:
    _rejects(
        _payload(loss={"ce_weight": 0.5}),
        "loss.ce_weight is ambiguous in distillation modes",
    )
    config = RunConfig.from_mapping(_payload(loss={"sampled_token_nll_weight": 0.5}))
    assert config.loss.sampled_token_nll_weight == pytest.approx(0.5)


def test_strict_opd_rejects_two_updates_from_one_rollout_batch() -> None:
    _rejects(
        _payload(
            train={
                "rollouts_per_cycle": 8,
                "gradient_accumulation_steps": 4,
                "opd_freshness": "strict",
            }
        ),
        "opd_freshness=strict requires exactly one optimizer step",
    )


def test_replay_is_explicit_and_never_on_policy() -> None:
    config = RunConfig.from_mapping(
        _payload(
            train={
                "rollouts_per_cycle": 8,
                "gradient_accumulation_steps": 4,
                "opd_freshness": "replay",
            }
        )
    )
    assert config.train.opd_freshness is OPDFreshness.REPLAY
    assert config.is_on_policy is False


def test_sft_rejects_an_sft_warmup() -> None:
    _rejects(
        _payload(run={"mode": "sft"}, train={"sft_warmup_cycles": 1}),
        "train.sft_warmup_cycles applies to offline_kd/opd runs",
    )


def test_sft_warmup_is_accepted_for_opd() -> None:
    config = RunConfig.from_mapping(_payload(train={"sft_warmup_cycles": 250}))
    assert config.train.sft_warmup_cycles == 250


def test_sft_without_a_warmup_is_accepted() -> None:
    config = RunConfig.from_mapping(_payload(run={"mode": "sft"}, train={"sft_warmup_cycles": 0}))
    assert config.train.sft_warmup_cycles == 0


@pytest.mark.parametrize("quantization", ["nf4", "int8"])
def test_toy_backend_rejects_a_quantized_student(quantization: str) -> None:
    _rejects(
        _payload(models={"student": {"quantization": quantization}}),
        "the toy backend does not support quantization "
        "(models.student.quantization must be 'none')",
    )


@pytest.mark.parametrize("quantization", ["nf4", "int8"])
def test_toy_backend_rejects_a_quantized_teacher(quantization: str) -> None:
    _rejects(
        _payload(models={"teacher": {"quantization": quantization}}),
        "the toy backend does not support quantization "
        "(models.teacher.quantization must be 'none')",
    )


def test_hf_backend_accepts_a_quantized_student() -> None:
    """The quantization ban is a property of the toy backend, not of quantization."""
    config = RunConfig.from_mapping(
        _payload(
            models={
                "backend": "hf",
                "student": {"quantization": "nf4", "lora": {"enabled": True}},
            }
        )
    )
    assert config.models.backend is ModelBackend.HF
    assert config.models.student.quantization is Quantization.NF4


@pytest.mark.parametrize(
    ("max_total", "per_turn"), [(64, 64), (32, 64), (16, 8192)], ids=["equal", "less", "far-less"]
)
def test_total_token_budget_must_exceed_the_per_turn_budget(max_total: int, per_turn: int) -> None:
    _rejects(
        _payload(rollout={"max_total_tokens": max_total, "max_new_tokens_per_turn": per_turn}),
        "rollout.max_total_tokens must exceed rollout.max_new_tokens_per_turn",
    )


def test_a_total_token_budget_one_above_the_per_turn_budget_is_accepted() -> None:
    config = RunConfig.from_mapping(
        _payload(rollout={"max_total_tokens": 65, "max_new_tokens_per_turn": 64})
    )
    assert config.rollout.max_total_tokens == 65


def test_eval_max_turns_far_above_the_rollout_bound_is_rejected() -> None:
    _rejects(
        _payload(rollout={"max_turns": 4}, eval={"max_turns": 17}),
        "eval.max_turns is implausibly larger than rollout.max_turns",
    )


def test_eval_max_turns_at_four_times_the_rollout_bound_is_accepted() -> None:
    config = RunConfig.from_mapping(_payload(rollout={"max_turns": 4}, eval={"max_turns": 16}))
    assert config.eval.max_turns == 16


@pytest.mark.parametrize("selector", ["all_model_tokens", "tool_and_final"])
def test_a_stale_ratio_is_accepted_by_the_ratio_free_selectors(selector: str) -> None:
    """Pins today's behaviour: ``selection.ratio`` is ignored, not rejected.

    ``RunConfig._validate_combination`` reaches a branch whose comment promises to reject a
    stale ratio for these two selectors, but the branch body is ``pass``.
    """
    config = RunConfig.from_mapping(_payload(selection={"selector": selector, "ratio": 0.9}))
    assert config.selection.selector is SelectorName(selector)
    assert config.selection.ratio == pytest.approx(0.9)


# -- sub-model rules -----------------------------------------------------------


@pytest.mark.parametrize(("hidden_size", "num_heads"), [(66, 4), (64, 6), (100, 8)])
def test_toy_model_rejects_a_hidden_size_that_is_not_divisible_by_the_head_count(
    hidden_size: int, num_heads: int
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        ToyModelConfig(hidden_size=hidden_size, num_heads=num_heads)
    assert f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}" in str(
        excinfo.value
    )


@pytest.mark.parametrize(("hidden_size", "num_heads"), [(64, 4), (96, 4), (160, 16)])
def test_toy_model_accepts_a_divisible_hidden_size(hidden_size: int, num_heads: int) -> None:
    toy = ToyModelConfig(hidden_size=hidden_size, num_heads=num_heads)
    assert toy.hidden_size % toy.num_heads == 0


def test_the_head_divisibility_rule_also_applies_through_a_run_config() -> None:
    _rejects(
        _payload(models={"student": {"toy": {"hidden_size": 66, "num_heads": 4}}}),
        "hidden_size=66 must be divisible by num_heads=4",
    )


@pytest.mark.parametrize("quantization", ["nf4", "int8"])
def test_a_quantized_student_without_lora_is_rejected(quantization: str) -> None:
    _rejects(
        _payload(
            models={
                "backend": "hf",
                "student": {"quantization": quantization, "lora": {"enabled": False}},
            }
        ),
        "a quantized student must be trained with LoRA adapters",
    )


def test_an_unquantized_student_may_disable_lora() -> None:
    config = RunConfig.from_mapping(
        _payload(models={"student": {"quantization": "none", "lora": {"enabled": False}}})
    )
    assert config.models.student.lora.enabled is False


# -- serialization -------------------------------------------------------------


@pytest.mark.parametrize("path", RUN_RECIPES, ids=lambda p: p.name)
def test_to_yaml_round_trips_through_from_yaml(path: Path, tmp_path: Path) -> None:
    original = RunConfig.from_yaml(path)
    copied = _write(tmp_path, "round_trip.yaml", original.to_yaml())
    assert RunConfig.from_yaml(copied).model_dump() == original.model_dump()


def test_a_resolved_config_is_always_loadable_again(tmp_path: Path) -> None:
    """``config.resolved.yaml`` is re-read by ``miniverl evaluate``, so it must re-validate.

    ``to_yaml`` writes every field including defaults, which is a trap for any rule that treats
    an explicitly written value differently from an omitted one: a recipe that omits ``top_k``
    under ``exact_full_vocab`` would otherwise dump ``top_k: 64`` and then reject itself.
    """
    original = RunConfig.from_mapping(_payload(loss={"mode": "exact_full_vocab"}))
    written = original.write_yaml(tmp_path / "config.resolved.yaml")

    assert yaml.safe_load(written.read_text(encoding="utf-8"))["loss"]["top_k"] == 1
    reloaded = RunConfig.from_yaml(written)
    assert reloaded.model_dump() == original.model_dump()
    assert reloaded.to_yaml() == original.to_yaml()


def test_to_yaml_serializes_enums_as_their_string_values() -> None:
    config = RunConfig.from_mapping(_payload(run={"mode": "sft"}, train={"lr_schedule": "cosine"}))
    raw = yaml.safe_load(config.to_yaml())
    assert raw["run"]["mode"] == "sft"
    assert raw["train"]["lr_schedule"] == "cosine"
    assert raw["loss"]["divergence"] == "reverse_kl"
    assert raw["models"]["backend"] == "toy"


def test_write_yaml_creates_missing_parents_and_can_be_read_back(tmp_path: Path) -> None:
    original = RunConfig.from_yaml(RECIPES_DIR / "toy_cpu.yaml")
    target = tmp_path / "nested" / "deeper" / "config.resolved.yaml"

    written = original.write_yaml(target)

    assert written == target
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == original.to_yaml()
    assert RunConfig.from_yaml(written).model_dump() == original.model_dump()


def test_write_yaml_overwrites_an_existing_file(tmp_path: Path) -> None:
    target = _write(tmp_path, "config.yaml", "stale: true\n")
    original = RunConfig.from_yaml(RECIPES_DIR / "toy_exact_full_vocab.yaml")

    original.write_yaml(target)

    assert "stale" not in target.read_text(encoding="utf-8")
    assert RunConfig.from_yaml(target).run.name == original.run.name


# -- convenience properties ----------------------------------------------------


def test_effective_eval_tasks_falls_back_to_the_environment_split_size() -> None:
    config = RunConfig.from_mapping(_payload(environment={"eval_tasks": 40}, eval={"tasks": None}))
    assert config.eval.tasks is None
    assert config.effective_eval_tasks == 40


def test_effective_eval_tasks_honours_an_explicit_override() -> None:
    config = RunConfig.from_mapping(_payload(environment={"eval_tasks": 40}, eval={"tasks": 5}))
    assert config.effective_eval_tasks == 5


@pytest.mark.parametrize("path", RUN_RECIPES, ids=lambda p: p.name)
def test_effective_eval_tasks_agrees_with_each_shipped_recipe(path: Path) -> None:
    config = RunConfig.from_yaml(path)
    declared = (_read_raw(path).get("eval") or {}).get("tasks")
    if declared is None:
        assert config.eval.tasks is None
        assert config.effective_eval_tasks == config.environment.eval_tasks
    else:
        assert config.effective_eval_tasks == declared


def test_the_shipped_recipes_exercise_both_eval_task_branches() -> None:
    """Keeps the parametrized test above honest: both branches are really reached."""
    declared = [(_read_raw(p).get("eval") or {}).get("tasks") for p in RUN_RECIPES]
    assert any(d is None for d in declared), "no recipe falls back to environment.eval_tasks"
    assert any(d is not None for d in declared), "no recipe overrides eval.tasks"


@pytest.mark.parametrize(
    ("mode", "extra", "expected"),
    [
        ("opd", {}, True),
        ("sft", {}, False),
        ("offline_kd", {"cache": {"reuse_across_policy_versions": True}}, False),
    ],
)
def test_is_on_policy_is_true_only_for_opd(
    mode: str, extra: dict[str, Any], expected: bool
) -> None:
    config = RunConfig.from_mapping(_payload(run={"mode": mode}, **extra))
    assert config.run.mode is TrainingMode(mode)
    assert config.is_on_policy is expected
