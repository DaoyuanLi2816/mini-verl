"""Regression tests for benchmark-v2 provenance and cumulative accounting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from miniverl.config import TrainingMode
from miniverl.errors import ConfigError, RunDirectoryError
from miniverl.evaluation.benchmark import (
    _training_accounting,
    portable_payload,
    resolve_benchmark_configs,
    run_benchmark,
)
from miniverl.evaluation.schema import BenchmarkConfig, BenchmarkResult
from miniverl.utils.runs import JsonlWriter

REPO_ROOT = Path(__file__).resolve().parents[2]


def _base() -> dict[str, Any]:
    return {
        "models": {
            "student": {"model_id": "toy-student"},
            "teacher": {"model_id": "toy-teacher"},
        },
        "environment": {
            "name": "calculator",
            "difficulty": "medium",
            "test_tasks": 48,
        },
        "train": {
            "rollouts_per_cycle": 8,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1.0e-4,
            "lr_schedule": "cosine",
        },
    }


def _spec(**updates: Any) -> BenchmarkConfig:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "name": "provenance-regression",
        "base": _base(),
        "common_overrides": {
            "environment": {"difficulty": "hard", "test_tasks": 24},
            "train": {"learning_rate": 5.0e-5, "lr_schedule": "constant"},
        },
        "cold_start_overrides": {"environment": {"difficulty": "hard"}},
        "cold_start_cycles": 3,
        "allowed_differences": ["run.mode", "train.cycles"],
        "budget_axis": "optimizer_steps",
        "arms": [
            {
                "name": "cold-start-only",
                "overrides": {"run": {"mode": "sft"}, "train": {"cycles": 0}},
            },
            {
                "name": "opd",
                "overrides": {"run": {"mode": "opd"}, "train": {"cycles": 3}},
            },
        ],
    }
    payload.update(updates)
    return BenchmarkConfig.model_validate(payload)


def test_cold_start_and_arms_resolve_from_explicit_overrides() -> None:
    common, cold, arms = resolve_benchmark_configs(_spec())

    assert common.environment.difficulty == "hard"
    assert common.environment.test_tasks == 24
    assert common.train.learning_rate == pytest.approx(5.0e-5)
    assert common.train.lr_schedule.value == "constant"
    assert cold.environment.difficulty == "hard"
    assert cold.train.cycles == 3
    assert cold.run.mode is TrainingMode.SFT
    assert {arm.name: cfg.environment.difficulty for arm, cfg, _ in arms} == {
        "cold-start-only": "hard",
        "opd": "hard",
    }


def test_the_legacy_medium_to_hard_bug_cannot_hide_in_controlled_metadata() -> None:
    spec = _spec(cold_start_overrides={"environment": {"difficulty": "medium"}})
    common, cold, arms = resolve_benchmark_configs(spec)

    assert common.environment.difficulty == "hard"
    assert cold.environment.difficulty == "medium"
    assert all(cfg.environment.difficulty == "hard" for _, cfg, _ in arms)


def test_undeclared_difference_fails_during_preflight() -> None:
    spec = _spec(
        arms=[
            {
                "name": "bad",
                "overrides": {
                    "run": {"mode": "opd"},
                    "train": {"cycles": 3},
                    "environment": {"difficulty": "easy"},
                },
            }
        ]
    )
    with pytest.raises(ConfigError, match="undeclared arm differences") as excinfo:
        resolve_benchmark_configs(spec)
    assert "environment.difficulty" in str(excinfo.value)
    assert "before any model is loaded" in str(excinfo.value)


def test_local_teacher_adapter_path_is_relative_to_benchmark_file(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "benchmarks" / "configs"
    benchmark_dir.mkdir(parents=True)
    path = benchmark_dir / "adapter.yaml"
    path.write_text(
        """
schema_version: 2
name: adapter-relative
base: {}
allowed_differences: [models.teacher.adapter]
arms:
  - name: protocol-teacher
    overrides:
      models:
        teacher:
          adapter:
            path: ../../artifacts/protocol-teacher
""",
        encoding="utf-8",
    )

    config = BenchmarkConfig.from_yaml(path)
    adapter = config.arms[0].overrides["models"]["teacher"]["adapter"]
    assert Path(adapter["path"]) == (tmp_path / "artifacts" / "protocol-teacher").resolve()


def test_published_provenance_replaces_machine_local_absolute_paths() -> None:
    payload = {
        "run": {"output_dir": r"C:\Users\alice\runs"},
        "models": {
            "student": {"model_id": "Qwen/Qwen3-0.6B"},
            "teacher": {
                "adapter": {
                    "source": "local",
                    "path": r"C:\Users\alice\artifacts\protocol-teacher",
                }
            },
        },
    }
    portable = portable_payload(payload)
    assert portable["run"]["output_dir"] == "<local>/runs"
    assert portable["models"]["student"]["model_id"] == "Qwen/Qwen3-0.6B"
    assert portable["models"]["teacher"]["adapter"]["path"] == "<local>/protocol-teacher"
    assert "alice" not in json.dumps(portable).lower()


@pytest.mark.parametrize(
    "private_path",
    [
        r"C:\Users\Alice\OneDrive\private\adapter",
        "/home/alice/private/adapter",
        "/Users/alice/private/adapter",
    ],
)
def test_portable_payload_redacts_cross_platform_paths_and_secrets(private_path: str) -> None:
    portable = portable_payload(
        {
            "adapter": {"source": "local", "path": private_path},
            "api_token": "secret-value",
            "hostname": "alice-workstation",
        }
    )
    rendered = json.dumps(portable, sort_keys=True)
    assert private_path not in rendered
    assert "alice" not in rendered.lower()
    assert "secret-value" not in rendered
    assert portable["adapter"]["path"].endswith("/adapter")
    assert portable["api_token"] == "<redacted>"
    assert portable["hostname"] == "<redacted>"


@pytest.mark.parametrize(
    "key",
    [
        "github_token",
        "hf_token",
        "wandb_api_key",
        "authorization",
        "cookie",
        "session",
        "session_token",
        "nested-client-secret",
        "databasePassword",
        "private_key",
        "accessKey",
    ],
)
def test_portable_payload_redacts_semantic_secret_keys(key: str) -> None:
    portable = portable_payload({key: "unique-sensitive-value", "tokenizer_id": "public/tokenizer"})

    assert portable[key] == "<redacted>"
    assert portable["tokenizer_id"] == "public/tokenizer"
    assert "unique-sensitive-value" not in json.dumps(portable)


@pytest.mark.parametrize(
    "key",
    [
        "authorization_header",
        "proxy_authorization",
        "set_cookie",
        "session_id",
        "session_key",
        "client_secret_key",
        "auth_token_value",
        "oauth_access_token_value",
        "database_password_file",
        "proxyAuthorization",
        "ClientSecretKey",
    ],
)
def test_portable_payload_redacts_secret_components_anywhere_in_key(key: str) -> None:
    sentinel = "recognizable-private-sentinel"

    portable = portable_payload({key: sentinel})

    assert portable[key] == "<redacted>"
    assert sentinel not in json.dumps(portable)


@pytest.mark.parametrize(
    "key",
    [
        "tokenizer",
        "tokenizer_id",
        "token_count",
        "token_budget",
        "token_type",
        "selected_tokens",
        "session_length",
        "session_count",
        "cookie_count",
    ],
)
def test_portable_payload_preserves_explicit_benign_token_session_and_cookie_keys(key: str) -> None:
    assert portable_payload({key: "public-metadata"})[key] == "public-metadata"


@pytest.mark.parametrize(
    ("raw", "public_fragment"),
    [
        (
            r"failed at C:\Users\Alice Smith\OneDrive\private\adapter",
            "<local>/adapter",
        ),
        (r"failed at \\server\share\private\adapter", "<local>/adapter"),
        (
            "download https://user:password@example.com/path?q=public",
            "https://<redacted>@example.com/path?q=public",
        ),
        (
            r"path=C:&#92;Users&#92;Alice Smith&#92;private&#92;adapter",
            "<local>/adapter",
        ),
        (
            "path=C:/Users/Alice Smith/private/adapter",
            "<local>/adapter",
        ),
    ],
)
def test_portable_payload_redacts_credentials_and_embedded_private_paths(
    raw: str,
    public_fragment: str,
) -> None:
    rendered = json.dumps(portable_payload({"details": raw}), sort_keys=True)

    assert public_fragment in rendered
    for sensitive in ("Alice Smith", "user:password", r"\\server\share"):
        assert sensitive not in rendered


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "ssh://user:password@example.com/repo",
            "ssh://<redacted>@example.com/repo",
        ),
        (
            "git+ssh://user:password@example.com/repo",
            "git+ssh://<redacted>@example.com/repo",
        ),
        (
            "postgresql://user:password@example.com/database",
            "postgresql://<redacted>@example.com/database",
        ),
        (
            "redis://user:password@example.com:6379/cache",
            "redis://<redacted>@example.com:6379/cache",
        ),
    ],
)
def test_portable_text_structurally_redacts_credentialed_urls(raw: str, expected: str) -> None:
    assert portable_payload({"source": raw})["source"] == expected


@given(
    scheme=st.sampled_from(
        ["http", "https", "ssh", "git+ssh", "postgresql", "postgres", "mysql", "mongodb", "redis"]
    ),
    username=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")), min_size=1, max_size=12
    ),
    password=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")), min_size=1, max_size=18
    ),
    port=st.one_of(st.none(), st.integers(min_value=1, max_value=65535)),
)
@settings(max_examples=80)
def test_generated_credentialed_urls_preserve_public_structure_only(
    scheme: str,
    username: str,
    password: str,
    port: int | None,
) -> None:
    authority = f"example.com:{port}" if port is not None else "example.com"
    raw = f"{scheme}://{username}:{password}@{authority}/public/path?q=value#fragment"

    portable = portable_payload({"source": raw})["source"]

    assert portable == f"{scheme}://<redacted>@{authority}/public/path?q=value#fragment"
    assert f"{username}:{password}" not in portable


@pytest.mark.parametrize(
    "private_path",
    [
        "/mnt/data/alice/private/model",
        "/workspace/alice/private/model",
        "/opt/project/private/file",
        "/srv/build user/private/checkpoint",
        "/custom-root/alice/private/model",
    ],
)
def test_portable_text_redacts_embedded_absolute_posix_paths_under_arbitrary_roots(
    private_path: str,
) -> None:
    rendered = portable_payload({"error": f"construction failed at {private_path}; retry"})["error"]

    assert private_path not in rendered
    assert "alice" not in rendered
    assert "<local>/" in rendered


@given(
    root=st.text(alphabet=st.characters(whitelist_categories=("Ll",)), min_size=1, max_size=12),
    owner=st.text(alphabet=st.characters(whitelist_categories=("Ll",)), min_size=1, max_size=12),
    filename=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Nd")), min_size=1, max_size=12
    ),
)
@settings(max_examples=80)
def test_generated_embedded_posix_paths_never_expose_parent_segments(
    root: str,
    owner: str,
    filename: str,
) -> None:
    private_path = f"/{root}/{owner}/private/{filename}.bin"

    portable = portable_payload({"error": f"failed at {private_path}; retry"})["error"]

    assert private_path not in portable
    assert f"/{root}/" not in portable
    assert portable == f"failed at <local>/{filename}.bin; retry"


@pytest.mark.parametrize(
    "public_text",
    [
        "ratio = 1 / 2 and loss/token = 0.5",
        "https://example.com/public/model?q=a/b#docs",
        "Qwen/Qwen3-0.6B",
        "docs/reproducibility.md",
    ],
)
def test_portable_text_does_not_rewrite_relative_or_public_slash_text(public_text: str) -> None:
    assert portable_payload({"note": public_text})["note"] == public_text


@given(
    prefix=st.sampled_from(["auth", "oauth_access", "client", "database", "proxy"]),
    secret_component=st.sampled_from(["token", "secret", "password", "credential"]),
    suffix=st.sampled_from(["value", "file", "header", "key"]),
    separator=st.sampled_from(["_", "-"]),
)
@settings(max_examples=60)
def test_generated_semantic_secret_key_components_are_always_redacted(
    prefix: str,
    secret_component: str,
    suffix: str,
    separator: str,
) -> None:
    key = separator.join((prefix, secret_component, suffix))
    assert portable_payload({key: "generated-private-sentinel"})[key] == "<redacted>"


def test_portable_payload_redacts_nested_secret_values_without_touching_tokenizer() -> None:
    payload = {
        "outer": [
            {"github_token": "github-private-value"},
            {"metadata": {"authorization": "Basic private-auth-value"}},
            {"tokenizer_id": "owner/tokenizer", "revision": "immutable-revision"},
        ]
    }

    rendered = json.dumps(portable_payload(payload), sort_keys=True)

    assert "github-private-value" not in rendered
    assert "private-auth-value" not in rendered
    assert "owner/tokenizer" in rendered
    assert "immutable-revision" in rendered


@given(
    prefix=st.text(alphabet=st.characters(categories=("Ll", "Lu")), min_size=1, max_size=20),
    secret=st.text(min_size=1, max_size=80),
)
@settings(max_examples=75)
def test_semantic_secret_suffixes_are_redacted_in_arbitrary_nested_payloads(
    prefix: str,
    secret: str,
) -> None:
    payload = [{"nested": [{f"{prefix}_token": secret}]}]

    portable = portable_payload(payload)

    assert portable == [{"nested": [{f"{prefix}_token": "<redacted>"}]}]


@pytest.mark.torch
def test_schema_v2_benchmark_runs_end_to_end_and_writes_provenance(tmp_path: Path) -> None:
    base = _base()
    base["environment"].update(
        {"difficulty": "easy", "train_tasks": 1, "eval_tasks": 1, "test_tasks": 1}
    )
    base["eval"] = {"enabled": False, "tasks": 1, "split": "test"}
    spec = BenchmarkConfig.model_validate(
        {
            "schema_version": 2,
            "name": "tiny-v2",
            "base": base,
            "cold_start_cycles": 0,
            "allowed_differences": ["run.mode", "train.cycles"],
            "budget_axis": "optimizer_steps",
            "seeds": [7],
            "arms": [
                {
                    "name": "cold-start-only",
                    "overrides": {"run": {"mode": "sft"}, "train": {"cycles": 0}},
                }
            ],
        }
    )

    result = run_benchmark(
        spec,
        output_dir=tmp_path / "benchmarks",
        invocation=["miniverl", "benchmark", "tiny.yaml"],
    )

    assert result.schema_version == 2
    assert result.invocation == ["miniverl", "benchmark", "tiny.yaml"]
    assert result.common_resolved_config_digest
    assert result.controlled["digest"] == result.common_resolved_config_digest
    assert result.cold_start["checkpoints"][0]["checkpoint_digest"] is None
    assert len(result.arms) == 1
    arm = result.arms[0]
    assert arm.objective == "sft_cross_entropy"
    assert arm.teacher_model_id is None
    assert arm.teacher_queried_positions_total is None
    assert arm.optimizer_steps == 0
    assert arm.wall_seconds >= arm.evaluation_seconds
    assert arm.declared_config_digest == arm.resolved_config_digest
    assert arm.runtime_resolved_config_digest
    assert {row["path"] for row in arm.scientific_config_diff} == {
        "run.mode",
        "train.cycles",
    }
    runtime_paths = {row["path"] for row in arm.runtime_resolution_diff}
    assert {"models.device", "memory.strategy"} <= runtime_paths
    harness_paths = {row["path"] for row in arm.harness_config_diff}
    assert {"run.name", "run.seed", "run.run_id", "report.enabled"} <= harness_paths
    assert not (
        {row["path"] for row in arm.scientific_config_diff}
        & {row["path"] for row in arm.harness_config_diff}
    )
    persisted_text = (tmp_path / "benchmarks" / "tiny-v2.json").read_text(encoding="utf-8")
    assert str(tmp_path).lower() not in persisted_text.lower()
    persisted = BenchmarkResult.model_validate_json(persisted_text)
    assert persisted.common_resolved_config_digest == result.common_resolved_config_digest
    assert persisted.common_declared_config == persisted.common_resolved_config
    assert persisted.common_declared_config_digest == persisted.common_resolved_config_digest

    with pytest.raises(RunDirectoryError, match="already exists"):
        run_benchmark(spec, output_dir=tmp_path / "benchmarks")

    resumed = run_benchmark(spec, output_dir=tmp_path / "benchmarks", resume=True)
    assert len(resumed.arms) == 1
    assert resumed.arms[0].run_id == arm.run_id


def _metrics(tmp_path: Path, rows: list[dict[str, Any]]) -> Any:
    path = tmp_path / "metrics.jsonl"
    writer = JsonlWriter(path)
    for row in rows:
        writer.write(row)
    return SimpleNamespace(paths=SimpleNamespace(metrics=path))


def test_accounting_sums_numerators_and_denominators_across_cycles(tmp_path: Path) -> None:
    trainer = _metrics(
        tmp_path,
        [
            {
                "phase": "opd_cycle",
                "rollouts": {"rollouts": 2, "generated_tokens": 10},
                "selection": {"selected_model_tokens": 2, "total_model_tokens": 4},
            },
            {
                "phase": "opd_cycle",
                "rollouts": {"rollouts": 5, "generated_tokens": 90},
                "selection": {"selected_model_tokens": 9, "total_model_tokens": 10},
            },
        ],
    )
    result = _training_accounting(trainer, TrainingMode.OPD)
    assert result["total_trajectories"] == 7
    assert result["generated_training_tokens_total"] == 100
    assert result["model_generated_training_tokens_total"] == 100
    assert result["selected_training_tokens_total"] == 11
    assert result["selected_position_ratio"] == pytest.approx(11 / 14)
    assert result["teacher_queried_positions_total"] == 11
    assert result["teacher_queried_position_ratio"] == pytest.approx(11 / 14)


def test_sft_and_zero_step_accounting_are_mode_correct(tmp_path: Path) -> None:
    sft = _metrics(
        tmp_path / "sft",
        [
            {
                "phase": "sft_cycle",
                "rollouts": {"rollouts": 3, "generated_tokens": 30},
                "selection": {"selected_model_tokens": 12, "total_model_tokens": 20},
                "cache": {"actual_bytes": 999},
            }
        ],
    )
    result = _training_accounting(sft, TrainingMode.SFT)
    assert result["model_generated_training_tokens_total"] == 0
    assert result["teacher_queried_positions_total"] is None
    assert result["teacher_queried_position_ratio"] is None

    empty = _training_accounting(
        _metrics(tmp_path / "empty", []),
        TrainingMode.OPD,
    )
    assert empty["total_trajectories"] == 0
    assert empty["selected_training_tokens_total"] == 0
    assert empty["selected_position_ratio"] is None


@pytest.mark.parametrize(
    "path",
    [
        "benchmarks/results/rtx4080-calc-hard-matched.json",
        "benchmarks/results/cpu-toy-calc-matched.json",
    ],
)
def test_committed_v1_results_remain_readable(path: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = BenchmarkResult.model_validate(payload)
    assert result.schema_version == 1
    assert result.arms
    assert (
        result.arms[0].selected_training_tokens_total
        == payload["arms"][0]["selected_training_tokens"]
    )


@pytest.mark.parametrize(
    "path",
    [
        "recipes/benchmark_calc.yaml",
        "benchmarks/configs/cpu_toy_calc.yaml",
        "benchmarks/configs/gpu_calc_hard.yaml",
        "benchmarks/configs/gpu_calc_hard_local_adapter.yaml",
    ],
)
def test_every_shipped_v2_benchmark_config_resolves_without_model_loading(path: str) -> None:
    config = BenchmarkConfig.from_yaml(REPO_ROOT / path)
    common, cold, arms = resolve_benchmark_configs(config)

    assert common.models.student.model_id
    assert cold.train.cycles == config.cold_start_cycles
    assert len(arms) == len(config.arms)


def test_legacy_gpu_config_is_an_explicit_immutable_archive() -> None:
    path = REPO_ROOT / "benchmarks/configs/gpu_calc_hard_legacy_v1.yaml"
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)

    assert payload["schema_version"] == 1
    assert "3383f2b9a3c595e0fa143fecdc27522ab368b27f" in text
