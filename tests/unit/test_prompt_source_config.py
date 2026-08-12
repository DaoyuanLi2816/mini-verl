from __future__ import annotations

import pytest
from pydantic import ValidationError

from miniverl.config.models import RunConfig, SourceKind


def _models() -> dict[str, object]:
    return {
        "backend": "toy",
        "student": {"model_id": "toy-student"},
        "teacher": {"model_id": "toy-teacher"},
    }


def test_verl_parquet_source_does_not_require_an_environment() -> None:
    config = RunConfig.model_validate(
        {
            "models": _models(),
            "source": {
                "kind": "verl_parquet",
                "train_files": ["train.parquet"],
                "val_files": ["val.parquet"],
                "prompt_key": "prompt",
            },
            "eval": {"tasks": 4},
        }
    )

    assert config.environment is None
    assert config.source.kind is SourceKind.VERL_PARQUET
    assert config.effective_eval_tasks == 4


def test_legacy_environment_recipe_remains_backward_compatible() -> None:
    config = RunConfig.model_validate(
        {
            "models": _models(),
            "environment": {"name": "calculator"},
        }
    )

    assert config.source.kind is SourceKind.ENVIRONMENT
    assert config.environment is not None


def test_environment_and_parquet_source_cannot_be_mixed() -> None:
    with pytest.raises(ValidationError, match="must not define environment"):
        RunConfig.model_validate(
            {
                "models": _models(),
                "environment": {"name": "calculator"},
                "source": {
                    "kind": "verl_parquet",
                    "train_files": ["train.parquet"],
                },
            }
        )


def test_plain_string_prompts_require_an_explicit_opt_in() -> None:
    config = RunConfig.model_validate(
        {
            "models": _models(),
            "source": {
                "kind": "verl_parquet",
                "train_files": ["train.parquet"],
                "allow_plain_string_prompts": True,
                "truncation": "left",
            },
        }
    )

    assert config.source.allow_plain_string_prompts is True
    assert config.source.truncation.value == "left"
