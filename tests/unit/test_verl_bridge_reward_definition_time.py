"""Definition-time expressions and signature contracts the v0.6.3 RC accepted.

Every source below was accepted by the pre-fix static policy with
``status: ok``, ``findings: []`` and the strongest default level, even though
importing the module would have run ``exploit()``. The static checker never
executes any of it -- these tests assert that the checker *names* what would
run, and that the marker file is never created while doing so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from miniverl.bridge.reward_static import inspect_reward_scaffold

PRELUDE = """\
from pathlib import Path


def exploit():
    Path("PWNED.txt").write_text("executed", encoding="utf-8")
    return object


"""

VALID = """\
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return 0.0
"""


def _inspect(tmp_path: Path, source: str) -> dict:
    target = tmp_path / "reward_or_verifier_scaffold.py"
    target.write_text(source, encoding="utf-8")
    check = inspect_reward_scaffold(target)
    # No inspection may ever run the file, whatever the verdict is.
    assert not (tmp_path / "PWNED.txt").exists()
    assert check["code_executed"] is False
    assert check["untrusted_code_executed"] is False
    return check


def _categories(check: dict) -> set[str]:
    return {finding["category"] for finding in check["findings"]}


# --------------------------------------------------- definition-time expressions


@pytest.mark.parametrize(
    ("name", "source", "category"),
    [
        (
            "class_base",
            PRELUDE + "class Hidden(exploit()):\n    pass\n\n\n" + VALID,
            "class_base_executes",
        ),
        (
            "class_metaclass",
            PRELUDE + "class Hidden(object, metaclass=exploit()):\n    pass\n\n\n" + VALID,
            "class_keyword_executes",
        ),
        (
            "parameter_annotation",
            PRELUDE + "def compute_score(data_source: exploit(), solution_str, ground_truth,"
            " extra_info=None):\n    return 0.0\n",
            "annotation_executes",
        ),
        (
            "return_annotation",
            PRELUDE + "def compute_score(data_source, solution_str, ground_truth,"
            " extra_info=None) -> exploit():\n    return 0.0\n",
            "annotation_executes",
        ),
        (
            "annotated_assignment",
            PRELUDE + "VALUE: exploit() = 1\n\n\n" + VALID,
            "annotation_executes",
        ),
        (
            "class_body_annotation",
            PRELUDE + "class Hidden:\n    VALUE: exploit() = 1\n\n\n" + VALID,
            "annotation_executes",
        ),
        (
            "nested_call_in_base",
            PRELUDE + "class Hidden(tuple([exploit()])):\n    pass\n\n\n" + VALID,
            "class_base_executes",
        ),
    ],
)
def test_definition_time_expression_is_rejected(
    tmp_path: Path, name: str, source: str, category: str
) -> None:
    check = _inspect(tmp_path, source)

    assert check["status"] == "fail", name
    assert category in _categories(check), name


def test_plain_type_annotations_remain_acceptable(tmp_path: Path) -> None:
    """A normal annotated scaffold must not be collateral damage."""
    check = _inspect(
        tmp_path,
        "from typing import Any\n\n\n"
        "def compute_score(\n"
        "    data_source: str,\n"
        "    solution_str: str,\n"
        "    ground_truth: str,\n"
        "    extra_info: dict[str, Any] | None = None,\n"
        ") -> float:\n"
        "    return 0.0\n",
    )

    assert check["status"] == "ok"
    assert check["findings"] == []


def test_benign_class_base_remains_acceptable(tmp_path: Path) -> None:
    check = _inspect(tmp_path, "class Scorer(object):\n    pass\n\n\n" + VALID)

    assert check["status"] == "ok"


# ------------------------------------------------------------ signature contract


def test_required_keyword_only_extra_info_is_rejected(tmp_path: Path) -> None:
    """verl may call with three positional arguments, so this would TypeError."""
    check = _inspect(
        tmp_path,
        "def compute_score(data_source, solution_str, ground_truth, *, extra_info):\n"
        "    return 0.0\n",
    )

    assert check["status"] == "fail"
    assert "signature_mismatch" in _categories(check)


def test_unknown_keyword_only_parameter_is_rejected(tmp_path: Path) -> None:
    check = _inspect(
        tmp_path,
        "def compute_score(data_source, solution_str, ground_truth, *, other=1):\n    return 0.0\n",
    )

    assert check["status"] == "fail"
    assert "signature_mismatch" in _categories(check)


def test_keyword_only_extra_info_with_a_default_is_accepted(tmp_path: Path) -> None:
    check = _inspect(
        tmp_path,
        "def compute_score(data_source, solution_str, ground_truth, *, extra_info=None):\n"
        "    return 0.0\n",
    )

    assert check["status"] == "ok"


# ------------------------------------------------------------------- terminology


def test_the_strongest_default_level_does_not_claim_import_safety(tmp_path: Path) -> None:
    check = _inspect(tmp_path, VALID)

    assert check["status"] == "ok"
    assert check["verification_level"] == "interface_shape_verified"
    detail = check["detail"].lower()
    assert "safe to import" not in detail
    assert "not executed" in detail or "were not executed" in detail
    assert check["import_runtime_safety"] == "not_verified"


def test_imports_are_recorded_but_never_called_verified(tmp_path: Path) -> None:
    check = _inspect(tmp_path, "import json\nimport os.path\n\n\n" + VALID)

    assert check["status"] == "ok"
    assert check["imports_present"] == ["json", "os.path"]
    assert check["import_runtime_safety"] == "not_verified"


def test_relative_bundle_local_import_is_rejected(tmp_path: Path) -> None:
    check = _inspect(tmp_path, "from . import helper\n\n\n" + VALID)

    assert check["status"] == "fail"
    assert "relative_import" in _categories(check)


# ----------------------------------------------------------------- hostile bounds


def test_oversized_source_is_bounded_not_parsed(tmp_path: Path) -> None:
    check = _inspect(tmp_path, "# " + "A" * (2 * 1024 * 1024) + "\n" + VALID)

    assert check["status"] == "fail"
    assert "source_too_large" in _categories(check)
    assert check["verification_level"] == "not_present"


def test_pathologically_nested_source_is_bounded(tmp_path: Path) -> None:
    depth = 400
    source = "VALUE = " + "[" * depth + "]" * depth + "\n\n\n" + VALID
    check = _inspect(tmp_path, source)

    assert check["status"] == "fail"
    assert {"ast_too_deep", "ast_too_large", "syntax_error"} & _categories(check)


def test_findings_are_bounded(tmp_path: Path) -> None:
    source = "".join(f"call_{index}()\n" for index in range(500)) + "\n\n" + VALID
    check = _inspect(tmp_path, source)

    assert check["status"] == "fail"
    assert len(check["findings"]) <= 50
    assert check["findings_truncated"] is True


def test_undecodable_source_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "reward_or_verifier_scaffold.py"
    target.write_bytes(b"\xff\xfe\x00invalid utf-8 \xc3\x28\n")

    check = inspect_reward_scaffold(target)

    assert check["status"] == "fail"
    assert "encoding_error" in _categories(check)
    assert check["code_executed"] is False


def test_findings_do_not_echo_source_content(tmp_path: Path) -> None:
    secret = "SUPER_SECRET_VALUE_9876543210"
    check = _inspect(tmp_path, PRELUDE + f'TOKEN = exploit_call("{secret}")\n\n\n' + VALID)

    assert check["status"] == "fail"
    assert secret not in repr(check)
