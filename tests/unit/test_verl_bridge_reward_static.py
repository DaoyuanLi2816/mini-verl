"""`bridge doctor` must never execute an untrusted bundle's reward code.

Up to v0.6.2 the reward check imported the scaffold with ``exec_module``, so a
bundle could act simply by being diagnosed while the report claimed a
"side-effect-free import". Every test here treats the bundle as hostile input.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from miniverl.bridge.export import _reward_scaffold
from miniverl.bridge.reward_static import inspect_reward_scaffold

MARKER_NAME = "PWNED.txt"


# ------------------------------------------------------------------ fixtures


def _scaffold(tmp_path: Path, source: str) -> Path:
    reward = tmp_path / "reward"
    reward.mkdir(parents=True, exist_ok=True)
    path = reward / "reward_or_verifier_scaffold.py"
    path.write_text(source, encoding="utf-8")
    return path


def _valid_signature(body: str = "    return 0.0\n") -> str:
    return "def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n" + body


def _categories(check: dict[str, Any]) -> set[str]:
    return {finding["category"] for finding in check["findings"]}


# ----------------------------------------------------- the pre-fix exploit


def test_top_level_marker_write_is_never_executed(tmp_path: Path) -> None:
    """The v0.6.2 reproducer: top-level code wrote a file during inspection."""
    marker = tmp_path / MARKER_NAME
    path = _scaffold(
        tmp_path,
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('arbitrary code ran', encoding='utf-8')\n"
        + _valid_signature(),
    )

    check = inspect_reward_scaffold(path)

    assert not marker.exists(), "inspecting a bundle executed its reward code"
    assert check["status"] == "fail"
    assert check["code_executed"] is False
    assert check["untrusted_code_executed"] is False
    assert "top_level_call" in _categories(check)


def test_environment_variable_read_is_rejected(tmp_path: Path) -> None:
    path = _scaffold(
        tmp_path,
        "import os\nLEAKED = os.environ.get('PATH')\n" + _valid_signature(),
    )
    check = inspect_reward_scaffold(path)
    assert check["status"] == "fail"
    assert "non_literal_assignment" in _categories(check)


def test_top_level_subprocess_is_rejected(tmp_path: Path) -> None:
    path = _scaffold(
        tmp_path,
        "import subprocess\nsubprocess.run(['echo', 'hi'], check=False)\n" + _valid_signature(),
    )
    check = inspect_reward_scaffold(path)
    assert check["status"] == "fail"
    findings = [item for item in check["findings"] if item["category"] == "top_level_call"]
    assert findings and "subprocess" in findings[0]["detail"]


def test_default_argument_call_is_rejected(tmp_path: Path) -> None:
    """Defaults are evaluated at definition time, so they run on import too."""
    marker = tmp_path / MARKER_NAME
    path = _scaffold(
        tmp_path,
        "from pathlib import Path\n"
        "def compute_score(data_source, solution_str, ground_truth,\n"
        f"                  extra_info=Path({str(marker)!r}).write_text('x')):\n"
        "    return 0.0\n",
    )
    check = inspect_reward_scaffold(path)
    assert not marker.exists()
    assert check["status"] == "fail"
    assert "default_argument_executes" in _categories(check)


def test_class_body_statement_is_rejected(tmp_path: Path) -> None:
    """Class bodies execute at import just like module level."""
    path = _scaffold(
        tmp_path,
        "import os\nclass Config:\n    root = os.getcwd()\n" + _valid_signature(),
    )
    check = inspect_reward_scaffold(path)
    assert check["status"] == "fail"
    assert "non_literal_assignment" in _categories(check)


# ----------------------------------------------------------- valid scaffold


def test_generated_scaffold_verifies_statically(tmp_path: Path) -> None:
    """The scaffold miniVERL itself exports must pass without being executed."""
    path = _scaffold(tmp_path, _reward_scaffold())

    check = inspect_reward_scaffold(path)

    assert check["status"] == "ok"
    assert check["verification_level"] == "interface_statically_verified"
    assert check["findings"] == []
    assert check["code_executed"] is False
    assert check["signature"] == [
        "data_source",
        "solution_str",
        "ground_truth",
        "extra_info",
    ]
    # The fail-closed placeholder is still in place, detected without running it.
    assert check["implementation_complete"] is False


def test_completed_scaffold_reports_implementation_complete(tmp_path: Path) -> None:
    path = _scaffold(
        tmp_path,
        '"""Real reward."""\n'
        "from __future__ import annotations\n"
        "\n"
        "SCALE = 2.0\n"
        "\n" + _valid_signature("    return float(solution_str == ground_truth) * SCALE\n"),
    )
    check = inspect_reward_scaffold(path)
    assert check["status"] == "ok"
    assert check["implementation_complete"] is True


# --------------------------------------------------------- interface defects


def test_syntax_error_reports_not_interface_verified(tmp_path: Path) -> None:
    path = _scaffold(tmp_path, "def compute_score(:\n")
    check = inspect_reward_scaffold(path)
    assert check["status"] == "fail"
    assert check["verification_level"] == "not_present"
    assert _categories(check) == {"syntax_error"}


def test_missing_file_reports_not_present(tmp_path: Path) -> None:
    check = inspect_reward_scaffold(tmp_path / "reward" / "absent.py")
    assert check["status"] == "fail"
    assert check["verification_level"] == "not_present"


def test_missing_compute_score_is_rejected(tmp_path: Path) -> None:
    path = _scaffold(tmp_path, "def other(a, b):\n    return a\n")
    check = inspect_reward_scaffold(path)
    assert check["status"] == "fail"
    assert check["verification_level"] == "syntax_valid"
    assert "missing_compute_score" in _categories(check)


def test_wrong_signature_is_rejected(tmp_path: Path) -> None:
    path = _scaffold(tmp_path, "def compute_score(prompt, answer):\n    return 0.0\n")
    check = inspect_reward_scaffold(path)
    assert check["status"] == "fail"
    assert "signature_mismatch" in _categories(check)


def test_varargs_signature_is_not_statically_verifiable(tmp_path: Path) -> None:
    path = _scaffold(
        tmp_path,
        "def compute_score(data_source, solution_str, ground_truth, *args, **kwargs):\n"
        "    return 0.0\n",
    )
    check = inspect_reward_scaffold(path)
    assert check["status"] == "fail"
    assert "signature_not_statically_verifiable" in _categories(check)


def test_async_compute_score_is_rejected(tmp_path: Path) -> None:
    path = _scaffold(
        tmp_path,
        "async def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n"
        "    return 0.0\n",
    )
    check = inspect_reward_scaffold(path)
    assert check["status"] == "fail"
    assert "async_compute_score" in _categories(check)


def test_decorated_compute_score_is_rejected(tmp_path: Path) -> None:
    path = _scaffold(
        tmp_path,
        "import functools\n\n@functools.cache\n" + _valid_signature(),
    )
    check = inspect_reward_scaffold(path)
    assert check["status"] == "fail"
    assert "decorator" in _categories(check)


def test_dynamically_assigned_compute_score_is_rejected(tmp_path: Path) -> None:
    path = _scaffold(
        tmp_path,
        "def _impl(data_source, solution_str, ground_truth, extra_info=None):\n"
        "    return 0.0\n"
        "\n"
        "compute_score = _impl\n",
    )
    check = inspect_reward_scaffold(path)
    assert check["status"] == "fail"
    assert "dynamic_compute_score" in _categories(check)


# ------------------------------------------------------- no execution at all


def test_static_verification_never_imports_the_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove zero module execution, not merely an absent side effect."""

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("static verification must not create an import spec")

    monkeypatch.setattr(importlib.util, "spec_from_file_location", _forbidden)
    path = _scaffold(tmp_path, _reward_scaffold())

    check = inspect_reward_scaffold(path)

    assert check["status"] == "ok"
    assert "_miniverl_exported_reward" not in sys.modules


def test_doctor_default_does_not_execute_reward_code(tmp_path: Path) -> None:
    """The exploit must also fail through the public doctor entry point."""
    from miniverl.bridge.doctor import _check_reward

    marker = tmp_path / MARKER_NAME
    _scaffold(
        tmp_path,
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('arbitrary code ran', encoding='utf-8')\n"
        + _valid_signature(),
    )

    check = _check_reward(tmp_path)

    assert not marker.exists()
    assert check["status"] == "fail"
    assert check["code_executed"] is False


# ------------------------------------------------- explicit trusted opt-in


def test_trusted_import_executes_and_says_so(tmp_path: Path) -> None:
    """The escape hatch runs code and must label it, never call it isolated."""
    marker = tmp_path / MARKER_NAME
    path = _scaffold(
        tmp_path,
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('trusted', encoding='utf-8')\n" + _valid_signature(),
    )

    # Top-level code is still a static finding, so the file never reaches the
    # dynamic stage: opting in does not silence the static verdict.
    check = inspect_reward_scaffold(path, trust_and_import=True)
    assert check["status"] == "fail"
    assert not marker.exists()

    # A statically clean scaffold does reach it, and is labelled honestly.
    clean = _scaffold(tmp_path, _reward_scaffold())
    trusted = inspect_reward_scaffold(clean, trust_and_import=True)
    assert trusted["status"] == "ok"
    assert trusted["verification_level"] == "trusted_dynamic_import_verified"
    assert trusted["untrusted_code_executed"] is True
    assert "not a security sandbox" in trusted["isolation"].lower() or (
        "would not be a security sandbox" in trusted["isolation"]
    )
