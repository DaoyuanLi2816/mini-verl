"""The installed package must be complete.

An unanchored ``models/`` rule in ``.gitignore`` once matched
``src/miniverl/models/``, so the entire backend package was excluded from git
and from the built wheel. Everything still worked from an editable install, and
only a clean-environment install revealed it. These tests make the same mistake
impossible to reintroduce without a red test.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import miniverl

#: Every subpackage that must exist in an installed distribution.
REQUIRED_SUBPACKAGES = (
    "agent",
    "cache",
    "config",
    "environments",
    "evaluation",
    "losses",
    "models",
    "reporting",
    "schemas",
    "selection",
    "teachers",
    "training",
    "trajectory",
    "utils",
)

#: Modules importable without torch. Importing any of these must not pull torch in.
TORCH_FREE_MODULES = (
    "miniverl.cli",
    "miniverl.config",
    "miniverl.config.models",
    "miniverl.doctor",
    "miniverl.errors",
    "miniverl.inspection",
    "miniverl.schemas",
    "miniverl.schemas.alignment",
    "miniverl.schemas.cache",
    "miniverl.schemas.trajectory",
    "miniverl.trajectory",
    "miniverl.trajectory.alignment",
    "miniverl.trajectory.io",
    "miniverl.trajectory.masks",
    "miniverl.agent.protocol",
    "miniverl.agent.transcript",
    "miniverl.environments",
    "miniverl.environments.calculator",
    "miniverl.environments.jsonnav",
    "miniverl.environments.sqlite_env",
    "miniverl.environments.registry",
    "miniverl.cache",
    "miniverl.cache.store",
    "miniverl.cache.stats",
    "miniverl.selection",
    "miniverl.reporting",
    "miniverl.reporting.charts",
    "miniverl.reporting.data",
    "miniverl.reporting.html",
    "miniverl.reporting.markdown",
    "miniverl.utils",
    "miniverl.utils.env",
    "miniverl.utils.gpu",
    "miniverl.utils.lazy",
    "miniverl.utils.logging",
    "miniverl.utils.runs",
    "miniverl.utils.seeding",
    "miniverl.evaluation.schema",
    "miniverl.models.tokenizers",
)


def _package_root() -> Path:
    assert miniverl.__file__ is not None
    return Path(miniverl.__file__).parent


@pytest.mark.parametrize("name", REQUIRED_SUBPACKAGES)
def test_every_subpackage_is_present_on_disk(name: str) -> None:
    directory = _package_root() / name
    assert directory.is_dir(), f"{directory} is missing from the installed package"
    assert (directory / "__init__.py").is_file(), f"{name} has no __init__.py"


@pytest.mark.parametrize("name", REQUIRED_SUBPACKAGES)
def test_every_subpackage_is_importable(name: str) -> None:
    importlib.import_module(f"miniverl.{name}")


def test_no_subpackage_was_left_out_of_the_required_list() -> None:
    """If a new subpackage is added, this list has to grow with it."""
    found = {module.name for module in pkgutil.iter_modules([str(_package_root())]) if module.ispkg}
    assert found == set(REQUIRED_SUBPACKAGES), (
        f"packages on disk {sorted(found)} do not match REQUIRED_SUBPACKAGES"
    )


@pytest.mark.parametrize("name", TORCH_FREE_MODULES)
def test_torch_free_modules_import_without_torch(name: str) -> None:
    """These must work from a bare ``pip install miniverl``.

    The import itself is the assertion: if any of these grew a module-scope
    ``import torch``, the CI core job (which never installs torch) would fail.
    """
    importlib.import_module(name)


def test_the_report_template_ships_with_the_package() -> None:
    template = _package_root() / "reporting" / "templates" / "report.html.j2"
    assert template.is_file(), "the HTML report template is missing from the install"
    assert "miniVERL run" in template.read_text(encoding="utf-8")


def test_version_is_a_release_string() -> None:
    import re

    assert re.fullmatch(r"\d+\.\d+\.\d+([.-]\w+)?", miniverl.__version__), miniverl.__version__
