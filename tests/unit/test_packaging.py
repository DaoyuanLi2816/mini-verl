"""The installed package must be complete.

An unanchored ``models/`` rule in ``.gitignore`` once matched
``src/miniverl/models/``, so the entire backend package was excluded from git
and from the built wheel. Everything still worked from an editable install, and
only a clean-environment install revealed it. These tests make the same mistake
impossible to reintroduce without a red test.
"""

from __future__ import annotations

import getpass
import importlib
import json
import pkgutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import miniverl

#: The repository root, for the files that are shipped but not importable.
REPO_ROOT = Path(__file__).resolve().parents[2]

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

    Importing in-process only proves the module *can* be imported here, where an
    earlier test may already have loaded torch. The real property is asserted by
    :func:`test_importing_a_subpackage_does_not_pull_in_torch` below.
    """
    importlib.import_module(name)


def test_importing_a_subpackage_does_not_pull_in_torch() -> None:
    """No ``import miniverl.<subpackage>`` may require torch.

    ``miniverl.teachers`` used to re-export ``LocalTeacherScorer`` eagerly, and
    that module imports torch at module scope, so ``import miniverl.teachers``
    failed outright on a bare install. Nothing here caught it: the in-process
    test above passes whenever torch happens to be installed, so the defect only
    appeared on the CI matrix that omits torch.

    Checking ``sys.modules`` in a fresh interpreter asserts the property itself
    rather than a proxy for it, and it fails in a torch-*ful* environment too --
    which is the only reason it is useful during development.
    """
    program = textwrap.dedent(
        """
        import importlib, json, sys

        offenders = {}
        for name in json.loads(sys.argv[1]):
            module = f"miniverl.{name}"
            for loaded in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
                del sys.modules[loaded]
            importlib.import_module(module)
            if "torch" in sys.modules:
                offenders[module] = True
        print(json.dumps(sorted(offenders)))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, json.dumps(list(REQUIRED_SUBPACKAGES))],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    offenders = json.loads(completed.stdout.strip().splitlines()[-1])
    assert not offenders, (
        f"these subpackages import torch at module scope: {offenders}. "
        "Resolve the torch-dependent names through a module-level __getattr__, "
        "as miniverl.losses and miniverl.teachers do."
    )


def test_the_report_template_ships_with_the_package() -> None:
    template = _package_root() / "reporting" / "templates" / "report.html.j2"
    assert template.is_file(), "the HTML report template is missing from the install"
    assert "miniVERL run" in template.read_text(encoding="utf-8")


def test_version_is_a_release_string() -> None:
    import re

    assert re.fullmatch(r"\d+\.\d+\.\d+([.-]\w+)?", miniverl.__version__), miniverl.__version__


def test_every_published_benchmark_result_validates_against_the_schema():
    """A schema nobody runs is a schema that drifts.

    ``benchmarks/README.md`` asks contributors to submit results that conform to
    ``benchmarks/schema/benchmark-result.schema.json``. That request is only
    honest if the results already in the repository conform too, so this test
    validates each of them and fails if the schema and the files disagree.
    """
    jsonschema = pytest.importorskip("jsonschema")

    root = REPO_ROOT
    schema = json.loads(
        (root / "benchmarks" / "schema" / "benchmark-result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    results = sorted((root / "benchmarks" / "results").glob("*.json"))
    assert results, "benchmarks/results/ has no published result to validate"
    for path in results:
        document = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
        assert not errors, f"{path.name}: " + "; ".join(
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
        )


def test_published_benchmark_results_carry_no_personal_information():
    """A result file is published verbatim, so it must not leak the machine."""
    root = REPO_ROOT
    forbidden = (getpass.getuser().lower(), str(Path.home()).lower().replace("\\", "/"), "onedrive")
    for path in sorted((root / "benchmarks" / "results").glob("*.*")):
        blob = path.read_text(encoding="utf-8").lower().replace("\\\\", "/").replace("\\", "/")
        for needle in forbidden:
            assert needle not in blob, f"{path.name} leaks {needle!r}"


def test_the_committed_json_schema_matches_the_pydantic_model():
    """``schema.py`` claims the two cannot drift; this is what makes that true.

    The file under ``benchmarks/schema/`` is what contributors validate against,
    so if it falls behind the model, a submission can be rejected for the wrong
    reason or accepted with a field the code no longer reads.
    """
    from miniverl.evaluation.schema import json_schema

    committed = json.loads(
        (REPO_ROOT / "benchmarks" / "schema" / "benchmark-result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert committed == json_schema(), (
        "run `miniverl schema > benchmarks/schema/benchmark-result.schema.json`"
    )
