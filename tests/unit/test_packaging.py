"""The installed package must be complete.

An unanchored ``models/`` rule in ``.gitignore`` once matched
``src/miniverl/models/``, so the entire backend package was excluded from git
and from the built wheel. Everything still worked from an editable install, and
only a clean-environment install revealed it. These tests make the same mistake
impossible to reintroduce without a red test.
"""

from __future__ import annotations

import getpass
import hashlib
import importlib
import json
import pkgutil
import re
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

import miniverl

#: The repository root, for the files that are shipped but not importable.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every subpackage that must exist in an installed distribution.
REQUIRED_SUBPACKAGES = (
    "agent",
    "alignment",
    "alignment_external",
    "bridge",
    "cache",
    "community",
    "config",
    "environments",
    "evaluation",
    "losses",
    "models",
    "reporting",
    "runtime",
    "schemas",
    "selection",
    "teachers",
    "training",
    "trajectory",
    "utils",
)

#: Modules importable without torch. Importing any of these must not pull torch in.
TORCH_FREE_MODULES = (
    "miniverl.bridge",
    "miniverl.bridge.community",
    "miniverl.bridge.config",
    "miniverl.bridge.contract",
    "miniverl.bridge.dataset",
    "miniverl.bridge.doctor",
    "miniverl.bridge.export",
    "miniverl.cli",
    "miniverl.config",
    "miniverl.config.models",
    "miniverl.community",
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
    "miniverl.runtime",
    "miniverl.runtime.roles",
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


def test_the_endpoint_registry_resolves_wherever_miniverl_is_installed() -> None:
    """The registry has to be findable from an installed package, not just a checkout.

    It first resolved by walking up from ``__file__`` to a repository root,
    which works in a source tree and lands on ``lib/python3.12`` from
    site-packages. Only the extracted-sdist job caught that, so this asserts
    the property directly: whatever the layout, the default path exists and
    loads.
    """
    from miniverl.alignment_external.registry import default_registry_path, load_registry

    path = default_registry_path()
    assert path.is_file(), f"the endpoint registry is not reachable at {path}"
    assert load_registry()["endpoints"]


def test_the_registry_travels_inside_the_wheel() -> None:
    """A built wheel must carry the registry beside the module that loads it."""
    wheels = sorted((REPO_ROOT / "dist").glob("*.whl"))
    if not wheels:
        pytest.skip("no built wheel to inspect; run python -m build first")

    with zipfile.ZipFile(wheels[-1]) as archive:
        names = set(archive.namelist())

    assert "miniverl/alignment_external/registry.yaml" in names
    assert "miniverl/alignment_external/profile-v1.yaml" in names


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


def test_github_actions_use_full_commit_shas() -> None:
    """A 39-character near-SHA looks pinned but GitHub refuses the workflow."""
    import re

    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = re.search(r"\buses:\s+[^@\s]+@([^\s#]+)", line)
            if match:
                assert re.fullmatch(r"[0-9a-f]{40}", match.group(1)), (
                    f"{path.name}:{number} action ref is not a full commit SHA: {match.group(1)}"
                )


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

    # RecoveryBench's paired analysis, consumer-runtime, Alignment Lab and
    # external-alignment early-stop artifacts have dedicated exact/schema
    # contracts; none is a BenchmarkResult document.
    results = sorted(
        path
        for path in (root / "benchmarks" / "results").glob("*.json")
        if path.name
        not in {
            "recoverybench-v1-analysis.json",
            "consumer-runtime-v1.json",
            "consumer-runtime-v1-profiler.json",
            "alignment-lab-v1.json",
            "alignment-lab-v1-state-supervision.json",
            "alignment-external-v1.json",
        }
    )
    assert results, "benchmarks/results/ has no published result to validate"
    for path in results:
        document = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
        assert not errors, f"{path.name}: " + "; ".join(
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
        )


def test_headline_schema_v2_result_is_preserved_byte_for_byte() -> None:
    path = REPO_ROOT / "benchmarks" / "results" / "gpu-calc-hard-equal-update-v2.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"
    )


def test_published_benchmark_results_carry_no_personal_information():
    """A result file is published verbatim, so it must not leak the machine."""
    root = REPO_ROOT
    forbidden = (getpass.getuser().lower(), str(Path.home()).lower().replace("\\", "/"), "onedrive")
    for path in sorted((root / "benchmarks" / "results").glob("*.*")):
        blob = path.read_text(encoding="utf-8").lower().replace("\\\\", "/").replace("\\", "/")
        for needle in forbidden:
            assert needle not in blob, f"{path.name} leaks {needle!r}"


def test_published_gpu_visualization_matches_its_source_result():
    """The checked-in SVG must remain parseable and tied to the JSON it summarizes."""
    json_path = REPO_ROOT / "benchmarks" / "results" / "gpu-calc-hard-equal-update-v2.json"
    svg_path = REPO_ROOT / "docs" / "gpu-calc-hard-equal-update-v2.svg"
    document = json.loads(json_path.read_text(encoding="utf-8"))
    root = ET.parse(svg_path).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    groups = {
        group.attrib["data-arm"]: group
        for group in root.findall("svg:g", namespace)
        if "data-arm" in group.attrib
    }

    expected_names = {arm["name"] for arm in document["arms"]}
    assert set(groups) == expected_names
    for name in expected_names:
        arms = [arm for arm in document["arms"] if arm["name"] == name]
        success_mean = sum(arm["strict_task_success_rate"] for arm in arms) / len(arms)
        train_mean = sum(arm["train_seconds"] for arm in arms) / len(arms)
        assert float(groups[name].attrib["data-success-mean"]) == pytest.approx(
            success_mean, abs=1e-6
        )
        assert float(groups[name].attrib["data-train-seconds-mean"]) == pytest.approx(
            train_mean, abs=1e-6
        )

    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    text_nodes = [text.strip() for text in root.itertext() if text.strip()]
    visible_text = " ".join(text_nodes)
    assert f"Source JSON SHA-256 {digest[:16]}" in visible_text
    assert "Saturated calculator task · same success, 6.1× OPD continuation time" in visible_text
    assert "DIAGNOSTIC NEGATIVE CONTROLS · TEACHERS NOT PROTOCOL-QUALIFIED" in visible_text
    assert "intentionally incompatible" not in visible_text.lower()
    assert "Strict success on v0.2 test set" in visible_text
    assert "Continuation train time" in visible_text
    assert "Teacher preparation is not included in continuation bars." in visible_text
    assert "~555" not in visible_text
    assert "optimizer_steps" not in visible_text
    cold_text = [text.strip() for text in groups["cold-start-only"].itertext() if text.strip()]
    assert "0 CONTINUATION UPDATES" in cold_text
    assert "0s" not in cold_text
    for name in ("opd-raw-teacher", "opd-privileged-context"):
        arm_text = [text.strip() for text in groups[name].itertext() if text.strip()]
        assert "NEGATIVE CONTROL · UNGATED" in arm_text
        assert "0%" in arm_text
        assert "PROTOCOL MISMATCH" not in arm_text
    assert "COLLAPSED" not in visible_text
    assert text_nodes.index("OPD · protocol-aligned teacher") < text_nodes.index(
        "DIAGNOSTIC NEGATIVE CONTROLS · TEACHERS NOT PROTOCOL-QUALIFIED"
    )
    assert "\ufffd" not in visible_text
    grid_lines = [
        element
        for element in root.findall("svg:line", namespace)
        if element.attrib.get("class") == "grid"
    ]
    axis_labels = [
        element
        for element in root.findall("svg:text", namespace)
        if element.attrib.get("class") == "axis"
    ]
    assert grid_lines and axis_labels
    assert min(float(line.attrib["y1"]) for line in grid_lines) > max(
        float(label.attrib["y"]) for label in axis_labels
    )
    assert 'fill="url(#background)"' in svg_path.read_text(encoding="utf-8")

    from miniverl.evaluation.schema import BenchmarkResult
    from scripts.publish_benchmark_artifacts import render_svg

    result = BenchmarkResult.model_validate(document)
    assert svg_path.read_text(encoding="utf-8") == render_svg(result, digest)


def test_single_gpu_visual_identity_and_pypi_link_are_prominent() -> None:
    banner = (REPO_ROOT / "docs" / "banner.svg").read_text(encoding="utf-8")
    assert "SINGLE-GPU LLM POST-TRAINING" in banner
    assert "1× CUDA GPU" in banner
    assert "BF16 / FP16 auto" in banner
    assert "typed provenance" in banner
    assert "Alignment · utility · queries · GPU time" in banner
    assert 'pip install "miniverl[train,cuda]"' not in banner
    assert "the GPU you have" not in banner
    assert "16 GB first" not in banner
    ET.fromstring(banner)

    pypi_url = "https://pypi.org/project/miniverl/"
    for readme in ("README.md", "README.zh-CN.md"):
        text = (REPO_ROOT / readme).read_text(encoding="utf-8")
        assert text.count(pypi_url) >= 2
        assert "docs/single-gpu-guide.md" in text


def test_verl_bridge_visual_and_launch_assets_are_accessible_and_honest() -> None:
    architecture = (REPO_ROOT / "docs" / "verl-bridge-architecture.svg").read_text(encoding="utf-8")
    social = (REPO_ROOT / "docs" / "social-preview-v0.6.svg").read_text(encoding="utf-8")
    for asset in (architecture, social):
        root = ET.fromstring(asset)
        assert root.attrib.get("role") == "img"
        assert root.attrib.get("aria-labelledby")
        assert "distributed execution" in " ".join(root.itertext()).lower()

    launch = (REPO_ROOT / "docs" / "verl-bridge-launch.md").read_text(encoding="utf-8")
    demo = (REPO_ROOT / "docs" / "verl-bridge-demo.md").read_text(encoding="utf-8")
    assert "Flagship article" in launch
    assert "Hugging Face card addenda" in launch
    assert "Distributed execution" in launch
    assert "Verified-bridge demo recording script" in demo
    assert "does not prove a distributed verl job ran" in demo


def test_generated_pypi_readme_is_byte_bound_and_has_only_navigable_project_links() -> None:
    from scripts.build_pypi_readme import (
        build_pypi_readme,
        relative_project_targets,
        release_ref,
    )

    generated = build_pypi_readme(REPO_ROOT)
    committed = (REPO_ROOT / "PYPI.md").read_text(encoding="utf-8")
    ref = release_ref(miniverl.__version__)
    assert committed == generated
    assert "C:\\Users\\" not in committed
    assert "/home/" not in committed
    assert not re.search(r"!?\[[^]]*]\((?!https?://|#|mailto:)[^)]+\)", committed)
    assert not re.search(r'(?:src|href)="(?!https?://|#|mailto:)[^"]+"', committed)
    assert relative_project_targets(committed) == []
    assert f"https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/{ref}/docs/banner.svg" in (
        committed
    )
    for path in (
        "docs/single-gpu-guide.md",
        "recipes/qwen_consumer_gpu_calc.yaml",
        "benchmarks/results/gpu-calc-hard-equal-update-v2.json",
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE",
        "CONTRIBUTING.md",
        "SECURITY.md",
    ):
        assert f"https://github.com/DaoyuanLi2816/mini-verl/blob/{ref}/{path}" in committed


@pytest.mark.parametrize(
    ("version", "ref"),
    [("0.6.0", "v0.6.0"), ("0.6.1.dev0", "main")],
)
def test_pypi_generator_rewrites_nested_images_and_every_project_target(
    tmp_path: Path,
    version: str,
    ref: str,
) -> None:
    from scripts.build_pypi_readme import build_pypi_readme, relative_project_targets

    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "[![License](docs/license.svg)](LICENSE)",
                "![Hero](docs/hero.svg)",
                "[Guide](docs/guide.md)",
                '<img src="docs/html.svg"><a href="docs/html.md">HTML</a>',
                "[Anchor](#local-section) [Mail](mailto:user@example.com)",
            ]
        ),
        encoding="utf-8",
    )

    rendered = build_pypi_readme(tmp_path, version=version)

    assert relative_project_targets(rendered) == []
    assert (
        f"[![License](https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/{ref}/"
        "docs/license.svg)]"
        f"(https://github.com/DaoyuanLi2816/mini-verl/blob/{ref}/LICENSE)"
    ) in rendered
    assert (
        f'<img src="https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/{ref}/docs/html.svg">'
    ) in rendered
    assert (
        f'<a href="https://github.com/DaoyuanLi2816/mini-verl/blob/{ref}/docs/html.md">'
    ) in rendered
    assert "[Anchor](#local-section)" in rendered
    assert "[Mail](mailto:user@example.com)" in rendered


def test_project_metadata_and_sdist_policy_use_generated_long_description() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'readme = "PYPI.md"' in pyproject
    for entry in (
        '"/scripts"',
        '"/.github"',
        '"/PYPI.md"',
        '"/CONTRIBUTING.md"',
        '"/CODE_OF_CONDUCT.md"',
        '"/SECURITY.md"',
        '"/PROJECT_STATE.md"',
    ):
        assert entry in pyproject


def test_release_quality_has_one_version_bound_machine_readable_record() -> None:
    record = json.loads(
        (REPO_ROOT / "docs" / "generated" / "quality.json").read_text(encoding="utf-8")
    )
    assert record["schema_version"] == 2
    # The floor must name the release this record measures. Hard-coding it here
    # is what let quality_floor keep saying v0.6.1 inside the v0.6.2 record.
    assert record["quality_floor"] == (
        f"2,000+ tests and 85%+ branch coverage at v{record['release']}"
    )

    local = record["local_validation"]
    release = record["release_validation"]
    # Two different things measured on two different commits. Merging them is
    # what made a Windows RTX 4080 count read as though it had been taken on
    # the squash commit that was actually published.
    assert re.fullmatch(r"[0-9a-f]{40}", local["commit"])
    if release["commit"] == "pending":
        assert record["status"] == "candidate"
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", release["commit"])
        assert local["commit"] != release["commit"]
    assert "commit_relationship" in local
    # No GPU runner exists, so the release commit cannot carry GPU evidence.
    assert "none" in release["gpu_coverage"]

    if ".dev" in miniverl.__version__:
        assert record["status"] in {"candidate", "released"}
        if record["status"] == "candidate":
            assert record["release"] == miniverl.__version__.split(".dev", 1)[0]
    else:
        assert record["release"] == miniverl.__version__
        assert record["status"] in {"candidate", "validated"}
        for section in ("cpu_non_gpu_non_network", "gpu", "network"):
            assert isinstance(local[section]["passed"], int)


def test_repository_markdown_links_and_anchors_are_navigable() -> None:
    from scripts.check_markdown_links import check_markdown_links

    assert check_markdown_links(REPO_ROOT) == []


def test_built_wheel_metadata_uses_generated_readme_and_valid_project_links(
    tmp_path: Path,
) -> None:
    pytest.importorskip("build")
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    [wheel] = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        [metadata_name] = [name for name in archive.namelist() if name.endswith("/METADATA")]
        metadata = archive.read(metadata_name).decode("utf-8")
    _headers, separator, description = metadata.partition("\n\n")
    assert separator
    assert description == (REPO_ROOT / "PYPI.md").read_text(encoding="utf-8")
    from scripts.build_pypi_readme import relative_project_targets

    assert relative_project_targets(description) == []

    project_file_urls = re.findall(
        r"https://(?:github\.com/DaoyuanLi2816/mini-verl/blob|"
        r"raw\.githubusercontent\.com/DaoyuanLi2816/mini-verl)/[^)\s\"<>]+",
        description,
    )
    assert project_file_urls
    for url in project_file_urls:
        path = unquote(urlsplit(url).path)
        if url.startswith("https://github.com/"):
            relative = path.split("/blob/", 1)[1].split("/", 1)[1]
        else:
            relative = path.removeprefix("/DaoyuanLi2816/mini-verl/").split("/", 1)[1]
        relative = relative.split("#", 1)[0]
        assert (REPO_ROOT / relative).is_file(), f"wheel METADATA has a broken link: {url}"


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
