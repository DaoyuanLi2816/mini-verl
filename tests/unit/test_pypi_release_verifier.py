"""Tests for public PyPI release verification without contacting PyPI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_pypi_release.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_pypi_release", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_hashes_requires_one_wheel_and_one_sdist(tmp_path: Path) -> None:
    verifier = _module()
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "miniverl-0.2.0-py3-none-any.whl"
    sdist = dist / "miniverl-0.2.0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(
        "a" * 64 + f"  dist/{wheel.name}\n" + "b" * 64 + f"  dist/{sdist.name}\n",
        encoding="ascii",
    )

    expected = verifier._expected_hashes(sums, tmp_path)

    assert expected == {wheel.name: "a" * 64, sdist.name: "b" * 64}


def test_metadata_verification_checks_exact_names_types_and_hashes(monkeypatch) -> None:
    verifier = _module()
    expected = {
        "miniverl-0.2.0-py3-none-any.whl": "a" * 64,
        "miniverl-0.2.0.tar.gz": "b" * 64,
    }

    def response(url: str, **kwargs):
        assert url == "https://pypi.org/pypi/miniverl/0.2.0/json"
        return {
            "info": {"name": "miniverl", "version": "0.2.0"},
            "urls": [
                {
                    "filename": filename,
                    "packagetype": "bdist_wheel" if filename.endswith(".whl") else "sdist",
                    "digests": {"sha256": digest},
                    "url": f"https://files.pythonhosted.org/packages/{filename}",
                }
                for filename, digest in expected.items()
            ],
        }

    monkeypatch.setattr(verifier, "_request_json", response)
    urls = verifier._verify_metadata(
        project="miniverl",
        version="0.2.0",
        expected=expected,
    )

    assert len(urls) == 2


def test_integrity_verification_requires_attestations_for_every_file(monkeypatch) -> None:
    verifier = _module()
    seen: list[str] = []

    def response(url: str, **kwargs):
        seen.append(url)
        return {"attestation_bundles": [{"attestations": [{"envelope": {}}]}]}

    monkeypatch.setattr(verifier, "_request_json", response)
    verifier._verify_integrity_metadata(
        project="miniverl",
        version="0.2.0",
        filenames=["miniverl-0.2.0-py3-none-any.whl", "miniverl-0.2.0.tar.gz"],
    )

    assert len(seen) == 2
    assert all(url.endswith("/provenance") for url in seen)


def test_hash_mismatch_fails_before_attestation_verification(monkeypatch) -> None:
    verifier = _module()

    def response(url: str, **kwargs):
        return {
            "info": {"name": "miniverl", "version": "0.2.0"},
            "urls": [
                {
                    "filename": "miniverl-0.2.0-py3-none-any.whl",
                    "packagetype": "bdist_wheel",
                    "digests": {"sha256": "b" * 64},
                    "url": (
                        "https://files.pythonhosted.org/packages/miniverl-0.2.0-py3-none-any.whl"
                    ),
                },
                {
                    "filename": "miniverl-0.2.0.tar.gz",
                    "packagetype": "sdist",
                    "digests": {"sha256": "wrong"},
                    "url": "https://files.pythonhosted.org/packages/miniverl-0.2.0.tar.gz",
                },
            ],
        }

    monkeypatch.setattr(verifier, "_request_json", response)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verifier._verify_metadata(
            project="miniverl",
            version="0.2.0",
            expected={
                "miniverl-0.2.0-py3-none-any.whl": "b" * 64,
                "miniverl-0.2.0.tar.gz": "a" * 64,
            },
        )


def test_long_description_requires_tag_pinned_links_on_the_rendered_page(monkeypatch) -> None:
    verifier = _module()
    repository = "https://github.com/DaoyuanLi2816/mini-verl"
    tag = "v0.2.4"
    paths = (
        "docs/single-gpu-guide.md",
        "recipes/qwen_consumer_gpu_calc.yaml",
        "benchmarks/results/gpu-calc-hard-equal-update-v2.json",
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE",
        "CONTRIBUTING.md",
        "SECURITY.md",
    )
    links = [f"{repository}/blob/{tag}/{path}" for path in paths]
    banner = f"https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/{tag}/docs/banner.svg"
    links.append(banner)
    description = "\n".join([*links[:-1], f"![release banner]({banner})"])

    monkeypatch.setattr(
        verifier,
        "_request_json",
        lambda *_args, **_kwargs: {"info": {"description": description}},
    )
    requested: list[str] = []

    def request_text(url: str) -> str:
        requested.append(url)
        if url == "https://pypi.org/project/miniverl/0.2.4/":
            return "\n".join(
                [*links[:-1], '<img src="https://pypi-camo.example/banner" alt="release banner">']
            )
        return "ok"

    monkeypatch.setattr(verifier, "_request_text", request_text)
    verifier._verify_long_description_links(
        project="miniverl",
        version="0.2.4",
        repository=repository,
    )
    assert set(links) <= set(requested)


def test_long_description_retries_a_pypi_client_challenge(monkeypatch) -> None:
    verifier = _module()
    repository = "https://github.com/DaoyuanLi2816/mini-verl"
    tag = "v0.2.4"
    paths = (
        "docs/single-gpu-guide.md",
        "recipes/qwen_consumer_gpu_calc.yaml",
        "benchmarks/results/gpu-calc-hard-equal-update-v2.json",
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE",
        "CONTRIBUTING.md",
        "SECURITY.md",
    )
    banner = f"https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/{tag}/docs/banner.svg"
    description = "\n".join(
        [
            *(f"{repository}/blob/{tag}/{path}" for path in paths),
            f"![release banner]({banner})",
        ]
    )
    monkeypatch.setattr(
        verifier,
        "_request_json",
        lambda *_args, **_kwargs: {"info": {"description": description}},
    )
    monkeypatch.setattr(
        verifier,
        "_request_text",
        lambda *_args, **_kwargs: "<title>Client Challenge</title>",
    )

    with pytest.raises(RuntimeError, match="client challenge"):
        verifier._verify_long_description_links(
            project="miniverl",
            version="0.2.4",
            repository=repository,
        )


def test_long_description_can_defer_a_client_challenge_to_browser_inspection(
    monkeypatch,
    capsys,
) -> None:
    verifier = _module()
    repository = "https://github.com/DaoyuanLi2816/mini-verl"
    tag = "v0.2.4"
    paths = (
        "docs/single-gpu-guide.md",
        "recipes/qwen_consumer_gpu_calc.yaml",
        "benchmarks/results/gpu-calc-hard-equal-update-v2.json",
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE",
        "CONTRIBUTING.md",
        "SECURITY.md",
    )
    banner = f"https://raw.githubusercontent.com/DaoyuanLi2816/mini-verl/{tag}/docs/banner.svg"
    links = [f"{repository}/blob/{tag}/{path}" for path in paths]
    description = "\n".join([*links, f'<img src="{banner}" alt="release banner" width="880">'])
    monkeypatch.setattr(
        verifier,
        "_request_json",
        lambda *_args, **_kwargs: {"info": {"description": description}},
    )
    requested: list[str] = []

    def request_text(url: str) -> str:
        requested.append(url)
        if url == "https://pypi.org/project/miniverl/0.2.4/":
            return "<title>Client Challenge</title>"
        return "ok"

    monkeypatch.setattr(verifier, "_request_text", request_text)
    verifier._verify_long_description_links(
        project="miniverl",
        version="0.2.4",
        repository=repository,
        allow_rendered_page_challenge=True,
    )

    assert {*links, banner} <= set(requested)
    assert "requires a browser" in capsys.readouterr().out


def test_long_description_rejects_main_drift_for_a_stable_release(monkeypatch) -> None:
    verifier = _module()
    monkeypatch.setattr(
        verifier,
        "_request_json",
        lambda *_args, **_kwargs: {
            "info": {
                "description": ("https://github.com/DaoyuanLi2816/mini-verl/blob/main/CHANGELOG.md")
            }
        },
    )
    with pytest.raises(RuntimeError, match="missing release-pinned"):
        verifier._verify_long_description_links(
            project="miniverl",
            version="0.2.4",
            repository="https://github.com/DaoyuanLi2816/mini-verl",
        )
