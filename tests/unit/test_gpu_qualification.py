from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest


def _payload(tmp_path: Path) -> tuple[dict[str, object], Path]:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir(parents=True)
    wheel = artifact_root / "miniverl-0.10.1.dev0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    run_manifest = artifact_root / "run-manifest.json"
    run_manifest.write_text('{"status":"completed"}\n', encoding="utf-8")

    from miniverl.qualification import sha256_file

    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "miniverl_gpu_qualification",
        "level": "release_smoke",
        "status": "passed",
        "measured_at": "2026-08-14T12:00:00Z",
        "source_commit": "a" * 40,
        "miniverl_version": "0.10.1.dev0",
        "wheel": {
            "filename": wheel.name,
            "sha256": sha256_file(wheel),
        },
        "candidate": {
            "manifest_sha256": "9" * 64,
            "artifact_name": "candidate-distributions",
            "workflow_repository": None,
            "workflow_path": None,
            "workflow_run_id": None,
            "workflow_run_attempt": None,
            "installed_from_candidate": True,
            "import_origin_verified": True,
            "cli_origin_verified": True,
            "import_origin": "qualification_venv_site_packages",
        },
        "profile": {
            "name": "verl-opd-v0.8-single-gpu-v1",
            "identity_digest": "b" * 64,
            "upstream_tag": "v0.8.0",
            "upstream_commit": "c" * 40,
        },
        "environment": {
            "known_good_manifest_sha256": "d" * 64,
            "gpu_name": "NVIDIA GeForce RTX 4080",
            "gpu_count": 1,
            "vram_gib": 15.99,
            "driver": "596.49",
            "cuda_runtime": "13.0",
            "python": "3.10.11",
            "packages": {
                "torch": "2.13.0+cu130",
                "transformers": "5.14.1",
                "peft": "0.18.0",
                "accelerate": "1.14.0",
                "bitsandbytes": "0.50.0",
                "numpy": "2.2.6",
                "pyarrow": "25.0.0",
                "safetensors": "0.8.0",
            },
        },
        "models": [
            {"role": "actor", "model_id": "org/actor", "revision": "e" * 40},
            {"role": "teacher", "model_id": "org/teacher", "revision": "f" * 40},
        ],
        "execution": {
            "rollout_completed": True,
            "teacher_scoring_completed": True,
            "optimizer_updates": 1,
            "peft_adapter_exported": True,
            "peft_adapter_reload_verified": True,
            "cuda_allocated_before_bytes": 0,
            "cuda_allocated_after_teardown_bytes": 0,
            "cuda_teardown_tolerance_bytes": 2097152,
        },
        "inputs": {
            "config_sha256": "1" * 64,
            "plan_sha256": "2" * 64,
            "parquet_sha256": "3" * 64,
        },
        "artifacts": [
            {
                "name": "run_manifest",
                "path": run_manifest.name,
                "sha256": sha256_file(run_manifest),
                "bytes": run_manifest.stat().st_size,
            }
        ],
        "checks": {
            "executed": [
                "wheel_install",
                "doctor",
                "plan",
                "run_dry_run",
                "real_actor_teacher_update",
                "peft_reload",
                "cuda_teardown",
            ],
            "skipped": [],
            "not_applicable": ["distributed_verl_execution"],
        },
        "scientific_scope": {
            "runtime_correctness_only": True,
            "task_quality_evaluated": False,
            "alignment_quality_evaluated": False,
            "distributed_execution_tested": False,
            "other_hardware_measured": False,
        },
    }
    return payload, artifact_root


def test_strict_qualification_round_trip_and_artifact_hashes(tmp_path: Path) -> None:
    from miniverl.qualification import validate_qualification_payload

    payload, root = _payload(tmp_path / "private")
    assert validate_qualification_payload(payload, artifact_root=root) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda p: p.update(extra="invented"), "schema"),
        (lambda p: p["execution"].update(optimizer_updates=0), "schema"),  # type: ignore[union-attr]
        (lambda p: p["environment"].update(vram_gib=float("nan")), "schema"),  # type: ignore[union-attr]
        (lambda p: p.update(status="passed", source_commit="short"), "schema"),
    ],
)
def test_qualification_rejects_invalid_or_invented_fields(
    tmp_path: Path, mutation, expected: str
) -> None:
    from miniverl.qualification import validate_qualification_payload

    payload, root = _payload(tmp_path / "checksum")
    mutation(payload)
    assert validate_qualification_payload(payload, artifact_root=root)[0].startswith(expected)


def test_qualification_rejects_private_paths_and_hash_mismatch(tmp_path: Path) -> None:
    from miniverl.qualification import validate_qualification_payload

    payload, root = _payload(tmp_path / "private")
    payload["artifacts"][0]["path"] = "C:\\Users\\someone\\secret.json"  # type: ignore[index]
    assert validate_qualification_payload(payload, artifact_root=root)[0].startswith("privacy")

    payload, root = _payload(tmp_path / "checksum")
    payload["artifacts"][0]["sha256"] = "0" * 64  # type: ignore[index]
    assert any(
        "checksum" in problem
        for problem in validate_qualification_payload(payload, artifact_root=root)
    )

    payload, root = _payload(tmp_path / "extra")
    (root / "unreferenced.bin").write_bytes(b"not declared")
    assert any(
        "unreferenced files" in problem
        for problem in validate_qualification_payload(payload, artifact_root=root)
    )


@pytest.mark.parametrize("replacement", [b"run", b'{"status":"completed"}\nextra'])
def test_qualification_rejects_artifact_size_drift_even_with_updated_hash(
    tmp_path: Path, replacement: bytes
) -> None:
    from miniverl.qualification import sha256_file, validate_qualification_payload

    payload, root = _payload(tmp_path)
    artifact = root / "run-manifest.json"
    declared_bytes = payload["artifacts"][0]["bytes"]  # type: ignore[index]
    artifact.write_bytes(replacement)
    assert artifact.stat().st_size != declared_bytes
    payload["artifacts"][0]["sha256"] = sha256_file(artifact)  # type: ignore[index]
    assert any(
        "size mismatch" in problem
        for problem in validate_qualification_payload(payload, artifact_root=root)
    )


def _full_result(name: str) -> dict[str, object]:
    kinds = {
        "direct": ("single_gpu_opd_developer_workload", "measured"),
        "pg_k1": ("single_gpu_verl_pg_k1_systems_workload", "measured"),
        "smollm2": (
            "single_gpu_smollm2_direct_gkd_developer_workload",
            "maintainer_measured",
        ),
    }
    kind, status = kinds[name]
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "status": status,
        "source_commit": "a" * 40,
        "miniverl_version": "0.10.1.dev0",
        "hardware": {"gpu": "NVIDIA GeForce RTX 4080", "gpu_count": 1},
        "resume": {
            "status": "exact_match",
            "adapter_and_optimizer_byte_identical": True,
            "training_state_fields_identical": True,
            "trajectories_byte_identical": True,
        },
        "resource_contract": {"peak_reserved_within_limit": True},
        "verl": {"distributed_execution_tested": False},
        "artifacts": {"standard_peft_load_verified": name != "pg_k1"},
    }
    if name == "smollm2":
        payload["scaleout"] = {
            "artifact_bundle_complete": True,
            "upstream_config_parse_passed": True,
            "model_data_load_smoke_passed": True,
            "launchable": True,
            "distributed_execution_tested": False,
        }
    return payload


def test_full_qualification_promotion_binds_all_canonical_results(tmp_path: Path) -> None:
    from miniverl.qualification import sha256_file, validate_qualification_file
    from scripts.promote_full_gpu_qualification import promote

    payload, root = _payload(tmp_path)
    qualification = root / "qualification.json"
    qualification.write_text(json.dumps(payload), encoding="utf-8")
    results: dict[str, Path] = {}
    for name in ("direct", "pg_k1", "smollm2"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_full_result(name)), encoding="utf-8")
        results[name] = path

    promoted = promote(
        qualification,
        direct=results["direct"],
        pg_k1=results["pg_k1"],
        smollm2=results["smollm2"],
    )

    assert promoted.level == "full_qualification"
    assert validate_qualification_file(qualification) == []
    assert {item.name for item in promoted.artifacts} >= {
        "release_smoke_record",
        "full_direct_result",
        "full_pg_k1_result",
        "full_smollm2_result",
    }

    incomplete = promoted.model_dump(mode="json")
    incomplete["artifacts"] = [
        item for item in incomplete["artifacts"] if item["name"] != "full_smollm2_result"
    ]
    from miniverl.qualification import validate_qualification_payload

    assert any(
        "full qualification evidence is missing" in problem
        for problem in validate_qualification_payload(incomplete, artifact_root=root)
    )

    complete = promoted.model_dump(mode="json")
    full_result = root / "full/direct.json"
    evidence = next(item for item in complete["artifacts"] if item["path"] == "full/direct.json")
    full_result.write_bytes(full_result.read_bytes() + b"\n")
    evidence["sha256"] = sha256_file(full_result)
    assert any(
        "size mismatch" in problem
        for problem in validate_qualification_payload(complete, artifact_root=root)
    )


def test_full_qualification_rejects_failed_resume_equivalence(tmp_path: Path) -> None:
    from scripts.promote_full_gpu_qualification import promote

    payload, root = _payload(tmp_path)
    qualification = root / "qualification.json"
    qualification.write_text(json.dumps(payload), encoding="utf-8")
    results: dict[str, Path] = {}
    for name in ("direct", "pg_k1", "smollm2"):
        result = _full_result(name)
        if name == "pg_k1":
            result["resume"]["trajectories_byte_identical"] = False  # type: ignore[index]
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        results[name] = path
    with pytest.raises(ValueError, match="equivalence"):
        promote(
            qualification,
            direct=results["direct"],
            pg_k1=results["pg_k1"],
            smollm2=results["smollm2"],
        )


def test_process_teardown_clears_bnb_and_cublas_caches() -> None:
    from scripts.run_gpu_qualification import _clear_process_global_cuda_caches

    calls: list[str] = []
    torch_module = SimpleNamespace(
        _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: calls.append("cublas"))
    )
    bnb_functional = SimpleNamespace(name2qmap={"dynamic": object()})
    _clear_process_global_cuda_caches(torch_module, bnb_functional)
    assert bnb_functional.name2qmap == {}
    assert calls == ["cublas"]


def test_qualification_rejects_candidate_manifest_mismatch(tmp_path: Path) -> None:
    from miniverl.qualification import validate_qualification_payload

    payload, root = _payload(tmp_path)
    problems = validate_qualification_payload(
        payload,
        artifact_root=root,
        expected_candidate_manifest_sha256="8" * 64,
    )
    assert any("candidate manifest" in problem for problem in problems)


def test_install_origin_must_be_inside_qualification_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.run_gpu_qualification import _verify_install_origin

    venv = tmp_path / "venv"
    site = venv / "Lib/site-packages"
    scripts = venv / "Scripts"
    site.mkdir(parents=True)
    scripts.mkdir()
    imported = site / "miniverl/__init__.py"
    imported.parent.mkdir()
    imported.write_text("", encoding="utf-8")
    cli = scripts / "miniverl.exe"
    cli.write_bytes(b"exe")
    monkeypatch.setattr("sys.prefix", str(venv))
    monkeypatch.setattr("sys.base_prefix", str(tmp_path / "base"))
    monkeypatch.setattr("sysconfig.get_paths", lambda: {"purelib": str(site)})
    _verify_install_origin(str(imported), str(cli))
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not originate"):
        _verify_install_origin(str(outside), str(cli))


def test_committed_schema_is_generated_from_the_strict_model() -> None:
    from miniverl.qualification import qualification_json_schema

    committed = json.loads(
        Path("docs/generated/gpu-qualification-v1.schema.json").read_text(encoding="utf-8")
    )
    assert committed == qualification_json_schema()


def test_release_artifact_extraction_rejects_traversal_and_symlinks(tmp_path: Path) -> None:
    from scripts.verify_release_qualification import _safe_extract

    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escape.json", "{}")
    with pytest.raises(ValueError, match="unsafe"):
        _safe_extract(traversal, tmp_path / "out-traversal")

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("qualification.json")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(ValueError, match="symlink"):
        _safe_extract(symlink, tmp_path / "out-symlink")

    duplicate = tmp_path / "duplicate.zip"
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(duplicate, "w") as archive,
    ):
        archive.writestr("qualification.json", "{}")
        archive.writestr("qualification.json", "{}")
    with pytest.raises(ValueError, match="duplicate"):
        _safe_extract(duplicate, tmp_path / "out-duplicate")


def _zip_tree(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


def _remote_pair(
    tmp_path: Path,
    *,
    candidate_run_id: int = 42,
    candidate_run_attempt: int = 1,
    qualification_run_attempt: int = 1,
) -> tuple[Path, Path, Path]:
    from miniverl.qualification import sha256_file
    from miniverl.release_candidate import PINNED_BUILD_TOOLS

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    wheel = candidate / "miniverl-0.10.1.dev0-py3-none-any.whl"
    sdist = candidate / "miniverl-0.10.1.dev0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    manifest = {
        "schema_version": 1,
        "kind": "miniverl_release_candidate",
        "source_commit": "a" * 40,
        "miniverl_version": "0.10.1.dev0",
        "created_at": "2026-08-15T12:00:00Z",
        "artifact_name": "candidate-distributions",
        "workflow": {
            "kind": "github_actions",
            "repository": "DaoyuanLi2816/mini-verl",
            "workflow_path": ".github/workflows/gpu.yml",
            "run_id": candidate_run_id,
            "run_attempt": candidate_run_attempt,
        },
        "build": {"os": "Linux", "python": "3.12.0", "tools": PINNED_BUILD_TOOLS},
        "wheel": {
            "filename": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": sha256_file(wheel),
        },
        "sdist": {
            "filename": sdist.name,
            "bytes": sdist.stat().st_size,
            "sha256": sha256_file(sdist),
        },
    }
    manifest_path = candidate / "candidate-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (candidate / "SHA256SUMS").write_text(
        f"{sha256_file(wheel)}  {wheel.name}\n{sha256_file(sdist)}  {sdist.name}\n",
        encoding="utf-8",
    )
    payload, qualification = _payload(tmp_path / "qualification-source")
    payload["candidate"] = {
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_name": "candidate-distributions",
        "workflow_repository": "DaoyuanLi2816/mini-verl",
        "workflow_path": ".github/workflows/gpu.yml",
        "workflow_run_id": 42,
        "workflow_run_attempt": qualification_run_attempt,
        "installed_from_candidate": True,
        "import_origin_verified": True,
        "cli_origin_verified": True,
        "import_origin": "qualification_venv_site_packages",
    }
    payload["environment"]["known_good_manifest_sha256"] = sha256_file(  # type: ignore[index]
        tmp_path / "known-good.json"
    )
    (qualification / "qualification.json").write_text(json.dumps(payload), encoding="utf-8")
    candidate_zip = tmp_path / "candidate.zip"
    qualification_zip = tmp_path / "qualification.zip"
    _zip_tree(candidate, candidate_zip)
    _zip_tree(qualification, qualification_zip)
    return candidate_zip, qualification_zip, tmp_path / "known-good.json"


def test_release_verifier_accepts_only_same_run_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known_good = tmp_path / "known-good.json"
    known_good.write_text("{}", encoding="utf-8")
    candidate_zip, qualification_zip, _ = _remote_pair(tmp_path)
    from scripts import verify_release_qualification as verifier

    run = {
        "id": 42,
        "head_sha": "a" * 40,
        "conclusion": "success",
        "event": "workflow_dispatch",
        "workflow_id": 7,
        "path": ".github/workflows/gpu.yml",
        "repository": {"full_name": "DaoyuanLi2816/mini-verl"},
        "head_repository": {"full_name": "DaoyuanLi2816/mini-verl"},
        "run_attempt": 1,
        "created_at": "2026-08-15T12:00:00Z",
        "html_url": "https://example.invalid/run/42",
    }

    def request(url: str, token: str):
        del token
        if url.endswith("/actions/workflows/gpu.yml"):
            return {"id": 7, "path": ".github/workflows/gpu.yml"}
        if "/runs?" in url:
            return {"workflow_runs": [run]}
        return {
            "artifacts": [
                {
                    "id": 1,
                    "name": "candidate-distributions",
                    "expired": False,
                    "archive_download_url": "candidate",
                },
                {
                    "id": 2,
                    "name": "gpu-release-smoke",
                    "expired": False,
                    "archive_download_url": "qualification",
                },
            ]
        }

    def download(url: str, token: str, destination: Path) -> None:
        del token
        shutil.copy2(candidate_zip if url == "candidate" else qualification_zip, destination)

    monkeypatch.setattr(verifier, "_request_json", request)
    monkeypatch.setattr(verifier, "_download", download)
    result = verifier.fetch_and_validate(
        repository="DaoyuanLi2816/mini-verl",
        workflow="gpu.yml",
        commit="a" * 40,
        qualification_artifact_name="gpu-release-smoke",
        candidate_artifact_name="candidate-distributions",
        token="token",
        required_gpu_name="NVIDIA GeForce RTX 4080",
        known_good=known_good,
        output=tmp_path / "accepted",
    )
    assert result["workflow_run_id"] == 42


def test_release_verifier_rejects_cross_run_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known_good = tmp_path / "known-good.json"
    known_good.write_text("{}", encoding="utf-8")
    candidate_zip, qualification_zip, _ = _remote_pair(tmp_path, candidate_run_id=41)
    from scripts import verify_release_qualification as verifier

    run = {
        "id": 42,
        "head_sha": "a" * 40,
        "conclusion": "success",
        "event": "workflow_dispatch",
        "workflow_id": 7,
        "path": ".github/workflows/gpu.yml",
        "repository": {"full_name": "DaoyuanLi2816/mini-verl"},
        "head_repository": {"full_name": "DaoyuanLi2816/mini-verl"},
        "run_attempt": 1,
        "created_at": "2026-08-15T12:00:00Z",
        "html_url": "https://example.invalid/run/42",
    }
    monkeypatch.setattr(
        verifier,
        "_request_json",
        lambda url, token: (
            {"id": 7, "path": ".github/workflows/gpu.yml"}
            if url.endswith("/actions/workflows/gpu.yml")
            else {"workflow_runs": [run]}
            if "/runs?" in url
            else {
                "artifacts": [
                    {
                        "id": 1,
                        "name": "candidate-distributions",
                        "expired": False,
                        "archive_download_url": "candidate",
                    },
                    {
                        "id": 2,
                        "name": "gpu-release-smoke",
                        "expired": False,
                        "archive_download_url": "qualification",
                    },
                ]
            }
        ),
    )
    monkeypatch.setattr(
        verifier,
        "_download",
        lambda url, token, destination: shutil.copy2(
            candidate_zip if url == "candidate" else qualification_zip, destination
        ),
    )
    with pytest.raises(RuntimeError, match="same-run"):
        verifier.fetch_and_validate(
            repository="DaoyuanLi2816/mini-verl",
            workflow="gpu.yml",
            commit="a" * 40,
            qualification_artifact_name="gpu-release-smoke",
            candidate_artifact_name="candidate-distributions",
            token="token",
            required_gpu_name="NVIDIA GeForce RTX 4080",
            known_good=known_good,
            output=tmp_path / "rejected",
        )


def test_release_verifier_rejects_cross_attempt_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known_good = tmp_path / "known-good.json"
    known_good.write_text("{}", encoding="utf-8")
    candidate_zip, qualification_zip, _ = _remote_pair(
        tmp_path,
        candidate_run_attempt=1,
        qualification_run_attempt=2,
    )
    from scripts import verify_release_qualification as verifier

    run = {
        "id": 42,
        "head_sha": "a" * 40,
        "conclusion": "success",
        "event": "workflow_dispatch",
        "workflow_id": 7,
        "path": ".github/workflows/gpu.yml",
        "repository": {"full_name": "DaoyuanLi2816/mini-verl"},
        "head_repository": {"full_name": "DaoyuanLi2816/mini-verl"},
        "run_attempt": 2,
        "created_at": "2026-08-15T12:00:00Z",
        "html_url": "https://example.invalid/run/42",
    }
    monkeypatch.setattr(
        verifier,
        "_request_json",
        lambda url, token: (
            {"id": 7, "path": ".github/workflows/gpu.yml"}
            if url.endswith("/actions/workflows/gpu.yml")
            else {"workflow_runs": [run]}
            if "/runs?" in url
            else {
                "artifacts": [
                    {
                        "id": 1,
                        "name": "candidate-distributions",
                        "expired": False,
                        "archive_download_url": "candidate",
                    },
                    {
                        "id": 2,
                        "name": "gpu-full-qualification",
                        "expired": False,
                        "archive_download_url": "qualification",
                    },
                ]
            }
        ),
    )
    monkeypatch.setattr(
        verifier,
        "_download",
        lambda url, token, destination: shutil.copy2(
            candidate_zip if url == "candidate" else qualification_zip, destination
        ),
    )
    with pytest.raises(RuntimeError, match="same-run"):
        verifier.fetch_and_validate(
            repository="DaoyuanLi2816/mini-verl",
            workflow="gpu.yml",
            commit="a" * 40,
            qualification_artifact_name="gpu-full-qualification",
            candidate_artifact_name="candidate-distributions",
            token="token",
            required_gpu_name="NVIDIA GeForce RTX 4080",
            known_good=known_good,
            output=tmp_path / "rejected-attempt",
        )


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_cross_origin_redirect_strips_sensitive_headers(status: int) -> None:
    from scripts.verify_release_qualification import _SafeRedirectHandler

    request = urllib.request.Request(
        "https://api.github.com/repos/org/repo/actions/artifacts/1/zip",
        headers={
            "Authorization": "Bearer top-secret",
            "Proxy-Authorization": "Basic also-secret",
            "Cookie": "session=secret",
            "Accept": "application/octet-stream",
            "User-Agent": "miniVERL-test",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    redirected = _SafeRedirectHandler().redirect_request(
        request,
        None,
        status,
        "redirect",
        {},
        "https://objects.githubusercontent.com/artifact.zip",
    )
    assert redirected is not None
    lowered = {name.lower(): value for name, value in redirected.header_items()}
    assert "authorization" not in lowered
    assert "proxy-authorization" not in lowered
    assert "cookie" not in lowered
    assert lowered["accept"] == "application/octet-stream"
    assert lowered["user-agent"] == "miniVERL-test"
    assert lowered["x-github-api-version"] == "2022-11-28"


def test_same_origin_redirect_preserves_authorization() -> None:
    from scripts.verify_release_qualification import _SafeRedirectHandler

    request = urllib.request.Request(
        "https://api.github.com/old",
        headers={"Authorization": "Bearer top-secret"},
    )
    redirected = _SafeRedirectHandler().redirect_request(
        request, None, 302, "redirect", {}, "https://api.github.com/new"
    )
    assert redirected is not None
    assert redirected.get_header("Authorization") == "Bearer top-secret"


def test_download_redirect_does_not_send_token_to_another_origin() -> None:
    from scripts.verify_release_qualification import _request_json

    class TargetHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            payload = json.dumps(dict(self.headers.items())).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args) -> None:
            del format, args

    target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/artifact")
            self.end_headers()

        def log_message(self, format, *args) -> None:
            del format, args

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    threads = [Thread(target=server.serve_forever, daemon=True) for server in (target, redirect)]
    for thread in threads:
        thread.start()
    try:
        received = _request_json(f"http://127.0.0.1:{redirect.server_port}/start", "top-secret")
    finally:
        redirect.shutdown()
        target.shutdown()
        redirect.server_close()
        target.server_close()
    lowered = {name.lower(): value for name, value in received.items()}
    assert "authorization" not in lowered
    assert lowered["accept"] == "application/vnd.github+json"
    assert lowered["user-agent"] == "miniVERL-release-qualification"
    assert lowered["x-github-api-version"] == "2022-11-28"


def test_network_errors_never_expose_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import verify_release_qualification as verifier

    class BrokenOpener:
        def open(self, request, timeout):
            del request, timeout
            raise urllib.error.URLError("Bearer top-secret")

    monkeypatch.setattr(verifier, "_URL_OPENER", BrokenOpener())
    with pytest.raises(RuntimeError) as error:
        verifier._request_json("https://api.github.com/example", "top-secret")
    assert "top-secret" not in str(error.value)


def test_release_chain_binds_exact_candidate_bytes(tmp_path: Path) -> None:
    from miniverl.qualification import sha256_file
    from miniverl.release_candidate import load_candidate_manifest
    from miniverl.release_chain import validate_release_chain

    known_good = tmp_path / "known-good.json"
    known_good.write_text("{}", encoding="utf-8")
    candidate_zip, qualification_zip, _ = _remote_pair(tmp_path)
    candidate = tmp_path / "candidate-unpacked"
    qualification = tmp_path / "qualification-unpacked"
    with zipfile.ZipFile(candidate_zip) as archive:
        archive.extractall(candidate)
    with zipfile.ZipFile(qualification_zip) as archive:
        archive.extractall(qualification)
    manifest_path = candidate / "candidate-manifest.json"
    qualification_path = qualification / "qualification.json"
    expected = {
        "expected_commit": "a" * 40,
        "expected_known_good_sha256": sha256_file(known_good),
        "required_gpu_name": "NVIDIA GeForce RTX 4080",
    }
    assert validate_release_chain(candidate, manifest_path, qualification_path, **expected) == []

    record = load_candidate_manifest(manifest_path)
    wheel = candidate / record.wheel.filename
    wheel.write_bytes(b"different wheel from the same source commit")
    payload = record.model_dump(mode="json")
    payload["wheel"]["bytes"] = wheel.stat().st_size
    payload["wheel"]["sha256"] = sha256_file(wheel)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    (candidate / "SHA256SUMS").write_text(
        f"{payload['wheel']['sha256']}  {record.wheel.filename}\n"
        f"{record.sdist.sha256}  {record.sdist.filename}\n",
        encoding="utf-8",
    )
    problems = validate_release_chain(candidate, manifest_path, qualification_path, **expected)
    assert any("wheel checksum" in problem for problem in problems)
    assert any("candidate manifest checksum" in problem for problem in problems)


def test_release_verifier_rejects_expired_or_wrong_workflow_artifacts() -> None:
    from scripts.verify_release_qualification import _artifact, _run_is_exact

    with pytest.raises(ValueError, match="expired"):
        _artifact(
            [{"name": "candidate-distributions", "expired": True}],
            "candidate-distributions",
        )
    base = {
        "head_sha": "a" * 40,
        "conclusion": "success",
        "event": "workflow_dispatch",
        "workflow_id": 7,
        "path": ".github/workflows/other.yml",
        "repository": {"full_name": "DaoyuanLi2816/mini-verl"},
        "head_repository": {"full_name": "DaoyuanLi2816/mini-verl"},
    }
    assert not _run_is_exact(
        base,
        repository="DaoyuanLi2816/mini-verl",
        workflow_id=7,
        workflow_path=".github/workflows/gpu.yml",
        commit="a" * 40,
    )
    base["path"] = ".github/workflows/gpu.yml"
    base["head_repository"] = {"full_name": "someone/fork"}
    assert not _run_is_exact(
        base,
        repository="DaoyuanLi2816/mini-verl",
        workflow_id=7,
        workflow_path=".github/workflows/gpu.yml",
        commit="a" * 40,
    )
