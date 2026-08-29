"""Bind the three canonical full workloads into a strict qualification artifact."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from miniverl.qualification import GPUQualification, sha256_file, validate_qualification_file
from miniverl.utils.runs import write_json_atomic

_RESULTS = {
    "direct": ("single_gpu_opd_developer_workload", "measured"),
    "pg_k1": ("single_gpu_verl_pg_k1_systems_workload", "measured"),
    "smollm2": ("single_gpu_smollm2_direct_gkd_developer_workload", "maintainer_measured"),
}
_EQUIVALENCE = (
    "adapter_and_optimizer_byte_identical",
    "training_state_fields_identical",
    "trajectories_byte_identical",
)
_V011_CHECKS = (
    "v011_hf_cached_direct_n1",
    "v011_hf_cached_pg_n4",
    "v011_rewarded_pg_n4",
    "v011_exact_wheel_runtime_gate",
    "v011_vllm_direct_gkd_n4_r256",
    "v011_policy_refresh_cache_invalidation",
    "v011_external_engine_teardown",
)
_MAX_GPU_MEMORY_MIB = 14.5 * 1024
_PREREGISTRATION_SHA256 = "8cc3ba738c69b59ed19c22c1de874fd00249404198a3e05983477dc8899bb7e5"
_FROZEN_CALCULATOR_SHA256 = "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _validate_result(name: str, payload: dict[str, Any], qualification: GPUQualification) -> None:
    expected_kind, expected_status = _RESULTS[name]
    if payload.get("schema_version") != 1:
        raise ValueError(f"{name}: unsupported schema version")
    if payload.get("kind") != expected_kind or payload.get("status") != expected_status:
        raise ValueError(f"{name}: unexpected kind or status")
    if payload.get("source_commit") != qualification.source_commit:
        raise ValueError(f"{name}: source commit does not match release smoke")
    if payload.get("miniverl_version") != qualification.miniverl_version:
        raise ValueError(f"{name}: miniVERL version does not match release smoke")
    hardware = payload.get("hardware") or {}
    if hardware.get("gpu") != qualification.environment.gpu_name or hardware.get("gpu_count") != 1:
        raise ValueError(f"{name}: hardware does not match release smoke")
    resume = payload.get("resume") or {}
    if resume.get("status") != "exact_match" or any(
        resume.get(field) is not True for field in _EQUIVALENCE
    ):
        raise ValueError(f"{name}: uninterrupted/resume equivalence did not pass")
    if (payload.get("resource_contract") or {}).get("peak_reserved_within_limit") is not True:
        raise ValueError(f"{name}: VRAM resource contract did not pass")
    if (payload.get("verl") or {}).get("distributed_execution_tested") is not False:
        raise ValueError(f"{name}: distributed execution must remain not tested")
    if (
        name in {"direct", "smollm2"}
        and (payload.get("artifacts") or {}).get("standard_peft_load_verified") is not True
    ):
        raise ValueError(f"{name}: standard PEFT reload was not verified")
    if name == "smollm2":
        scaleout = payload.get("scaleout") or {}
        required = (
            "artifact_bundle_complete",
            "upstream_config_parse_passed",
            "model_data_load_smoke_passed",
            "launchable",
        )
        if any(scaleout.get(field) is not True for field in required):
            raise ValueError("smollm2: export/materialize/load checks did not pass")
        if scaleout.get("distributed_execution_tested") is not False:
            raise ValueError("smollm2: distributed execution must remain not tested")


def _is_v011(version: str) -> bool:
    match = version.split(".", 2)
    try:
        return (int(match[0]), int(match[1])) >= (0, 11)
    except (IndexError, ValueError):
        return False


def _validate_v011_profiles(payload: dict[str, Any], qualification: GPUQualification) -> None:
    if payload.get("schema_version") != 1 or payload.get("status") != "passed":
        raise ValueError("v0.11 profiles: unsupported schema or failed status")
    if payload.get("kind") != "miniverl_v011_profile_qualification":
        raise ValueError("v0.11 profiles: unexpected evidence kind")
    if payload.get("source_commit") != qualification.source_commit:
        raise ValueError("v0.11 profiles: source commit does not match release smoke")
    if payload.get("miniverl_version") != qualification.miniverl_version:
        raise ValueError("v0.11 profiles: miniVERL version does not match release smoke")
    if payload.get("wheel_sha256") != qualification.wheel.sha256:
        raise ValueError("v0.11 profiles: wheel binding does not match release smoke")
    hardware = payload.get("hardware") or {}
    if hardware.get("gpu") != qualification.environment.gpu_name or hardware.get("gpu_count") != 1:
        raise ValueError("v0.11 profiles: hardware does not match release smoke")
    if "microsoft" not in str(hardware.get("platform", "")).lower():
        raise ValueError("v0.11 profiles: execution was not measured under WSL2")
    if hardware.get("python") != qualification.environment.python:
        raise ValueError("v0.11 profiles: Python does not match release smoke")
    if hardware.get("cuda_runtime") != qualification.environment.cuda_runtime:
        raise ValueError("v0.11 profiles: CUDA runtime does not match release smoke")
    if hardware.get("driver") != qualification.environment.driver:
        raise ValueError("v0.11 profiles: driver does not match release smoke")
    if hardware.get("packages") != qualification.environment.packages:
        raise ValueError("v0.11 profiles: packages do not match release smoke")
    expected = {
        "direct_n1": 1,
        "grouped_pg_n4": 4,
        "rewarded_pg_n4": 4,
    }
    profiles = payload.get("profiles") or {}
    if set(profiles) != set(expected):
        raise ValueError("v0.11 profiles: required profile set is incomplete")
    for name, samples in expected.items():
        result = profiles[name]
        if result.get("rollout_backend") != "hf_cached":
            raise ValueError(f"v0.11 profiles: {name} did not use hf_cached")
        if result.get("samples_per_prompt") != samples:
            raise ValueError(f"v0.11 profiles: {name} has the wrong sample count")
        if result.get("optimizer_updates") != 1 or result.get("policy_version") != 1:
            raise ValueError(f"v0.11 profiles: {name} did not commit one optimizer update")
        if result.get("loss_finite") is not True or int(result.get("selected_positions", 0)) < 1:
            raise ValueError(f"v0.11 profiles: {name} optimizer update is empty or non-finite")
        if int(result.get("trajectories", 0)) != 2 * samples:
            raise ValueError(f"v0.11 profiles: {name} trajectory group is incomplete")
        if float(result.get("peak_reserved_gib", math.inf)) > 14.5:
            raise ValueError(f"v0.11 profiles: {name} exceeded the 14.5 GiB limit")
    if profiles["rewarded_pg_n4"].get("reward", {}).get("status") != "completed":
        raise ValueError("v0.11 profiles: rewarded PG reward/advantage path did not complete")
    if (payload.get("cuda_teardown") or {}).get("passed") is not True:
        raise ValueError("v0.11 profiles: CUDA teardown did not pass")


def _cells(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cells = payload.get("cells") or []
    if len(cells) != 24:
        raise ValueError("v0.11 runtime: expected exactly 24 workload cells")
    indexed = {cell.get("cell_id"): cell for cell in cells}
    if len(indexed) != 24 or None in indexed:
        raise ValueError("v0.11 runtime: cell identities are not unique")
    expected = {
        f"p{prompt}-r{response}-n{samples}-{sampling}": (
            prompt,
            response,
            samples,
            sampling,
        )
        for prompt in (128, 512)
        for response in (64, 256, 512)
        for samples in (1, 4)
        for sampling in ("greedy", "seeded_stochastic")
    }
    if set(indexed) != set(expected):
        raise ValueError("v0.11 runtime: workload cell set does not match the preregistration")
    for cell_id, (prompt, response, samples, sampling) in expected.items():
        cell = indexed[cell_id]
        if (
            cell.get("prompt_length") != prompt
            or cell.get("response_bound") != response
            or cell.get("samples_per_prompt") != samples
            or cell.get("sampling") != sampling
            or cell.get("logical_prompts") != 4
            or cell.get("measured_repetitions") != 3
            or cell.get("generated_trajectories") != 4 * samples
            or cell.get("generated_tokens") != 4 * samples * response
        ):
            raise ValueError(f"v0.11 runtime: cell {cell_id} does not match its declared workload")
    return indexed


def _validate_refresh_and_teardown(payload: dict[str, Any], *, external: bool) -> None:
    refresh = payload.get("refresh_probe") or {}
    cycles = refresh.get("cycles") or []
    if (
        len(cycles) != 8
        or refresh.get("all_policy_identities_unique") is not True
        or refresh.get("all_syncs_confirmed") is not True
        or refresh.get("strictly_monotonic_memory_growth") is not False
    ):
        raise ValueError("v0.11 runtime: policy refresh/cache invalidation gate failed")
    teardown = payload.get("teardown") or {}
    if teardown.get("backend_state") != "closed" or (
        external and teardown.get("port_closed") is not True
    ):
        raise ValueError("v0.11 runtime: backend teardown gate failed")


def _validate_v011_runtime_pair(
    hf_payload: dict[str, Any],
    vllm_payload: dict[str, Any],
    reference: dict[str, Any],
    qualification: GPUQualification,
) -> None:
    for name, payload in (("hf_cached", hf_payload), ("vllm", vllm_payload)):
        if payload.get("schema_version") != 1 or payload.get("measurement_status") != "completed":
            raise ValueError(f"v0.11 runtime: {name} measurement did not complete")
        source = payload.get("source") or {}
        if source.get("commit") != qualification.source_commit:
            raise ValueError(f"v0.11 runtime: {name} source commit does not match")
        if source.get("miniverl_version") != qualification.miniverl_version:
            raise ValueError(f"v0.11 runtime: {name} version does not match")
        if source.get("wheel_sha256") != qualification.wheel.sha256:
            raise ValueError(f"v0.11 runtime: {name} wheel binding does not match")
        if source.get("dirty") is not False:
            raise ValueError(f"v0.11 runtime: {name} source tree was dirty")
        if payload.get("preregistration_sha256") != _PREREGISTRATION_SHA256:
            raise ValueError(f"v0.11 runtime: {name} preregistration binding does not match")
        if payload.get("frozen_calculator_sha256") != _FROZEN_CALCULATOR_SHA256:
            raise ValueError(f"v0.11 runtime: {name} frozen calculator binding does not match")
        if (payload.get("backend") or {}).get("name") != name:
            raise ValueError(f"v0.11 runtime: {name} backend identity does not match")
        environment = payload.get("environment") or {}
        if environment.get("gpu") != qualification.environment.gpu_name:
            raise ValueError(f"v0.11 runtime: {name} hardware does not match")
        if "microsoft" not in str(environment.get("platform", "")).lower():
            raise ValueError(f"v0.11 runtime: {name} was not measured under WSL2")
        for field, expected in (
            ("python", qualification.environment.python),
            ("driver_version", qualification.environment.driver),
            ("cuda_runtime", qualification.environment.cuda_runtime),
        ):
            if environment.get(field) != expected:
                raise ValueError(f"v0.11 runtime: {name} environment {field} does not match")
        runtime_packages = environment.get("packages") or {}
        for package, actual in runtime_packages.items():
            if package == "vllm":
                if actual != "0.28.0":
                    raise ValueError("v0.11 runtime: vLLM version does not match")
            elif qualification.environment.packages.get(package) != actual:
                raise ValueError(
                    f"v0.11 runtime: {name} package {package} does not match release smoke"
                )
        if (
            float((payload.get("memory") or {}).get("peak_total_gpu_memory_mib", math.inf))
            > _MAX_GPU_MEMORY_MIB
        ):
            raise ValueError(f"v0.11 runtime: {name} exceeded the 14.5 GiB limit")
    hf_cells = _cells(hf_payload)
    vllm_cells = _cells(vllm_payload)
    reference_cells = {cell["cell_id"]: cell for cell in reference.get("cells") or []}
    for cell_id, cell in hf_cells.items():
        if int(cell["response_bound"]) not in {256, 512}:
            continue
        baseline = reference_cells.get(cell_id)
        if baseline is None:
            raise ValueError(f"v0.11 runtime: missing reference cell {cell_id}")
        baseline_rate = float(baseline["rates"]["output_tokens_per_second"])
        if float(cell["output_tokens_per_second"]) < 2.0 * baseline_rate:
            raise ValueError(f"v0.11 runtime: hf_cached cell {cell_id} missed the 2.0x gate")
        if float(vllm_cells[cell_id]["output_tokens_per_second"]) < 1.2 * float(
            cell["output_tokens_per_second"]
        ):
            raise ValueError(f"v0.11 runtime: vLLM cell {cell_id} missed the 1.2x gate")
    hf_conformance = hf_payload.get("conformance") or {}
    if (
        hf_conformance.get("tokens_equal") is not True
        or hf_conformance.get("threshold_passed") is not True
    ):
        raise ValueError("v0.11 runtime: hf_cached conformance gate failed")
    vllm_conformance = vllm_payload.get("conformance") or {}
    if vllm_conformance.get("token_agreement_fraction") != 1.0:
        raise ValueError("v0.11 runtime: vLLM greedy token conformance failed")
    if vllm_conformance.get("pg_threshold_passed") is not False:
        raise ValueError("v0.11 runtime: vLLM PG path must remain fail closed")
    lifecycle = (vllm_payload.get("timing") or {}).get("engine_lifecycle") or {}
    if (
        lifecycle.get("execution_mode") != "cuda_graph"
        or lifecycle.get("prefix_cache_enabled") is not False
    ):
        raise ValueError("v0.11 runtime: vLLM execution mode is not qualified")
    _validate_refresh_and_teardown(hf_payload, external=False)
    _validate_refresh_and_teardown(vllm_payload, external=True)


def promote(
    qualification_path: Path,
    *,
    direct: Path,
    pg_k1: Path,
    smollm2: Path,
    v011_profiles: Path | None = None,
    hf_cached_runtime: Path | None = None,
    vllm_runtime: Path | None = None,
    hf_reference: Path | None = None,
) -> GPUQualification:
    problems = validate_qualification_file(qualification_path)
    if problems:
        raise ValueError("invalid release smoke: " + "; ".join(problems))
    payload = _load(qualification_path)
    qualification = GPUQualification.model_validate(payload)
    if qualification.level != "release_smoke":
        raise ValueError("only a release_smoke artifact can be promoted")
    sources = {"direct": direct, "pg_k1": pg_k1, "smollm2": smollm2}
    results = {name: _load(path) for name, path in sources.items()}
    for name, result in results.items():
        _validate_result(name, result, qualification)
    v011_sources: dict[str, Path] = {}
    if _is_v011(qualification.miniverl_version):
        if None in (v011_profiles, hf_cached_runtime, vllm_runtime, hf_reference):
            raise ValueError("v0.11 full qualification requires profile and runtime evidence")
        assert v011_profiles is not None
        assert hf_cached_runtime is not None
        assert vllm_runtime is not None
        assert hf_reference is not None
        profile_payload = _load(v011_profiles)
        hf_payload = _load(hf_cached_runtime)
        vllm_payload = _load(vllm_runtime)
        _validate_v011_profiles(profile_payload, qualification)
        _validate_v011_runtime_pair(hf_payload, vllm_payload, _load(hf_reference), qualification)
        v011_sources = {
            "v011_profiles": v011_profiles,
            "hf_cached_runtime": hf_cached_runtime,
            "vllm_runtime": vllm_runtime,
        }

    root = qualification_path.parent
    destination = root / "full"
    destination.mkdir(parents=True, exist_ok=False)
    smoke_copy = destination / "release-smoke.json"
    shutil.copy2(qualification_path, smoke_copy)
    additions = [("release_smoke_record", smoke_copy)]
    for name, source in sources.items():
        target = destination / f"{name}.json"
        shutil.copy2(source, target)
        additions.append((f"full_{name}_result", target))
    for name, source in v011_sources.items():
        target = destination / f"{name.replace('_', '-')}.json"
        shutil.copy2(source, target)
        additions.append((f"full_{name}_result", target))
    for name, path in additions:
        payload["artifacts"].append(
            {
                "name": name,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    payload["level"] = "full_qualification"
    payload["checks"]["executed"].extend(
        [
            "direct_gkd_resume_equivalence",
            "sampled_k1_resume_equivalence",
            "smollm2_resume_equivalence",
            "export_materialize_doctor",
        ]
    )
    if v011_sources:
        payload["checks"]["executed"].extend(_V011_CHECKS)
    promoted = GPUQualification.model_validate(payload)
    write_json_atomic(qualification_path, promoted.model_dump(mode="json"))
    final_problems = validate_qualification_file(qualification_path)
    if final_problems:
        raise ValueError("promoted qualification is invalid: " + "; ".join(final_problems))
    return promoted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--direct", required=True, type=Path)
    parser.add_argument("--pg-k1", required=True, type=Path)
    parser.add_argument("--smollm2", required=True, type=Path)
    parser.add_argument("--v011-profiles", type=Path)
    parser.add_argument("--hf-cached-runtime", type=Path)
    parser.add_argument("--vllm-runtime", type=Path)
    parser.add_argument("--hf-reference", type=Path)
    args = parser.parse_args()
    promoted = promote(
        args.qualification,
        direct=args.direct,
        pg_k1=args.pg_k1,
        smollm2=args.smollm2,
        v011_profiles=args.v011_profiles,
        hf_cached_runtime=args.hf_cached_runtime,
        vllm_runtime=args.vllm_runtime,
        hf_reference=args.hf_reference,
    )
    print(json.dumps(promoted.model_dump(mode="json"), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
