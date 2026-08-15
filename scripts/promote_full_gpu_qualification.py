"""Bind the three canonical full workloads into a strict qualification artifact."""

from __future__ import annotations

import argparse
import json
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


def promote(
    qualification_path: Path,
    *,
    direct: Path,
    pg_k1: Path,
    smollm2: Path,
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
    args = parser.parse_args()
    promoted = promote(
        args.qualification,
        direct=args.direct,
        pg_k1=args.pg_k1,
        smollm2=args.smollm2,
    )
    print(json.dumps(promoted.model_dump(mode="json"), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
