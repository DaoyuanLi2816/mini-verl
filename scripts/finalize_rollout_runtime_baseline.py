#!/usr/bin/env python3
"""Bind immutable identity metadata to a completed rollout baseline.

This finalization step never changes a measured cell. It preserves the raw
measurement bytes, adds the exact base-weight digest and policy identity, then
validates and atomically publishes the public result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jsonschema
from huggingface_hub import snapshot_download

from miniverl.utils.runs import canonical_json, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "benchmarks/schema/rollout-runtime-v2.schema.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _weight_digest(model_id: str, revision: str) -> str:
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=["*.safetensors"],
            local_files_only=True,
        )
    )
    weights = sorted(snapshot.rglob("*.safetensors"), key=lambda path: path.as_posix())
    if not weights:
        raise RuntimeError(f"cached snapshot {model_id}@{revision} has no safetensors weights")
    digest = hashlib.sha256()
    for path in weights:
        relative = path.relative_to(snapshot).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _driver_version() -> str:
    return (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .splitlines()[0]
        .strip()
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize(*, measured: Path, raw_output: Path, result_output: Path) -> dict[str, Any]:
    raw = measured.read_bytes()
    payload = json.loads(raw)
    if payload.get("measurement_status") not in {"measured_baseline", "completed_with_failures"}:
        raise RuntimeError("input is not a completed rollout baseline")
    if "raw_measurement_sha256" in payload or "policy_identity" in payload:
        raise RuntimeError("input already contains finalization metadata")
    source_commit = payload.get("source", {}).get("commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise RuntimeError("raw measurement has no exact source commit")

    actor = payload["models"]["actor"]
    policy_identity = {
        "policy_version": 0,
        "profile_identity": "rollout-runtime-v2-hf-reference-baseline-v1",
        "base_revision": actor["revision"],
        "base_weight_digest_sha256": _weight_digest(actor["id"], actor["revision"]),
        "adapter_digest_sha256": None,
    }
    policy_identity["identity_digest_sha256"] = hashlib.sha256(
        canonical_json(policy_identity).encode("utf-8")
    ).hexdigest()
    payload["raw_measurement_sha256"] = _sha256_bytes(raw)
    payload["policy_identity"] = policy_identity
    payload["environment"]["driver_version"] = _driver_version()

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    result_output.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(raw_output, raw)
    write_json(result_output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
    payload = finalize(
        measured=args.measured.resolve(),
        raw_output=args.raw_output.resolve(),
        result_output=args.result_output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": payload["measurement_status"],
                "raw_measurement_sha256": payload["raw_measurement_sha256"],
                "policy_identity": payload["policy_identity"]["identity_digest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
