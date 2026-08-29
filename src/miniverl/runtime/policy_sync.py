"""Content identities for strict same-process actor-to-rollout synchronization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from miniverl.runtime.generation import RolloutBackendKind, RolloutPolicyIdentity

__all__ = ["adapter_tensor_digest", "build_rollout_policy_identity"]


def _digest_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def adapter_tensor_digest(backend: Any) -> str:
    """Hash live trainable actor tensors without importing torch at module import time."""

    model = getattr(backend, "model", None)
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return _digest_payload({"trainable_tensors": []})
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(named_parameters(), key=lambda item: item[0]):
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        tensor = parameter.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        raw = tensor.view(-1).view(dtype=__import__("torch").uint8).cpu().numpy().tobytes()
        digest.update(raw)
        count += 1
    if count == 0:
        digest.update(b"no-trainable-adapter-tensors")
    return digest.hexdigest()


def build_rollout_policy_identity(
    *,
    backend: Any,
    parameter_version: int,
    generation_backend: RolloutBackendKind,
    backend_version: str,
    profile_identity: object | None,
    execution_plan_digest: str | None,
) -> RolloutPolicyIdentity:
    """Bind live actor state and run identity to a generation request."""

    tokenizer_identity = getattr(getattr(backend, "tokenizer", None), "identity", {})
    structural = (
        tokenizer_identity.get("structural_digest_v2")
        if isinstance(tokenizer_identity, Mapping)
        else None
    )
    if not isinstance(structural, str):
        structural = _digest_payload({"tokenizer_identity": tokenizer_identity})
    provenance = getattr(backend, "adapter_provenance", None)
    manifest_digest = provenance.get("manifest_digest") if isinstance(provenance, Mapping) else None
    if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
        manifest_digest = _digest_payload({"adapter_provenance": provenance})
    capabilities = backend.capabilities
    return RolloutPolicyIdentity(
        parameter_version=parameter_version,
        base_model_id=str(getattr(backend, "model_id", capabilities.name)),
        base_model_revision=getattr(backend, "model_revision", None),
        tokenizer_structural_identity=structural,
        student_adapter_manifest_digest=manifest_digest,
        adapter_tensor_digest=adapter_tensor_digest(backend),
        quantization=str(capabilities.quantization),
        dtype=str(capabilities.dtype),
        generation_backend=generation_backend,
        backend_version=backend_version,
        profile_identity=(
            profile_identity
            if isinstance(profile_identity, str) and len(profile_identity) == 64
            else _digest_payload({"profile_identity": profile_identity})
        ),
        execution_plan_digest=(
            execution_plan_digest or _digest_payload({"execution_plan_digest": None})
        ),
    )
