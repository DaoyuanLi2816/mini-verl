"""Torch-free typed contract for Rollout Runtime v2 generation backends."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

__all__ = [
    "BackendCapabilities",
    "BackendLifecycleState",
    "BackendMetrics",
    "GenerationBackend",
    "GenerationBatch",
    "GenerationRequest",
    "GenerationResult",
    "PolicySnapshot",
    "PolicySyncResult",
    "ReproducibilityClass",
    "RolloutBackendKind",
    "RolloutGroupIdentity",
    "RolloutPolicyIdentity",
    "SamplingParameters",
    "derive_sample_seed",
]

_SEED_DERIVATION_VERSION = "miniverl-rollout-seed-v1"


class RolloutBackendKind(str, Enum):
    """Generation implementations supported by the v2 runtime."""

    HF_REFERENCE = "hf_reference"
    HF_CACHED = "hf_cached"
    VLLM = "vllm"


class BackendLifecycleState(str, Enum):
    """Fail-closed generation backend lifecycle."""

    NEW = "new"
    SYNCHRONIZED = "synchronized"
    QUIESCED = "quiesced"
    CLOSED = "closed"


class ReproducibilityClass(str, Enum):
    """Strength of a backend's replay contract."""

    BATCH_INVARIANT = "batch_invariant"
    DETERMINISTIC_GREEDY = "deterministic_greedy"
    BEST_EFFORT = "best_effort"


def _require_digest(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SamplingParameters:
    """Sampling controls whose meaning is shared by every backend."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 5.0:
            raise ValueError("temperature must be between 0 and 5")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")


@dataclass(frozen=True)
class RolloutGroupIdentity:
    """Logical sample identity, independent of physical batching."""

    prompt_group_id: str
    prompt_digest: str
    sample_index: int
    samples_per_prompt: int

    def __post_init__(self) -> None:
        if not self.prompt_group_id:
            raise ValueError("prompt_group_id cannot be empty")
        _require_digest("prompt_digest", self.prompt_digest)
        if self.samples_per_prompt < 1:
            raise ValueError("samples_per_prompt must be at least one")
        if not 0 <= self.sample_index < self.samples_per_prompt:
            raise ValueError("sample_index must be within samples_per_prompt")


@dataclass(frozen=True)
class RolloutPolicyIdentity:
    """Content-bound identity of the actor policy used for generation."""

    parameter_version: int
    base_model_id: str
    base_model_revision: str | None
    tokenizer_structural_identity: str
    student_adapter_manifest_digest: str
    adapter_tensor_digest: str
    quantization: str
    dtype: str
    generation_backend: RolloutBackendKind
    backend_version: str
    profile_identity: str
    execution_plan_digest: str

    def __post_init__(self) -> None:
        if self.parameter_version < 0:
            raise ValueError("parameter_version must be non-negative")
        if not self.base_model_id or not self.quantization or not self.dtype:
            raise ValueError("model, quantization and dtype identities cannot be empty")
        if not isinstance(self.generation_backend, RolloutBackendKind):
            raise ValueError("generation backend must be a RolloutBackendKind")
        if not self.backend_version:
            raise ValueError("backend_version cannot be empty")
        for name in (
            "tokenizer_structural_identity",
            "student_adapter_manifest_digest",
            "adapter_tensor_digest",
            "profile_identity",
            "execution_plan_digest",
        ):
            _require_digest(name, str(getattr(self, name)))

    @property
    def digest(self) -> str:
        """Canonical digest suitable for manifests and request matching."""

        payload = asdict(self)
        payload["generation_backend"] = self.generation_backend.value
        return _canonical_digest(payload)


@dataclass(frozen=True)
class PolicySnapshot:
    """One policy identity offered to a generation backend for synchronization."""

    identity: RolloutPolicyIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.identity.generation_backend, RolloutBackendKind):
            raise ValueError("generation backend must be a RolloutBackendKind")


@dataclass(frozen=True)
class PolicySyncResult:
    """Auditable result of synchronizing one backend to one snapshot."""

    previous_policy_digest: str | None
    active_policy_digest: str
    changed: bool
    state: BackendLifecycleState


@dataclass(frozen=True)
class BackendCapabilities:
    """Portable backend capability declaration."""

    kind: RolloutBackendKind
    backend_version: str
    supports_greedy: bool
    supports_seeded_sampling: bool
    supports_sampled_token_logprobs: bool
    supports_text_stops: bool
    reproducibility: ReproducibilityClass

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe manifest representation."""

        return {
            "kind": self.kind.value,
            "backend_version": self.backend_version,
            "supports_greedy": self.supports_greedy,
            "supports_seeded_sampling": self.supports_seeded_sampling,
            "supports_sampled_token_logprobs": self.supports_sampled_token_logprobs,
            "supports_text_stops": self.supports_text_stops,
            "reproducibility": self.reproducibility.value,
        }


@dataclass(frozen=True)
class BackendMetrics:
    """Per-request metrics without claiming unavailable phase boundaries."""

    total_seconds: float
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float | None = None
    decode_seconds: float | None = None
    peak_reserved_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.total_seconds < 0 or self.prompt_tokens < 0 or self.generated_tokens < 0:
            raise ValueError("backend metrics cannot be negative")


@dataclass(frozen=True)
class GenerationRequest:
    """One logical generation request with explicit policy and sample identity."""

    request_id: str
    group: RolloutGroupIdentity
    deterministic_sample_seed: int
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    sampling: SamplingParameters = field(default_factory=SamplingParameters)
    stop_sequences: tuple[str, ...] = ()
    need_sampled_token_logprobs: bool = False
    expected_policy_identity: RolloutPolicyIdentity | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if not self.prompt_token_ids or any(token < 0 for token in self.prompt_token_ids):
            raise ValueError("prompt_token_ids must contain non-negative token ids")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least one")
        if self.deterministic_sample_seed < 0:
            raise ValueError("deterministic_sample_seed must be non-negative")
        if self.expected_policy_identity is None:
            raise ValueError("expected_policy_identity is required")


@dataclass(frozen=True)
class GenerationResult:
    """One generated sample with exact request and policy provenance."""

    request_id: str
    group: RolloutGroupIdentity
    output_token_ids: tuple[int, ...]
    decoded_text: str
    sampled_token_logprobs: tuple[float, ...]
    stop_reason: str
    matched_stop: str | None
    policy_identity: RolloutPolicyIdentity
    backend_metrics: BackendMetrics
    raw_backend_request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.output_token_ids:
            raise ValueError("generation result cannot be empty")
        if self.sampled_token_logprobs and len(self.sampled_token_logprobs) != len(
            self.output_token_ids
        ):
            raise ValueError("sampled-token logprobs must align with output token ids")


@dataclass(frozen=True)
class GenerationBatch:
    """Results in the same logical order as their requests."""

    results: tuple[GenerationResult, ...]
    policy_identity: RolloutPolicyIdentity
    physical_batch_sizes: tuple[int, ...]
    oom_downshifts: int = 0

    def __post_init__(self) -> None:
        if any(result.policy_identity != self.policy_identity for result in self.results):
            raise ValueError("one generation batch cannot mix policy identities")
        if any(size < 1 for size in self.physical_batch_sizes):
            raise ValueError("physical batch sizes must be positive")
        if self.oom_downshifts < 0:
            raise ValueError("oom_downshifts cannot be negative")


def derive_sample_seed(
    *, run_seed: int, prompt_digest: str, policy_version: int, sample_index: int
) -> int:
    """Derive a stable per-sample CPU RNG seed without physical-batch inputs."""

    if run_seed < 0 or policy_version < 0 or sample_index < 0:
        raise ValueError("seed inputs must be non-negative")
    _require_digest("prompt_digest", prompt_digest)
    payload = (
        f"{_SEED_DERIVATION_VERSION}\0{run_seed}\0{prompt_digest}\0{policy_version}\0{sample_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@runtime_checkable
class GenerationBackend(Protocol):
    """Lifecycle and generation protocol implemented by rollout backends."""

    @property
    def state(self) -> BackendLifecycleState: ...

    def inspect(self) -> BackendCapabilities: ...

    def synchronize(self, snapshot: PolicySnapshot) -> PolicySyncResult: ...

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch: ...

    def quiesce(self) -> None: ...

    def release_generation_memory(self) -> None: ...

    def close(self) -> None: ...
