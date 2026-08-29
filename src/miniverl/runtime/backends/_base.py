"""Shared fail-closed lifecycle for local generation adapters."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any

from miniverl.models.base import GenerationOutput
from miniverl.runtime.generation import (
    BackendCapabilities,
    BackendLifecycleState,
    BackendMetrics,
    GenerationBatch,
    GenerationRequest,
    GenerationResult,
    PolicySnapshot,
    PolicySyncResult,
    ReproducibilityClass,
    RolloutBackendKind,
    RolloutPolicyIdentity,
)


class LocalGenerationBackend:
    """Policy-bound adapter around an in-process causal-LM backend."""

    kind: RolloutBackendKind
    backend_version: str

    def __init__(self, model_backend: Any) -> None:
        self.model_backend = model_backend
        self._state = BackendLifecycleState.NEW
        self._active_identity: RolloutPolicyIdentity | None = None

    @property
    def state(self) -> BackendLifecycleState:
        return self._state

    def inspect(self) -> BackendCapabilities:
        return BackendCapabilities(
            kind=self.kind,
            backend_version=self.backend_version,
            supports_greedy=True,
            supports_seeded_sampling=True,
            supports_sampled_token_logprobs=True,
            supports_text_stops=True,
            reproducibility=ReproducibilityClass.BATCH_INVARIANT,
        )

    def synchronize(self, snapshot: PolicySnapshot) -> PolicySyncResult:
        if self._state is BackendLifecycleState.CLOSED:
            raise RuntimeError("generation backend is closed")
        identity = snapshot.identity
        if identity.generation_backend is not self.kind:
            raise RuntimeError(
                f"policy identity selects {identity.generation_backend.value}, not {self.kind.value}"
            )
        if identity.backend_version != self.backend_version:
            raise RuntimeError(
                f"policy backend version {identity.backend_version!r} does not match "
                f"runtime {self.backend_version!r}"
            )
        previous = self._active_identity.digest if self._active_identity is not None else None
        self._active_identity = identity
        self._state = BackendLifecycleState.SYNCHRONIZED
        return PolicySyncResult(
            previous_policy_digest=previous,
            active_policy_digest=identity.digest,
            changed=previous != identity.digest,
            state=self._state,
        )

    def _validate_requests(self, requests: Sequence[GenerationRequest]) -> RolloutPolicyIdentity:
        if self._state is not BackendLifecycleState.SYNCHRONIZED:
            raise RuntimeError("generation backend must be synchronized before generate()")
        if not requests:
            raise ValueError("generation request batch cannot be empty")
        assert self._active_identity is not None
        active = self._active_identity
        seen: set[str] = set()
        for request in requests:
            if request.request_id in seen:
                raise ValueError(f"duplicate generation request id {request.request_id!r}")
            seen.add(request.request_id)
            if request.expected_policy_identity != active:
                raise RuntimeError(
                    f"request {request.request_id!r} policy identity does not match the "
                    "synchronized actor policy"
                )
        return active

    @staticmethod
    def _metrics(
        request: GenerationRequest, output: GenerationOutput, elapsed: float, divisor: int
    ) -> BackendMetrics:
        return BackendMetrics(
            total_seconds=elapsed / divisor,
            prompt_tokens=len(request.prompt_token_ids),
            generated_tokens=len(output.token_ids),
        )

    @classmethod
    def _result(
        cls,
        request: GenerationRequest,
        output: GenerationOutput,
        identity: RolloutPolicyIdentity,
        elapsed: float,
        divisor: int,
    ) -> GenerationResult:
        return GenerationResult(
            request_id=request.request_id,
            group=request.group,
            output_token_ids=tuple(output.token_ids),
            decoded_text=output.text,
            sampled_token_logprobs=tuple(output.logprobs),
            stop_reason=output.stop_reason,
            matched_stop=output.matched_stop,
            policy_identity=identity,
            backend_metrics=cls._metrics(request, output, elapsed, divisor),
        )

    def _generate_outputs(
        self, requests: Sequence[GenerationRequest]
    ) -> tuple[list[GenerationOutput], tuple[int, ...], float]:
        raise NotImplementedError

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        identity = self._validate_requests(requests)
        outputs, physical_sizes, elapsed = self._generate_outputs(requests)
        if len(outputs) != len(requests):
            raise RuntimeError(
                f"backend returned {len(outputs)} results for {len(requests)} requests"
            )
        divisor = max(len(requests), 1)
        return GenerationBatch(
            results=tuple(
                self._result(request, output, identity, elapsed, divisor)
                for request, output in zip(requests, outputs, strict=True)
            ),
            policy_identity=identity,
            physical_batch_sizes=physical_sizes,
        )

    def quiesce(self) -> None:
        if self._state is BackendLifecycleState.CLOSED:
            raise RuntimeError("generation backend is closed")
        if self._state is not BackendLifecycleState.SYNCHRONIZED:
            raise RuntimeError("only a synchronized generation backend can quiesce")
        self._state = BackendLifecycleState.QUIESCED

    def release_generation_memory(self) -> None:
        if self._state is BackendLifecycleState.CLOSED:
            raise RuntimeError("generation backend is closed")
        release = getattr(self.model_backend, "release_generation_memory", None)
        if callable(release):
            release()
            return
        device = str(getattr(getattr(self.model_backend, "capabilities", None), "device", ""))
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    def close(self) -> None:
        self._active_identity = None
        self._state = BackendLifecycleState.CLOSED


def timed_call(callable_: Any) -> tuple[Any, float]:
    """Return a callable's result and monotonic elapsed time."""

    started = perf_counter()
    result = callable_()
    return result, perf_counter() - started
