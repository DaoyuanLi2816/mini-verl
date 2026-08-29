"""Batched, KV-cache-aware Hugging Face generation backend."""

from __future__ import annotations

from collections.abc import Sequence

from miniverl.models.base import GenerationOutput
from miniverl.runtime.backends._base import LocalGenerationBackend, timed_call
from miniverl.runtime.generation import (
    GenerationRequest,
    PolicySnapshot,
    PolicySyncResult,
    RolloutBackendKind,
)

__all__ = ["HFCachedGenerationBackend"]


class HFCachedGenerationBackend(LocalGenerationBackend):
    """One padded prefill followed by incremental cached decode steps."""

    kind = RolloutBackendKind.HF_CACHED
    backend_version = "hf_cached-v1"

    def __init__(self, model_backend: object, *, compile_backend: bool = False) -> None:
        self._source_model_backend = model_backend
        self._owns_generation_mirror = False
        generation_backend = model_backend
        if compile_backend:
            quantization = str(
                getattr(getattr(model_backend, "capabilities", None), "quantization", "none")
            )
            if quantization == "nf4":
                build_mirror = getattr(model_backend, "build_cached_generation_mirror", None)
                if not callable(build_mirror):
                    raise RuntimeError(
                        "compiled NF4 hf_cached generation requires a verified BF16 mirror builder"
                    )
                generation_backend = build_mirror()
                self._owns_generation_mirror = True
                self.backend_version = "hf_cached-v2+nf4-bf16-mirror+inductor"
            enable = getattr(generation_backend, "enable_cached_generation_compilation", None)
            if not callable(enable):
                self._discard_owned_mirror(generation_backend)
                raise RuntimeError(
                    "rollout.compile_backend=true requires a compilable Hugging Face backend"
                )
            try:
                enable()
            except BaseException:
                self._discard_owned_mirror(generation_backend)
                raise
            if not self._owns_generation_mirror:
                self.backend_version = "hf_cached-v1+inductor-no-cudagraph"
        super().__init__(generation_backend)

    def _discard_owned_mirror(self, backend: object) -> None:
        if not self._owns_generation_mirror:
            return
        discard = getattr(backend, "discard_cached_generation_mirror", None)
        if callable(discard):
            discard()
        self._owns_generation_mirror = False

    def synchronize(self, snapshot: PolicySnapshot) -> PolicySyncResult:
        """Copy the exact live adapter into an owned mirror before activation."""

        if self._owns_generation_mirror:
            synchronize = getattr(
                self._source_model_backend, "synchronize_cached_generation_mirror", None
            )
            if not callable(synchronize):
                raise RuntimeError("NF4 generation mirror has no strict synchronization hook")
            synchronize(
                self.model_backend,
                expected_adapter_digest=snapshot.identity.adapter_tensor_digest,
            )
        return super().synchronize(snapshot)

    @staticmethod
    def _signature(request: GenerationRequest) -> tuple[object, ...]:
        return (
            request.max_new_tokens,
            request.sampling,
            request.stop_sequences,
            request.need_sampled_token_logprobs,
        )

    def _generate_outputs(
        self, requests: Sequence[GenerationRequest]
    ) -> tuple[list[GenerationOutput], tuple[int, ...], float]:
        cached = getattr(self.model_backend, "generate_batch_cached", None)
        if not callable(cached):
            raise RuntimeError(
                "hf_cached needs a model backend with generate_batch_cached(); "
                "select rollout.backend=hf_reference for the compatibility path"
            )
        groups: list[list[tuple[int, GenerationRequest]]] = []
        group_by_signature: dict[tuple[object, ...], int] = {}
        for index, request in enumerate(requests):
            signature = self._signature(request)
            target = group_by_signature.get(signature)
            if target is None:
                target = len(groups)
                group_by_signature[signature] = target
                groups.append([])
            groups[target].append((index, request))

        ordered: list[GenerationOutput | None] = [None] * len(requests)
        physical_sizes: list[int] = []

        def invoke() -> None:
            for group in groups:
                representative = group[0][1]
                outputs = cached(
                    [request.prompt_token_ids for _, request in group],
                    max_new_tokens=representative.max_new_tokens,
                    stop_sequences=representative.stop_sequences,
                    temperature=representative.sampling.temperature,
                    top_p=representative.sampling.top_p,
                    top_k=representative.sampling.top_k,
                    seeds=[request.deterministic_sample_seed for _, request in group],
                    record_logprobs=representative.need_sampled_token_logprobs,
                )
                if len(outputs) != len(group):
                    raise RuntimeError(
                        f"hf_cached returned {len(outputs)} rows for physical batch {len(group)}"
                    )
                physical_sizes.append(len(group))
                for (logical_index, _), output in zip(group, outputs, strict=True):
                    ordered[logical_index] = output

        _, elapsed = timed_call(invoke)
        if any(output is None for output in ordered):
            raise RuntimeError("hf_cached lost a logical generation result")
        return [output for output in ordered if output is not None], tuple(physical_sizes), elapsed

    def close(self) -> None:
        mirror = self.model_backend
        super().close()
        self._discard_owned_mirror(mirror)
        self.model_backend = self._source_model_backend
