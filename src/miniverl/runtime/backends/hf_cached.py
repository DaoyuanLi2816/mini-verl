"""Batched, KV-cache-aware Hugging Face generation backend."""

from __future__ import annotations

from collections.abc import Sequence

from miniverl.models.base import GenerationOutput
from miniverl.runtime.backends._base import LocalGenerationBackend, timed_call
from miniverl.runtime.generation import GenerationRequest, RolloutBackendKind

__all__ = ["HFCachedGenerationBackend"]


class HFCachedGenerationBackend(LocalGenerationBackend):
    """One padded prefill followed by incremental cached decode steps."""

    kind = RolloutBackendKind.HF_CACHED
    backend_version = "hf_cached-v1"

    def __init__(self, model_backend: object, *, compile_backend: bool = False) -> None:
        super().__init__(model_backend)
        if compile_backend:
            enable = getattr(model_backend, "enable_cached_generation_compilation", None)
            if not callable(enable):
                raise RuntimeError(
                    "rollout.compile_backend=true requires a compilable Hugging Face backend"
                )
            enable()
            self.backend_version = "hf_cached-v1+inductor-no-cudagraph"

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
