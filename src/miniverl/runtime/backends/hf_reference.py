"""Compatibility backend preserving pre-v0.11 per-request HF generation."""

from __future__ import annotations

from collections.abc import Sequence

from miniverl.models.base import GenerationOutput
from miniverl.runtime.backends._base import LocalGenerationBackend, timed_call
from miniverl.runtime.generation import GenerationRequest, RolloutBackendKind

__all__ = ["HFReferenceGenerationBackend"]


class HFReferenceGenerationBackend(LocalGenerationBackend):
    """Established sequential generation path retained as a reference."""

    kind = RolloutBackendKind.HF_REFERENCE
    backend_version = "hf_reference-v1"

    def _generate_outputs(
        self, requests: Sequence[GenerationRequest]
    ) -> tuple[list[GenerationOutput], tuple[int, ...], float]:
        def invoke() -> list[GenerationOutput]:
            first = requests[0]
            signature = (
                first.max_new_tokens,
                first.sampling,
                first.stop_sequences,
                first.need_sampled_token_logprobs,
            )
            if any(
                (
                    request.max_new_tokens,
                    request.sampling,
                    request.stop_sequences,
                    request.need_sampled_token_logprobs,
                )
                != signature
                for request in requests[1:]
            ):
                raise ValueError("hf_reference physical batches must share sampling parameters")
            return self.model_backend.generate_batch(
                [request.prompt_token_ids for request in requests],
                max_new_tokens=first.max_new_tokens,
                stop_sequences=first.stop_sequences,
                temperature=first.sampling.temperature,
                top_p=first.sampling.top_p,
                top_k=first.sampling.top_k,
                seeds=[request.deterministic_sample_seed for request in requests],
                record_logprobs=first.need_sampled_token_logprobs,
            )

        outputs, elapsed = timed_call(invoke)
        return outputs, (len(requests),), elapsed
