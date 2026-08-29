"""Concrete Rollout Runtime v2 generation backends."""

from miniverl.runtime.backends.hf_cached import HFCachedGenerationBackend
from miniverl.runtime.backends.hf_reference import HFReferenceGenerationBackend
from miniverl.runtime.backends.vllm import VLLMGenerationBackend

__all__ = ["HFCachedGenerationBackend", "HFReferenceGenerationBackend", "VLLMGenerationBackend"]
