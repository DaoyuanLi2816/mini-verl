"""Concrete Rollout Runtime v2 generation backends."""

from miniverl.runtime.backends.hf_cached import HFCachedGenerationBackend
from miniverl.runtime.backends.hf_reference import HFReferenceGenerationBackend

__all__ = ["HFCachedGenerationBackend", "HFReferenceGenerationBackend"]
