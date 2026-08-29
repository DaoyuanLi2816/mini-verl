"""Typed deterministic task rewards and advantage composition."""

from miniverl.rewards.composer import (
    ADVANTAGE_COMPOSER_VERSION,
    AdvantageComposer,
    AdvantageComposition,
    AdvantageMode,
)
from miniverl.rewards.models import (
    RewardComponent,
    RewardProviderIdentity,
    RewardRequest,
    RewardResult,
    RewardStatus,
)
from miniverl.rewards.providers import (
    EnvironmentVerifierRewardProvider,
    ExactAnswerRewardProvider,
    RewardProvider,
)

__all__ = [
    "ADVANTAGE_COMPOSER_VERSION",
    "AdvantageComposition",
    "AdvantageComposer",
    "AdvantageMode",
    "EnvironmentVerifierRewardProvider",
    "ExactAnswerRewardProvider",
    "RewardComponent",
    "RewardProvider",
    "RewardProviderIdentity",
    "RewardRequest",
    "RewardResult",
    "RewardStatus",
]
