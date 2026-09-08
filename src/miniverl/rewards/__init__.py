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
    BatchRewardProvider,
    EnvironmentVerifierRewardProvider,
    ExactAnswerRewardProvider,
    RewardProvider,
    TargetLengthRewardProvider,
    WeightedRewardProvider,
    score_reward_requests,
)

__all__ = [
    "ADVANTAGE_COMPOSER_VERSION",
    "AdvantageComposition",
    "AdvantageComposer",
    "AdvantageMode",
    "BatchRewardProvider",
    "EnvironmentVerifierRewardProvider",
    "ExactAnswerRewardProvider",
    "TargetLengthRewardProvider",
    "RewardComponent",
    "RewardProvider",
    "RewardProviderIdentity",
    "RewardRequest",
    "RewardResult",
    "RewardStatus",
    "WeightedRewardProvider",
    "score_reward_requests",
]
