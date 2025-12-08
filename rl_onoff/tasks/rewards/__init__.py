"""Rewards framework for evaluating model outputs."""

from rl_onoff.tasks.rewards.base import BaseReward, RewardRegistry
from rl_onoff.tasks.rewards.builtin import (
    PerplexityReward,
    BLEUReward,
    ROUGEReward,
    ExactMatchReward,
    MathVerifyReward,
)

__all__ = [
    "BaseReward",
    "RewardRegistry",
    "PerplexityReward",
    "BLEUReward",
    "ROUGEReward",
    "ExactMatchReward",
    "MathVerifyReward",
]

