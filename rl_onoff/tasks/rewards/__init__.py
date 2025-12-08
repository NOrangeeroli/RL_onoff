"""Rewards framework for evaluating model outputs."""

from rl_onoff.tasks.rewards.base import BaseReward, RewardRegistry
from rl_onoff.tasks.rewards.builtin import (
    PerplexityReward,
    BLEUReward,
    ROUGEReward,
    ExactMatchReward,
    MathVerifyReward,
)

# Registry mapping string names to reward classes
REWARD_REGISTRY = {
    "math_verify": MathVerifyReward,
    "exact_match": ExactMatchReward,
    "bleu": BLEUReward,
    "rouge": ROUGEReward,
    "perplexity": PerplexityReward,
}


def create_reward(name: str, **kwargs) -> BaseReward:
    """Create a reward instance from a name.
    
    Args:
        name: Name of the reward ("math_verify", "exact_match", "bleu", "rouge", "perplexity")
        **kwargs: Additional arguments for reward creation
        
    Returns:
        Reward instance
        
    Raises:
        ValueError: If name is not recognized
    """
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward name: {name}. "
            f"Available: {list(REWARD_REGISTRY.keys())}"
        )
    
    return REWARD_REGISTRY[name](**kwargs)


__all__ = [
    "BaseReward",
    "RewardRegistry",
    "PerplexityReward",
    "BLEUReward",
    "ROUGEReward",
    "ExactMatchReward",
    "MathVerifyReward",
    "REWARD_REGISTRY",
    "create_reward",
]

