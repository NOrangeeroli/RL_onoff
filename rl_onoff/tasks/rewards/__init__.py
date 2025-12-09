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


def create_reward(config: dict) -> BaseReward:
    """Create a reward instance from a config dict.
    
    Args:
        config: Dictionary with 'name' key and optional 'kwargs' key
                Example: {"name": "math_verify", "kwargs": {...}}
                Or: {"name": "exact_match"}
        
    Returns:
        Reward instance
        
    Raises:
        ValueError: If name is not recognized or config is invalid
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {type(config)}")
    
    name = config.get("name")
    if name is None:
        raise ValueError("config must have 'name' key")
    
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward name: {name}. "
            f"Available: {list(REWARD_REGISTRY.keys())}"
        )
    
    kwargs = config.get("kwargs", {})
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

