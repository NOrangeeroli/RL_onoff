"""Base reward interface for extensible rewards framework."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import numpy as np


class BaseReward(ABC):
    """Abstract base class for all rewards."""

    def __init__(self, name: Optional[str] = None):
        """Initialize reward.
        
        Args:
            name: Reward name (defaults to class name)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]],
        **kwargs
    ) -> Union[float, Dict[str, float]]:
        """Compute reward value(s).
        
        Args:
            predictions: Predicted text(s)
            references: Reference text(s) or list of reference lists
            **kwargs: Additional arguments
            
        Returns:
            Reward value(s) as float or dict of reward values
        """
        pass

    def __call__(self, *args, **kwargs):
        """Allow reward to be called directly."""
        return self.compute(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class RewardRegistry:
    """Registry for managing rewards."""

    def __init__(self):
        """Initialize registry."""
        self._rewards: Dict[str, BaseReward] = {}

    def register(self, reward: BaseReward, name: Optional[str] = None):
        """Register a reward.
        
        Args:
            reward: Reward instance to register
            name: Optional name override
        """
        reward_name = name or reward.name
        self._rewards[reward_name] = reward

    def get(self, name: str) -> BaseReward:
        """Get a reward by name.
        
        Args:
            name: Reward name
            
        Returns:
            Reward instance
            
        Raises:
            KeyError: If reward not found
        """
        if name not in self._rewards:
            raise KeyError(f"Reward '{name}' not found. Available: {list(self._rewards.keys())}")
        return self._rewards[name]

    def compute_all(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]],
        reward_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute all registered rewards or a subset.
        
        Args:
            predictions: Predicted text(s)
            references: Reference text(s)
            reward_names: Optional list of reward names to compute (all if None)
            **kwargs: Additional arguments passed to rewards
            
        Returns:
            Dictionary mapping reward names to their values
        """
        rewards_to_compute = reward_names or list(self._rewards.keys())
        results = {}
        
        for name in rewards_to_compute:
            reward = self.get(name)
            try:
                results[name] = reward.compute(predictions, references, **kwargs)
            except Exception as e:
                results[name] = {"error": str(e)}
        
        return results

    def list_rewards(self) -> List[str]:
        """List all registered reward names.
        
        Returns:
            List of reward names
        """
        return list(self._rewards.keys())

    def clear(self):
        """Clear all registered rewards."""
        self._rewards.clear()

