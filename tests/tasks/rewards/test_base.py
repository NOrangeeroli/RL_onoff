"""Tests for BaseReward and RewardRegistry."""

import pytest
from typing import Union, List, Dict
from unittest.mock import MagicMock
from rl_onoff.tasks.rewards.base import BaseReward, RewardRegistry


class ConcreteReward(BaseReward):
    """Concrete implementation for testing."""
    
    def compute(self, predictions, references, **kwargs):
        if isinstance(predictions, str):
            return 1.0
        return [1.0] * len(predictions)


class TestBaseReward:
    """Tests for BaseReward abstract class."""
    
    def test_init_default_name(self):
        """Test initialization with default name."""
        reward = ConcreteReward()
        assert reward.name == "ConcreteReward"
    
    def test_init_custom_name(self):
        """Test initialization with custom name."""
        reward = ConcreteReward(name="custom")
        assert reward.name == "custom"
    
    def test_compute_single(self):
        """Test compute with single prediction."""
        reward = ConcreteReward()
        result = reward.compute("prediction", "reference")
        assert result == 1.0
    
    def test_compute_multiple(self):
        """Test compute with multiple predictions."""
        reward = ConcreteReward()
        result = reward.compute(["pred1", "pred2"], ["ref1", "ref2"])
        assert result == [1.0, 1.0]
    
    def test_callable(self):
        """Test that reward is callable."""
        reward = ConcreteReward()
        result = reward("prediction", "reference")
        assert result == 1.0
    
    def test_repr(self):
        """Test __repr__ method."""
        reward = ConcreteReward()
        repr_str = repr(reward)
        assert "ConcreteReward" in repr_str
        assert "name=" in repr_str


class TestRewardRegistry:
    """Tests for RewardRegistry."""
    
    def test_init(self):
        """Test initialization."""
        registry = RewardRegistry()
        assert len(registry.list_rewards()) == 0
    
    def test_register_with_default_name(self):
        """Test registering reward with default name."""
        registry = RewardRegistry()
        reward = ConcreteReward()
        registry.register(reward)
        assert "ConcreteReward" in registry.list_rewards()
    
    def test_register_with_custom_name(self):
        """Test registering reward with custom name."""
        registry = RewardRegistry()
        reward = ConcreteReward()
        registry.register(reward, name="custom_reward")
        assert "custom_reward" in registry.list_rewards()
        assert "ConcreteReward" not in registry.list_rewards()
    
    def test_get_existing_reward(self):
        """Test getting existing reward."""
        registry = RewardRegistry()
        reward = ConcreteReward()
        registry.register(reward, name="test")
        retrieved = registry.get("test")
        assert retrieved is reward
    
    def test_get_nonexistent_reward(self):
        """Test getting nonexistent reward raises KeyError."""
        registry = RewardRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")
    
    def test_list_rewards(self):
        """Test listing all registered rewards."""
        registry = RewardRegistry()
        reward1 = ConcreteReward(name="reward1")
        reward2 = ConcreteReward(name="reward2")
        registry.register(reward1)
        registry.register(reward2)
        
        rewards = registry.list_rewards()
        assert len(rewards) == 2
        assert "reward1" in rewards
        assert "reward2" in rewards
    
    def test_compute_all(self):
        """Test computing all registered rewards."""
        registry = RewardRegistry()
        reward1 = ConcreteReward(name="reward1")
        reward2 = ConcreteReward(name="reward2")
        registry.register(reward1)
        registry.register(reward2)
        
        results = registry.compute_all("pred", "ref")
        assert "reward1" in results
        assert "reward2" in results
        assert results["reward1"] == 1.0
        assert results["reward2"] == 1.0
    
    def test_compute_all_subset(self):
        """Test computing subset of registered rewards."""
        registry = RewardRegistry()
        reward1 = ConcreteReward(name="reward1")
        reward2 = ConcreteReward(name="reward2")
        registry.register(reward1)
        registry.register(reward2)
        
        results = registry.compute_all("pred", "ref", reward_names=["reward1"])
        assert "reward1" in results
        assert "reward2" not in results
    
    def test_compute_all_with_error(self):
        """Test compute_all handles errors gracefully."""
        registry = RewardRegistry()
        
        class FailingReward(BaseReward):
            def compute(self, predictions, references, **kwargs):
                raise ValueError("Test error")
        
        reward = FailingReward()
        registry.register(reward, name="failing")
        
        results = registry.compute_all("pred", "ref")
        assert "failing" in results
        assert "error" in results["failing"]
        assert "Test error" in results["failing"]["error"]
    
    def test_clear(self):
        """Test clearing all registered rewards."""
        registry = RewardRegistry()
        reward = ConcreteReward()
        registry.register(reward)
        assert len(registry.list_rewards()) == 1
        
        registry.clear()
        assert len(registry.list_rewards()) == 0

