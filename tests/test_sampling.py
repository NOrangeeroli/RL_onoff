"""Tests for sampling utilities."""

import pytest
from unittest.mock import MagicMock, patch

from rl_onoff.sampling.sampler import Sampler, SamplingConfig
from rl_onoff.backends.base import BaseBackend


class MockBackend(BaseBackend):
    """Mock backend for testing."""
    
    def __init__(self, model_name="test_model"):
        """Initialize mock backend."""
        super().__init__(model_name)
        self._is_loaded = False
        # Create a MagicMock for generate to track calls
        self.generate = MagicMock(side_effect=self._generate_impl)
    
    def load(self, **kwargs) -> None:
        """Mock load implementation."""
        self.model = "mock_model"
        self.tokenizer = "mock_tokenizer"
        self._is_loaded = True
    
    def _generate_impl(self, prompts, max_new_tokens=100, temperature=1.0, 
                      top_k=None, top_p=None, do_sample=True, **kwargs):
        """Mock generate implementation."""
        if isinstance(prompts, str):
            return "generated_text"
        # Return different text for each prompt to test batching
        return [f"generated_text_{i}" for i in range(len(prompts))]
    
    def get_logits(self, prompts, max_new_tokens=1, **kwargs):
        """Mock get_logits implementation."""
        pass
    
    def get_tokenizer(self):
        """Mock get_tokenizer implementation."""
        return self.tokenizer


class TestSamplingConfig:
    """Test cases for SamplingConfig."""
    
    def test_default_config(self):
        """Test default SamplingConfig values."""
        config = SamplingConfig()
        assert config.max_new_tokens == 100
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.do_sample is True
        assert config.num_samples == 1
        assert config.seed is None
    
    def test_custom_config(self):
        """Test custom SamplingConfig values."""
        config = SamplingConfig(
            max_new_tokens=50,
            temperature=0.7,
            top_k=10,
            top_p=0.9,
            do_sample=False,
            num_samples=3,
            seed=42
        )
        assert config.max_new_tokens == 50
        assert config.temperature == 0.7
        assert config.top_k == 10
        assert config.top_p == 0.9
        assert config.do_sample is False
        assert config.num_samples == 3
        assert config.seed == 42


class TestSampler:
    """Test cases for Sampler."""
    
    def test_init_with_loaded_backend(self):
        """Test sampler initialization with already loaded backend."""
        backend = MockBackend()
        backend.load()
        sampler = Sampler(backend)
        assert sampler.backend == backend
        assert backend.is_loaded()
    
    def test_init_with_unloaded_backend(self):
        """Test sampler initialization with unloaded backend."""
        backend = MockBackend()
        assert not backend.is_loaded()
        sampler = Sampler(backend)
        assert sampler.backend == backend
        assert backend.is_loaded()
    
    def test_sample_single_prompt_default_config(self):
        """Test sampling with single prompt and default config."""
        backend = MockBackend()
        sampler = Sampler(backend)
        
        prompts = ["test prompt"]
        results = sampler.sample(prompts)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0] == "generated_text_0"
        # Verify backend.generate was called with batched prompts
        assert backend.generate.call_count == 1
        call_args = backend.generate.call_args
        assert call_args[0][0] == prompts
    
    def test_sample_multiple_prompts_default_config(self):
        """Test sampling with multiple prompts and default config."""
        backend = MockBackend()
        sampler = Sampler(backend)
        
        prompts = ["prompt1", "prompt2", "prompt3"]
        results = sampler.sample(prompts)
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert results == ["generated_text_0", "generated_text_1", "generated_text_2"]
        # Verify backend.generate was called once with all prompts batched
        assert backend.generate.call_count == 1
        call_args = backend.generate.call_args
        assert call_args[0][0] == prompts
    
    def test_sample_with_custom_config(self):
        """Test sampling with custom SamplingConfig."""
        backend = MockBackend()
        sampler = Sampler(backend)
        
        config = SamplingConfig(
            max_new_tokens=50,
            temperature=0.7,
            top_k=10,
            top_p=0.9,
            do_sample=False
        )
        prompts = ["test prompt"]
        results = sampler.sample(prompts, config=config)
        
        assert len(results) == 1
        # Verify generation kwargs were passed correctly
        assert backend.generate.call_count == 1
        call_kwargs = backend.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 50
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_k"] == 10
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["do_sample"] is False
    
    def test_sample_multiple_samples_per_prompt(self):
        """Test sampling with num_samples > 1."""
        backend = MockBackend()
        sampler = Sampler(backend)
        
        config = SamplingConfig(num_samples=3)
        prompts = ["prompt1", "prompt2"]
        results = sampler.sample(prompts, config=config)
        
        assert isinstance(results, list)
        assert len(results) == 2
        # Each prompt should have 3 samples
        assert len(results[0]) == 3
        assert len(results[1]) == 3
        # Verify backend.generate was called num_samples times
        assert backend.generate.call_count == 3
        # Each call should have all prompts batched
        for call in backend.generate.call_args_list:
            assert call[0][0] == prompts
    
    def test_sample_with_batch_size(self):
        """Test sampling with batch_size parameter."""
        backend = MockBackend()
        sampler = Sampler(backend)
        
        prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]
        results = sampler.sample(prompts, batch_size=2)
        
        assert isinstance(results, list)
        assert len(results) == 4
        # Verify backend.generate was called in batches
        assert backend.generate.call_count == 2
        # First batch should have 2 prompts
        assert len(backend.generate.call_args_list[0][0][0]) == 2
        # Second batch should have 2 prompts
        assert len(backend.generate.call_args_list[1][0][0]) == 2
    
    def test_sample_with_batch_size_and_multiple_samples(self):
        """Test sampling with batch_size and num_samples > 1."""
        backend = MockBackend()
        sampler = Sampler(backend)
        
        config = SamplingConfig(num_samples=2)
        prompts = ["prompt1", "prompt2", "prompt3"]
        results = sampler.sample(prompts, config=config, batch_size=2)
        
        assert isinstance(results, list)
        assert len(results) == 3
        # Each prompt should have 2 samples
        assert all(len(samples) == 2 for samples in results)
        # Verify backend.generate was called num_samples * num_batches times
        # 2 samples * 2 batches = 4 calls
        assert backend.generate.call_count == 4
    
    def test_sample_passes_additional_kwargs(self):
        """Test that additional kwargs are passed to backend.generate."""
        backend = MockBackend()
        sampler = Sampler(backend)
        
        prompts = ["test prompt"]
        results = sampler.sample(prompts, extra_arg="extra_value", another_arg=123)
        
        # Verify additional kwargs were passed
        call_kwargs = backend.generate.call_args[1]
        assert call_kwargs["extra_arg"] == "extra_value"
        assert call_kwargs["another_arg"] == 123
    
    def test_sample_empty_prompts_list(self):
        """Test sampling with empty prompts list."""
        backend = MockBackend()
        sampler = Sampler(backend)
        
        prompts = []
        results = sampler.sample(prompts)
        
        assert isinstance(results, list)
        assert len(results) == 0
        # Should still call generate with empty list
        assert backend.generate.call_count == 1
        call_args = backend.generate.call_args
        assert call_args[0][0] == []

