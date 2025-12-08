"""Tests for BaseBackend abstract class."""

import pytest
import numpy as np
from abc import ABC

from rl_onoff.backends.base import BaseBackend


class ConcreteBackend(BaseBackend):
    """Concrete implementation for testing BaseBackend."""
    
    def load(self, **kwargs) -> None:
        """Mock load implementation."""
        self.model = "mock_model"
        self.tokenizer = "mock_tokenizer"
        self._is_loaded = True
    
    def generate(self, prompts, max_new_tokens=100, temperature=1.0, 
                 top_k=None, top_p=None, do_sample=True, **kwargs):
        """Mock generate implementation."""
        if isinstance(prompts, str):
            return "generated_text"
        return ["generated_text"] * len(prompts)
    
    def get_logits(self, prompts, max_new_tokens=1, **kwargs):
        """Mock get_logits implementation."""
        vocab_size = 1000
        if isinstance(prompts, str):
            return np.random.randn(max_new_tokens, vocab_size)
        return [np.random.randn(max_new_tokens, vocab_size) for _ in prompts]
    
    def get_tokenizer(self):
        """Mock get_tokenizer implementation."""
        return self.tokenizer


class TestBaseBackend:
    """Test cases for BaseBackend."""
    
    def test_init(self):
        """Test backend initialization."""
        backend = ConcreteBackend(model_name="test_model")
        assert backend.model_name == "test_model"
        assert backend.model is None
        assert backend.tokenizer is None
        assert not backend._is_loaded
    
    def test_is_loaded(self):
        """Test is_loaded method."""
        backend = ConcreteBackend(model_name="test_model")
        assert not backend.is_loaded()
        
        backend.load()
        assert backend.is_loaded()
    
    def test_encode_decode(self):
        """Test encode and decode methods."""
        backend = ConcreteBackend(model_name="test_model")
        
        # Mock tokenizer with encode/decode methods
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3] if isinstance(text, str) else [[1, 2], [3, 4]]
            
            def decode(self, token_ids, skip_special_tokens=True):
                if isinstance(token_ids[0], int):
                    return "decoded_text"
                return ["decoded_text"] * len(token_ids)
        
        backend.tokenizer = MockTokenizer()
        backend._is_loaded = True
        
        # Test encode
        result = backend.encode("test")
        assert result == [1, 2, 3]
        
        result = backend.encode(["test1", "test2"])
        assert result == [[1, 2], [3, 4]]
        
        # Test decode
        result = backend.decode([1, 2, 3])
        assert result == "decoded_text"
        
        result = backend.decode([[1, 2], [3, 4]])
        assert result == ["decoded_text", "decoded_text"]
    
    def test_get_probabilities(self):
        """Test get_probabilities method."""
        backend = ConcreteBackend(model_name="test_model")
        backend._is_loaded = True
        
        # Mock get_logits to return known logits
        def mock_get_logits(prompts, max_new_tokens=1, **kwargs):
            # Return logits that will produce known probabilities
            logits = np.array([[1.0, 2.0, 3.0]])
            if isinstance(prompts, str):
                return logits
            return [logits] * len(prompts)
        
        backend.get_logits = mock_get_logits
        
        # Test single prompt
        probs = backend.get_probabilities("test", max_new_tokens=1, temperature=1.0)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (1, 3)  # (seq_len, vocab_size)
        # Probabilities should sum to 1
        assert np.allclose(probs.sum(axis=-1), 1.0)
        
        # Test multiple prompts
        probs = backend.get_probabilities(["test1", "test2"], max_new_tokens=1, temperature=1.0)
        assert isinstance(probs, list)
        assert len(probs) == 2
        for prob in probs:
            assert np.allclose(prob.sum(axis=-1), 1.0)
    
    def test_get_probabilities_with_temperature(self):
        """Test get_probabilities with different temperatures."""
        backend = ConcreteBackend(model_name="test_model")
        backend._is_loaded = True
        
        # Mock get_logits
        def mock_get_logits(prompts, max_new_tokens=1, **kwargs):
            logits = np.array([[1.0, 2.0, 3.0]])
            if isinstance(prompts, str):
                return logits
            return [logits] * len(prompts)
        
        backend.get_logits = mock_get_logits
        
        # Test with different temperatures
        probs_high = backend.get_probabilities("test", temperature=2.0)
        probs_low = backend.get_probabilities("test", temperature=0.5)
        
        # Higher temperature should make distribution more uniform
        # Lower temperature should make distribution more peaked
        assert probs_high.std() < probs_low.std() or np.allclose(probs_high, probs_low, atol=0.1)
    
    def test_repr(self):
        """Test string representation."""
        backend = ConcreteBackend(model_name="test_model")
        repr_str = repr(backend)
        assert "ConcreteBackend" in repr_str
        assert "test_model" in repr_str
    
    def test_abstract_methods(self):
        """Test that BaseBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseBackend(model_name="test")

