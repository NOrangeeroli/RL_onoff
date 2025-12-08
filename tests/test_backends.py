"""Tests for backends module covering all four purposes:
1. Sampling & Metrics
2. Distribution Extraction
3. Divergence Calculation
4. Conditional Model Switching
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List

from rl_onoff.backends.base import BaseBackend
from rl_onoff.backends.huggingface import HuggingFaceBackend
from rl_onoff.backends import get_backend
from rl_onoff.sampling import Sampler, SamplingConfig
from rl_onoff.metrics import MetricRegistry
from rl_onoff.metrics.builtin import ExactMatchMetric
from rl_onoff.distributions import DistributionExtractor
from rl_onoff.divergence import DivergenceCalculator
from rl_onoff.switching import ModelSwitcher


class MockBackend(BaseBackend):
    """Mock backend for testing."""
    
    def __init__(self, model_name: str = "mock_model", **kwargs):
        super().__init__(model_name, **kwargs)
        self.vocab_size = 1000
        self._mock_tokenizer = Mock()
        self._mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        self._mock_tokenizer.decode = Mock(return_value="mock output")
        self._mock_tokenizer.vocab_size = self.vocab_size
    
    def load(self, **kwargs):
        self.model = Mock()
        self.tokenizer = self._mock_tokenizer
        self._is_loaded = True
    
    def generate(self, prompts, max_new_tokens=100, temperature=1.0, 
                 top_k=None, top_p=None, do_sample=True, **kwargs):
        if isinstance(prompts, str):
            return f"Generated response for: {prompts[:20]}..."
        return [f"Generated response for: {p[:20]}..." for p in prompts]
    
    def get_logits(self, prompts, max_new_tokens=1, **kwargs):
        if isinstance(prompts, str):
            # Return random logits with shape (max_new_tokens, vocab_size)
            return np.random.randn(max_new_tokens, self.vocab_size)
        return [np.random.randn(max_new_tokens, self.vocab_size) for _ in prompts]
    
    def get_tokenizer(self):
        if not self._is_loaded:
            self.load()
        return self.tokenizer


@pytest.fixture
def mock_backend():
    """Fixture for mock backend."""
    backend = MockBackend()
    backend.load()
    return backend


@pytest.fixture
def mock_backend_b():
    """Fixture for second mock backend."""
    backend = MockBackend(model_name="mock_model_b")
    backend.load()
    return backend


class TestBackendsSamplingAndMetrics:
    """Tests for Purpose 1: Sampling & Metrics."""
    
    def test_backend_generate_single(self, mock_backend):
        """Test generating text from a single prompt."""
        result = mock_backend.generate("What is AI?", max_new_tokens=10)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_backend_generate_batch(self, mock_backend):
        """Test generating text from multiple prompts."""
        prompts = ["What is AI?", "What is ML?"]
        results = mock_backend.generate(prompts, max_new_tokens=10)
        assert isinstance(results, list)
        assert len(results) == 2
    
    def test_backend_with_sampler(self, mock_backend):
        """Test backend integration with Sampler."""
        sampler = Sampler(mock_backend)
        config = SamplingConfig(max_new_tokens=10, temperature=0.7)
        result = sampler.sample("What is AI?", config=config)
        assert isinstance(result, str)
    
    def test_backend_with_metrics(self, mock_backend):
        """Test backend with metrics calculation."""
        # Generate text
        prediction = mock_backend.generate("What is AI?", max_new_tokens=10)
        
        # Calculate metrics
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        
        results = registry.compute_all(
            predictions=[prediction],
            references=[["AI is artificial intelligence."]]
        )
        
        assert "exact_match" in results
        assert isinstance(results["exact_match"], (float, list))


class TestBackendsDistributionExtraction:
    """Tests for Purpose 2: Distribution Extraction."""
    
    def test_backend_get_logits(self, mock_backend):
        """Test getting logits from backend."""
        logits = mock_backend.get_logits("What is 2+2?", max_new_tokens=5)
        assert isinstance(logits, np.ndarray)
        assert len(logits.shape) == 2
        assert logits.shape[1] == mock_backend.vocab_size
    
    def test_backend_get_probabilities(self, mock_backend):
        """Test getting probabilities from backend."""
        probs = mock_backend.get_probabilities("What is 2+2?", max_new_tokens=5)
        assert isinstance(probs, np.ndarray)
        assert len(probs.shape) == 2
        # Check probabilities sum to approximately 1
        assert np.allclose(np.sum(probs, axis=-1), 1.0, atol=1e-5)
    
    def test_backend_with_distribution_extractor(self, mock_backend):
        """Test backend with DistributionExtractor."""
        extractor = DistributionExtractor(mock_backend)
        dists, token_ids = extractor.extract_distributions(
            question="What is 2+2?",
            solution="4",
            return_token_ids=True
        )
        assert isinstance(dists, np.ndarray)
        assert len(dists.shape) == 2
        assert isinstance(token_ids, list)


class TestBackendsDivergenceCalculation:
    """Tests for Purpose 3: Divergence Calculation."""
    
    def test_backend_get_distributions_for_divergence(self, mock_backend, mock_backend_b):
        """Test getting distributions from two backends for divergence."""
        extractor_a = DistributionExtractor(mock_backend)
        extractor_b = DistributionExtractor(mock_backend_b)
        
        dist_a = extractor_a.extract_distributions(
            question="What is 2+2?",
            solution="4"
        )
        dist_b = extractor_b.extract_distributions(
            question="What is 2+2?",
            solution="4"
        )
        
        assert dist_a.shape == dist_b.shape
        
        # Compute divergence
        calculator = DivergenceCalculator()
        divergences = calculator.compute_token_divergences(dist_a, dist_b, divergence_type="both")
        
        assert "kl" in divergences
        assert "js" in divergences
        assert len(divergences["kl"]) == dist_a.shape[0]
    
    def test_backend_encode_decode(self, mock_backend):
        """Test encoding and decoding functionality."""
        text = "Hello world"
        token_ids = mock_backend.encode(text)
        decoded = mock_backend.decode(token_ids)
        assert isinstance(token_ids, list)
        assert isinstance(decoded, str)


class TestBackendsModelSwitching:
    """Tests for Purpose 4: Conditional Model Switching."""
    
    def test_backend_with_model_switcher(self, mock_backend, mock_backend_b):
        """Test backend integration with ModelSwitcher."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        response = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=10
        )
        
        assert isinstance(response, str)
    
    def test_backend_get_probabilities_for_switching(self, mock_backend, mock_backend_b):
        """Test getting probabilities for model switching."""
        probs_a = mock_backend.get_probabilities("What is AI?", max_new_tokens=1)
        probs_b = mock_backend_b.get_probabilities("What is AI?", max_new_tokens=1)
        
        assert isinstance(probs_a, np.ndarray)
        assert isinstance(probs_b, np.ndarray)
        assert probs_a.shape == probs_b.shape


class TestBackendsBase:
    """Tests for base backend functionality."""
    
    def test_base_backend_is_loaded(self):
        """Test is_loaded method."""
        backend = MockBackend()
        assert not backend.is_loaded()
        backend.load()
        assert backend.is_loaded()
    
    def test_base_backend_repr(self):
        """Test string representation."""
        backend = MockBackend(model_name="test_model")
        repr_str = repr(backend)
        assert "MockBackend" in repr_str
        assert "test_model" in repr_str
    
    def test_get_backend_function(self):
        """Test get_backend factory function."""
        with patch('rl_onoff.backends.huggingface.HuggingFaceBackend') as mock_hf:
            backend = get_backend("huggingface", model_name="test")
            mock_hf.assert_called_once()
        
        with pytest.raises(ValueError):
            get_backend("unknown_backend")


class TestBackendsIntegration:
    """Integration tests for all four purposes."""
    
    def test_all_purposes_integration(self, mock_backend, mock_backend_b):
        """Test all four purposes together."""
        question = "What is 2+2?"
        solution = "4"
        
        # Purpose 1: Sampling & Metrics
        sampler = Sampler(mock_backend)
        prediction = sampler.sample(question, config=SamplingConfig(max_new_tokens=10))
        
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        metrics = registry.compute_all(
            predictions=[prediction],
            references=[[solution]]
        )
        assert "exact_match" in metrics
        
        # Purpose 2: Distribution Extraction
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(question=question, solution=solution)
        assert isinstance(dists, np.ndarray)
        
        # Purpose 3: Divergence Calculation
        extractor_b = DistributionExtractor(mock_backend_b)
        dists_b = extractor_b.extract_distributions(question=question, solution=solution)
        
        calculator = DivergenceCalculator()
        divergences = calculator.compute_token_divergences(dists, dists_b, divergence_type="both")
        assert "kl" in divergences
        assert "js" in divergences
        
        # Purpose 4: Model Switching
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        response, switch_points = switcher.generate_with_switching(
            question=question,
            max_new_tokens=10,
            return_switch_points=True
        )
        assert isinstance(response, str)
        assert isinstance(switch_points, list)
