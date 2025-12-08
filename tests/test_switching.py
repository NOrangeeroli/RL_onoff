"""Tests for switching module covering all four purposes:
1. Sampling & Metrics
2. Distribution Extraction
3. Divergence Calculation
4. Conditional Model Switching
"""

import pytest
import numpy as np
from unittest.mock import Mock

from rl_onoff.switching import ModelSwitcher
from rl_onoff.backends.base import BaseBackend
from rl_onoff.sampling import Sampler, SamplingConfig
from rl_onoff.metrics import MetricRegistry
from rl_onoff.metrics.builtin import ExactMatchMetric
from rl_onoff.distributions import DistributionExtractor
from rl_onoff.divergence import DivergenceCalculator


class MockBackend(BaseBackend):
    """Mock backend for testing."""
    
    def __init__(self, model_name: str = "mock_model", **kwargs):
        super().__init__(model_name, **kwargs)
        self.vocab_size = 1000
        self.counter = 0
        self._mock_tokenizer = Mock()
        self._mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4])
        self._mock_tokenizer.decode = Mock(return_value=" token")
        self._mock_tokenizer.vocab_size = self.vocab_size
    
    def load(self, **kwargs):
        self.model = Mock()
        self.tokenizer = self._mock_tokenizer
        self._is_loaded = True
    
    def generate(self, prompts, max_new_tokens=100, temperature=1.0, 
                 top_k=None, top_p=None, do_sample=True, **kwargs):
        # Generate a simple token that appends to the prompt
        if isinstance(prompts, str):
            # Return prompt + new token
            return prompts + " token"
        return [p + " token" for p in prompts]
    
    def get_logits(self, prompts, max_new_tokens=1, **kwargs):
        if isinstance(prompts, str):
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


class TestSwitchingSamplingAndMetrics:
    """Tests for Purpose 1: Sampling & Metrics."""
    
    def test_switcher_generate_with_sampling(self, mock_backend, mock_backend_b):
        """Test model switching generates text that can be used with sampling."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        switched_response = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=5
        )
        
        # Can use sampler on switched response
        sampler = Sampler(mock_backend)
        further_response = sampler.sample(
            switched_response,
            config=SamplingConfig(max_new_tokens=5)
        )
        
        assert isinstance(switched_response, str)
        assert isinstance(further_response, str)
    
    def test_switcher_with_metrics(self, mock_backend, mock_backend_b):
        """Test model switching with metrics calculation."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        switched_response = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=10
        )
        
        # Compute metrics on switched response
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        metrics = registry.compute_all(
            predictions=[switched_response],
            references=[["AI is artificial intelligence."]]
        )
        
        assert "exact_match" in metrics
    
    def test_switcher_compare_with_sampler(self, mock_backend, mock_backend_b):
        """Test comparing switched generation with normal sampling."""
        question = "What is AI?"
        
        # Generate with switching
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        switched_response = switcher.generate_with_switching(
            question=question,
            max_new_tokens=10
        )
        
        # Generate normally
        sampler = Sampler(mock_backend)
        normal_response = sampler.sample(question, config=SamplingConfig(max_new_tokens=10))
        
        # Compute metrics for both
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        
        switched_metrics = registry.compute_all(
            predictions=[switched_response],
            references=[["AI is artificial intelligence."]]
        )
        normal_metrics = registry.compute_all(
            predictions=[normal_response],
            references=[["AI is artificial intelligence."]]
        )
        
        assert "exact_match" in switched_metrics
        assert "exact_match" in normal_metrics


class TestSwitchingDistributionExtraction:
    """Tests for Purpose 2: Distribution Extraction."""
    
    def test_switcher_uses_distributions(self, mock_backend, mock_backend_b):
        """Test that switcher uses distributions internally."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        # Switcher uses get_probabilities which relies on distributions
        response = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=5
        )
        
        assert isinstance(response, str)
    
    def test_switcher_with_distribution_extraction(self, mock_backend, mock_backend_b):
        """Test switcher with distribution extraction."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        switched_response = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=10
        )
        
        # Extract distributions from switched response
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(
            question="What is AI?",
            solution=switched_response
        )
        
        assert isinstance(dists, np.ndarray)
    
    def test_switcher_distributions_at_switch_points(self, mock_backend, mock_backend_b):
        """Test extracting distributions at switch points."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        response, switch_points = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=10,
            return_switch_points=True
        )
        
        # Extract distributions before and after switches
        extractor_a = DistributionExtractor(mock_backend)
        extractor_b = DistributionExtractor(mock_backend_b)
        
        for switch in switch_points:
            # Extract distributions around switch point
            position = switch["position"]
            prefix = response[:position] if position < len(response) else response
            
            dist_a = extractor_a.extract_distributions(
                question="What is AI?",
                solution=prefix
            )
            dist_b = extractor_b.extract_distributions(
                question="What is AI?",
                solution=prefix
            )
            
            assert isinstance(dist_a, np.ndarray)
            assert isinstance(dist_b, np.ndarray)


class TestSwitchingDivergenceCalculation:
    """Tests for Purpose 3: Divergence Calculation."""
    
    def test_switcher_uses_divergence_calculator(self, mock_backend, mock_backend_b):
        """Test that switcher uses DivergenceCalculator."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        # Check that divergence calculator is initialized
        assert isinstance(switcher.divergence_calculator, DivergenceCalculator)
        
        response = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=5
        )
        
        assert isinstance(response, str)
    
    def test_switcher_divergence_at_switch_points(self, mock_backend, mock_backend_b):
        """Test divergence values at switch points."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        response, switch_points = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=10,
            return_switch_points=True
        )
        
        # Check that switch points contain divergence information
        for switch in switch_points:
            assert "divergence" in switch
            assert isinstance(switch["divergence"], (float, int))
            assert "from_model" in switch
            assert "to_model" in switch
            assert "position" in switch
    
    def test_switcher_kl_divergence(self, mock_backend, mock_backend_b):
        """Test switcher with KL divergence."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="kl",
            threshold=0.5
        )
        
        response = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=5
        )
        
        assert isinstance(response, str)
    
    def test_switcher_js_divergence(self, mock_backend, mock_backend_b):
        """Test switcher with JS divergence."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        response = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=5
        )
        
        assert isinstance(response, str)
    
    def test_switcher_threshold_behavior(self, mock_backend, mock_backend_b):
        """Test switcher with different thresholds."""
        # High threshold - should rarely switch
        switcher_high = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=10.0  # Very high
        )
        
        # Low threshold - might switch more
        switcher_low = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.1  # Very low
        )
        
        response_high = switcher_high.generate_with_switching(
            question="What is AI?",
            max_new_tokens=5
        )
        response_low = switcher_low.generate_with_switching(
            question="What is AI?",
            max_new_tokens=5
        )
        
        assert isinstance(response_high, str)
        assert isinstance(response_low, str)
    
    def test_switcher_switch_back_threshold(self, mock_backend, mock_backend_b):
        """Test switcher with custom switch-back threshold."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5,
            switch_back_threshold=0.25
        )
        
        response = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=10
        )
        
        assert isinstance(response, str)
        assert switcher.switch_back_threshold == 0.25


class TestSwitchingModelSwitching:
    """Tests for Purpose 4: Conditional Model Switching (core purpose)."""
    
    def test_switcher_generate_basic(self, mock_backend, mock_backend_b):
        """Test basic generation with switching."""
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
        assert len(response) > 0
    
    def test_switcher_generate_with_switch_points(self, mock_backend, mock_backend_b):
        """Test generation with switch points returned."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        response, switch_points = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=10,
            return_switch_points=True
        )
        
        assert isinstance(response, str)
        assert isinstance(switch_points, list)
    
    def test_switcher_generate_batch(self, mock_backend, mock_backend_b):
        """Test batch generation with switching."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        questions = ["What is AI?", "What is ML?"]
        responses = switcher.generate_with_switching_batch(
            questions=questions,
            max_new_tokens=5
        )
        
        assert isinstance(responses, list)
        assert len(responses) == 2
        assert all(isinstance(r, str) for r in responses)
    
    def test_switcher_generate_batch_with_switch_points(self, mock_backend, mock_backend_b):
        """Test batch generation with switch points."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        questions = ["What is AI?", "What is ML?"]
        results = switcher.generate_with_switching_batch(
            questions=questions,
            max_new_tokens=5,
            return_switch_points=True
        )
        
        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            response, switch_points = result
            assert isinstance(response, str)
            assert isinstance(switch_points, list)
    
    def test_switcher_sampling_parameters(self, mock_backend, mock_backend_b):
        """Test switcher with different sampling parameters."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        response = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=10,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        assert isinstance(response, str)


class TestSwitchingIntegration:
    """Integration tests for all four purposes."""
    
    def test_all_purposes_with_switching(self, mock_backend, mock_backend_b):
        """Test all four purposes using ModelSwitcher."""
        question = "What is 2+2?"
        
        # Purpose 4: Model Switching (core)
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        switched_response, switch_points = switcher.generate_with_switching(
            question=question,
            max_new_tokens=10,
            return_switch_points=True
        )
        assert isinstance(switched_response, str)
        assert isinstance(switch_points, list)
        
        # Purpose 1: Sampling & Metrics
        sampler = Sampler(mock_backend)
        prediction = sampler.sample(question, config=SamplingConfig(max_new_tokens=10))
        
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        metrics = registry.compute_all(
            predictions=[switched_response],
            references=[["4"]]
        )
        assert "exact_match" in metrics
        
        # Purpose 2: Distribution Extraction
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(question=question, solution=switched_response)
        assert isinstance(dists, np.ndarray)
        
        # Purpose 3: Divergence Calculation
        extractor_b = DistributionExtractor(mock_backend_b)
        dists_b = extractor_b.extract_distributions(question=question, solution=switched_response)
        
        calculator = DivergenceCalculator()
        divergences = calculator.compute_token_divergences(dists, dists_b, divergence_type="both")
        assert "kl" in divergences
        assert "js" in divergences
        
        # Verify switch points have divergence information
        for switch in switch_points:
            assert "divergence" in switch
            assert switch["divergence"] >= 0
