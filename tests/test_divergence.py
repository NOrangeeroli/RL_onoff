"""Tests for divergence module covering all four purposes:
1. Sampling & Metrics
2. Distribution Extraction
3. Divergence Calculation
4. Conditional Model Switching
"""

import pytest
import numpy as np
from unittest.mock import Mock

from rl_onoff.divergence import DivergenceCalculator
from rl_onoff.backends.base import BaseBackend
from rl_onoff.sampling import Sampler, SamplingConfig
from rl_onoff.metrics import MetricRegistry
from rl_onoff.metrics.builtin import ExactMatchMetric
from rl_onoff.distributions import DistributionExtractor
from rl_onoff.switching import ModelSwitcher


class MockBackend(BaseBackend):
    """Mock backend for testing."""
    
    def __init__(self, model_name: str = "mock_model", **kwargs):
        super().__init__(model_name, **kwargs)
        self.vocab_size = 1000
        self._mock_tokenizer = Mock()
        self._mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4])
        self._mock_tokenizer.decode = Mock(return_value="mock output")
        self._mock_tokenizer.vocab_size = self.vocab_size
    
    def load(self, **kwargs):
        self.model = Mock()
        self.tokenizer = self._mock_tokenizer
        self._is_loaded = True
    
    def generate(self, prompts, max_new_tokens=100, temperature=1.0, 
                 top_k=None, top_p=None, do_sample=True, **kwargs):
        if isinstance(prompts, str):
            return "Generated response."
        return ["Generated response." for _ in prompts]
    
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


@pytest.fixture
def sample_distributions():
    """Fixture for sample probability distributions."""
    # Create two different distributions
    vocab_size = 1000
    seq_len = 5
    
    # Distribution 1: peaked at different positions
    dist1 = np.random.rand(seq_len, vocab_size)
    dist1 = dist1 / dist1.sum(axis=-1, keepdims=True)
    
    # Distribution 2: different peaks
    dist2 = np.random.rand(seq_len, vocab_size)
    dist2 = dist2 / dist2.sum(axis=-1, keepdims=True)
    
    return dist1, dist2


class TestDivergenceSamplingAndMetrics:
    """Tests for Purpose 1: Sampling & Metrics."""
    
    def test_divergence_with_sampling(self, mock_backend, mock_backend_b):
        """Test divergence calculation with sampled text."""
        sampler_a = Sampler(mock_backend)
        sampler_b = Sampler(mock_backend_b)
        
        question = "What is AI?"
        pred_a = sampler_a.sample(question, config=SamplingConfig(max_new_tokens=5))
        pred_b = sampler_b.sample(question, config=SamplingConfig(max_new_tokens=5))
        
        # Extract distributions and compute divergence
        extractor_a = DistributionExtractor(mock_backend)
        extractor_b = DistributionExtractor(mock_backend_b)
        
        dist_a = extractor_a.extract_distributions(question=question, solution=pred_a)
        dist_b = extractor_b.extract_distributions(question=question, solution=pred_b)
        
        calculator = DivergenceCalculator()
        divergences = calculator.compute_token_divergences(dist_a, dist_b, divergence_type="both")
        
        assert "kl" in divergences
        assert "js" in divergences
    
    def test_divergence_with_metrics(self, mock_backend, mock_backend_b):
        """Test divergence calculation alongside metrics."""
        question = "What is 2+2?"
        
        # Generate and compute metrics
        sampler = Sampler(mock_backend)
        prediction = sampler.sample(question, config=SamplingConfig(max_new_tokens=5))
        
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        metrics = registry.compute_all(
            predictions=[prediction],
            references=[["4"]]
        )
        assert "exact_match" in metrics
        
        # Compute divergence
        calculator = DivergenceCalculator()
        divergences = calculator.compute_divergence_for_solutions(
            question=question,
            solution=prediction,
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="both"
        )
        
        assert "kl" in divergences or "js" in divergences


class TestDivergenceDistributionExtraction:
    """Tests for Purpose 2: Distribution Extraction."""
    
    def test_divergence_with_distribution_extraction(self, mock_backend, mock_backend_b):
        """Test divergence calculation using extracted distributions."""
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
        
        calculator = DivergenceCalculator()
        divergences = calculator.compute_token_divergences(dist_a, dist_b, divergence_type="both")
        
        assert "kl" in divergences
        assert "js" in divergences
        assert len(divergences["kl"]) == dist_a.shape[0]
    
    def test_divergence_batch_with_extraction(self, mock_backend, mock_backend_b):
        """Test batch divergence calculation with distribution extraction."""
        questions = ["What is 2+2?", "What is 3+3?"]
        solutions = ["4", "6"]
        
        calculator = DivergenceCalculator()
        divergences_list = calculator.compute_batch_divergences(
            questions=questions,
            solutions=solutions,
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="both"
        )
        
        assert isinstance(divergences_list, list)
        assert len(divergences_list) == 2
        for div in divergences_list:
            assert "kl" in div or "js" in div


class TestDivergenceDivergenceCalculation:
    """Tests for Purpose 3: Divergence Calculation (core purpose)."""
    
    def test_kl_divergence_basic(self, sample_distributions):
        """Test basic KL divergence calculation."""
        dist1, dist2 = sample_distributions
        calculator = DivergenceCalculator()
        
        kl = calculator.kl_divergence(dist1, dist2, axis=-1)
        
        assert isinstance(kl, np.ndarray)
        assert kl.shape == (dist1.shape[0],)
        # KL divergence should be non-negative
        assert np.all(kl >= 0)
    
    def test_kl_divergence_identical_distributions(self):
        """Test KL divergence with identical distributions."""
        dist = np.random.rand(5, 1000)
        dist = dist / dist.sum(axis=-1, keepdims=True)
        
        calculator = DivergenceCalculator()
        kl = calculator.kl_divergence(dist, dist, axis=-1)
        
        # KL divergence of identical distributions should be close to 0
        assert np.all(kl >= 0)
        assert np.allclose(kl, 0, atol=1e-5)
    
    def test_js_divergence_basic(self, sample_distributions):
        """Test basic JS divergence calculation."""
        dist1, dist2 = sample_distributions
        calculator = DivergenceCalculator()
        
        js = calculator.js_divergence(dist1, dist2, axis=-1)
        
        assert isinstance(js, np.ndarray)
        assert js.shape == (dist1.shape[0],)
        # JS divergence should be non-negative and bounded
        assert np.all(js >= 0)
        assert np.all(js <= 1)
    
    def test_js_divergence_identical_distributions(self):
        """Test JS divergence with identical distributions."""
        dist = np.random.rand(5, 1000)
        dist = dist / dist.sum(axis=-1, keepdims=True)
        
        calculator = DivergenceCalculator()
        js = calculator.js_divergence(dist, dist, axis=-1)
        
        # JS divergence of identical distributions should be close to 0
        assert np.all(js >= 0)
        assert np.allclose(js, 0, atol=1e-5)
    
    def test_js_divergence_symmetry(self, sample_distributions):
        """Test JS divergence is symmetric."""
        dist1, dist2 = sample_distributions
        calculator = DivergenceCalculator()
        
        js_12 = calculator.js_divergence(dist1, dist2, axis=-1)
        js_21 = calculator.js_divergence(dist2, dist1, axis=-1)
        
        # JS divergence is symmetric
        assert np.allclose(js_12, js_21, atol=1e-5)
    
    def test_compute_token_divergences_kl(self, sample_distributions):
        """Test compute_token_divergences with KL only."""
        dist1, dist2 = sample_distributions
        calculator = DivergenceCalculator()
        
        results = calculator.compute_token_divergences(dist1, dist2, divergence_type="kl")
        
        assert "kl" in results
        assert "js" not in results
        assert isinstance(results["kl"], np.ndarray)
        assert len(results["kl"]) == dist1.shape[0]
    
    def test_compute_token_divergences_js(self, sample_distributions):
        """Test compute_token_divergences with JS only."""
        dist1, dist2 = sample_distributions
        calculator = DivergenceCalculator()
        
        results = calculator.compute_token_divergences(dist1, dist2, divergence_type="js")
        
        assert "js" in results
        assert "kl" not in results
        assert isinstance(results["js"], np.ndarray)
        assert len(results["js"]) == dist1.shape[0]
    
    def test_compute_token_divergences_both(self, sample_distributions):
        """Test compute_token_divergences with both KL and JS."""
        dist1, dist2 = sample_distributions
        calculator = DivergenceCalculator()
        
        results = calculator.compute_token_divergences(dist1, dist2, divergence_type="both")
        
        assert "kl" in results
        assert "js" in results
        assert len(results["kl"]) == dist1.shape[0]
        assert len(results["js"]) == dist1.shape[0]
    
    def test_compute_token_divergences_shape_mismatch(self):
        """Test compute_token_divergences with mismatched shapes."""
        dist1 = np.random.rand(5, 1000)
        dist2 = np.random.rand(6, 1000)  # Different sequence length
        
        calculator = DivergenceCalculator()
        
        with pytest.raises(ValueError):
            calculator.compute_token_divergences(dist1, dist2)
    
    def test_compute_divergence_for_solutions(self, mock_backend, mock_backend_b):
        """Test compute_divergence_for_solutions method."""
        calculator = DivergenceCalculator()
        divergences = calculator.compute_divergence_for_solutions(
            question="What is 2+2?",
            solution="4",
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="both"
        )
        
        assert "kl" in divergences or "js" in divergences
        assert isinstance(divergences["kl"] if "kl" in divergences else divergences["js"], np.ndarray)
    
    def test_compute_batch_divergences(self, mock_backend, mock_backend_b):
        """Test compute_batch_divergences method."""
        calculator = DivergenceCalculator()
        questions = ["What is 2+2?", "What is 3+3?"]
        solutions = ["4", "6"]
        
        results = calculator.compute_batch_divergences(
            questions=questions,
            solutions=solutions,
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="both"
        )
        
        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert "kl" in result or "js" in result


class TestDivergenceModelSwitching:
    """Tests for Purpose 4: Conditional Model Switching."""
    
    def test_divergence_used_in_switching(self, mock_backend, mock_backend_b):
        """Test that divergence is used in model switching."""
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        # Switcher uses DivergenceCalculator internally
        assert isinstance(switcher.divergence_calculator, DivergenceCalculator)
        
        response = switcher.generate_with_switching(
            question="What is AI?",
            max_new_tokens=10
        )
        
        assert isinstance(response, str)
    
    def test_divergence_values_at_switch_points(self, mock_backend, mock_backend_b):
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
        
        # Check that switch points have divergence values
        for switch in switch_points:
            assert "divergence" in switch
            assert isinstance(switch["divergence"], (float, int))
    
    def test_divergence_kl_vs_js_in_switching(self, mock_backend, mock_backend_b):
        """Test using both KL and JS divergence types in switching."""
        # Test with JS divergence
        switcher_js = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        response_js = switcher_js.generate_with_switching(
            question="What is AI?",
            max_new_tokens=5
        )
        
        # Test with KL divergence
        switcher_kl = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="kl",
            threshold=0.5
        )
        response_kl = switcher_kl.generate_with_switching(
            question="What is AI?",
            max_new_tokens=5
        )
        
        assert isinstance(response_js, str)
        assert isinstance(response_kl, str)


class TestDivergenceIntegration:
    """Integration tests for all four purposes."""
    
    def test_all_purposes_with_divergence(self, mock_backend, mock_backend_b):
        """Test all four purposes using DivergenceCalculator."""
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
        
        # Purpose 3: Divergence Calculation (core)
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
        switched_response, switch_points = switcher.generate_with_switching(
            question=question,
            max_new_tokens=10,
            return_switch_points=True
        )
        assert isinstance(switched_response, str)
        assert isinstance(switch_points, list)
        
        # Compute divergence on switched response
        switched_dists = extractor.extract_distributions(
            question=question,
            solution=switched_response
        )
        divergences_switched = calculator.compute_token_divergences(
            dists, switched_dists, divergence_type="js"
        )
        assert "js" in divergences_switched
