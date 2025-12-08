"""Tests for sampling module covering all four purposes:
1. Sampling & Metrics
2. Distribution Extraction
3. Divergence Calculation
4. Conditional Model Switching
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from rl_onoff.sampling import Sampler, SamplingConfig
from rl_onoff.backends.base import BaseBackend
from rl_onoff.metrics import MetricRegistry
from rl_onoff.metrics.builtin import ExactMatchMetric, BLEUMetric
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
            return f"Generated: {prompts[:20]}"
        return [f"Generated: {p[:20]}" for p in prompts]
    
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


class TestSamplingSamplingAndMetrics:
    """Tests for Purpose 1: Sampling & Metrics."""
    
    def test_sampler_single_prompt(self, mock_backend):
        """Test sampling from a single prompt."""
        sampler = Sampler(mock_backend)
        result = sampler.sample("What is AI?", config=SamplingConfig(max_new_tokens=10))
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_sampler_multiple_prompts(self, mock_backend):
        """Test sampling from multiple prompts."""
        sampler = Sampler(mock_backend)
        prompts = ["What is AI?", "What is ML?"]
        results = sampler.sample(prompts, config=SamplingConfig(max_new_tokens=10))
        assert isinstance(results, list)
        assert len(results) == 2
    
    def test_sampler_with_config(self, mock_backend):
        """Test sampler with custom config."""
        sampler = Sampler(mock_backend)
        config = SamplingConfig(
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        result = sampler.sample("Test prompt", config=config)
        assert isinstance(result, str)
    
    def test_sampler_multiple_samples(self, mock_backend):
        """Test sampling multiple times from same prompt."""
        sampler = Sampler(mock_backend)
        config = SamplingConfig(max_new_tokens=10, num_samples=3)
        results = sampler.sample("What is AI?", config=config)
        assert isinstance(results, list)
        assert len(results) == 3
    
    def test_sampler_batch_processing(self, mock_backend):
        """Test batch processing."""
        sampler = Sampler(mock_backend)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = sampler.sample_batch(prompts, batch_size=2, config=SamplingConfig(max_new_tokens=10))
        assert isinstance(results, list)
        assert len(results) == 3
    
    def test_sampler_with_metrics(self, mock_backend):
        """Test sampler with metrics calculation."""
        sampler = Sampler(mock_backend)
        predictions = sampler.sample("What is AI?", config=SamplingConfig(max_new_tokens=10))
        
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        
        results = registry.compute_all(
            predictions=[predictions] if isinstance(predictions, str) else predictions,
            references=[["AI is artificial intelligence."]]
        )
        
        assert len(results) > 0
        assert "exact_match" in results


class TestSamplingDistributionExtraction:
    """Tests for Purpose 2: Distribution Extraction."""
    
    def test_sampler_generates_for_distribution_extraction(self, mock_backend):
        """Test that sampler can generate text for distribution extraction."""
        sampler = Sampler(mock_backend)
        generated_text = sampler.sample("What is 2+2?", config=SamplingConfig(max_new_tokens=5))
        
        # Use generated text as solution for distribution extraction
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(
            question="What is 2+2?",
            solution=generated_text,
            return_token_ids=False
        )
        assert isinstance(dists, np.ndarray)
    
    def test_sampler_batch_for_distribution_extraction(self, mock_backend):
        """Test batch sampling for distribution extraction."""
        sampler = Sampler(mock_backend)
        questions = ["What is 2+2?", "What is 3+3?"]
        generated_texts = sampler.sample(questions, config=SamplingConfig(max_new_tokens=5))
        
        extractor = DistributionExtractor(mock_backend)
        for question, solution in zip(questions, generated_texts):
            dists = extractor.extract_distributions(question=question, solution=solution)
            assert isinstance(dists, np.ndarray)


class TestSamplingDivergenceCalculation:
    """Tests for Purpose 3: Divergence Calculation."""
    
    def test_sampler_generates_for_divergence(self, mock_backend, mock_backend_b):
        """Test sampler generates text used in divergence calculation."""
        sampler_a = Sampler(mock_backend)
        sampler_b = Sampler(mock_backend_b)
        
        question = "What is 2+2?"
        solution_a = sampler_a.sample(question, config=SamplingConfig(max_new_tokens=5))
        solution_b = sampler_b.sample(question, config=SamplingConfig(max_new_tokens=5))
        
        # Extract distributions
        extractor_a = DistributionExtractor(mock_backend)
        extractor_b = DistributionExtractor(mock_backend_b)
        
        dist_a = extractor_a.extract_distributions(question=question, solution=solution_a)
        dist_b = extractor_b.extract_distributions(question=question, solution=solution_b)
        
        # Calculate divergence
        calculator = DivergenceCalculator()
        divergences = calculator.compute_token_divergences(dist_a, dist_b, divergence_type="both")
        
        assert "kl" in divergences
        assert "js" in divergences
    
    def test_sampler_with_calculator_compute_for_solutions(self, mock_backend, mock_backend_b):
        """Test sampler output with divergence calculator."""
        sampler_a = Sampler(mock_backend)
        question = "What is AI?"
        solution = sampler_a.sample(question, config=SamplingConfig(max_new_tokens=5))
        
        calculator = DivergenceCalculator()
        divergences = calculator.compute_divergence_for_solutions(
            question=question,
            solution=solution,
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="both"
        )
        
        assert "kl" in divergences or "js" in divergences


class TestSamplingModelSwitching:
    """Tests for Purpose 4: Conditional Model Switching."""
    
    def test_sampler_generates_for_switching(self, mock_backend, mock_backend_b):
        """Test sampler can generate text that can be used with switching."""
        sampler = Sampler(mock_backend)
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
        
        # Use sampler to generate from same question for comparison
        normal_response = sampler.sample(question, config=SamplingConfig(max_new_tokens=10))
        
        assert isinstance(switched_response, str)
        assert isinstance(normal_response, str)
    
    def test_sampler_with_switching_switch_points(self, mock_backend, mock_backend_b):
        """Test sampler with switching that tracks switch points."""
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
        
        # Sampler can generate from the switched response
        sampler = Sampler(mock_backend)
        further_response = sampler.sample(
            response,
            config=SamplingConfig(max_new_tokens=5)
        )
        assert isinstance(further_response, str)
    
    def test_sampler_batch_with_switching(self, mock_backend, mock_backend_b):
        """Test batch sampling with switching."""
        questions = ["What is AI?", "What is ML?"]
        
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        switched_responses = switcher.generate_with_switching_batch(
            questions=questions,
            max_new_tokens=10
        )
        
        assert isinstance(switched_responses, list)
        assert len(switched_responses) == 2
        
        # Can use sampler on switched responses
        sampler = Sampler(mock_backend)
        for response in switched_responses:
            further_response = sampler.sample(
                response,
                config=SamplingConfig(max_new_tokens=5)
            )
            assert isinstance(further_response, str)


class TestSamplingIntegration:
    """Integration tests for all four purposes."""
    
    def test_all_purposes_with_sampler(self, mock_backend, mock_backend_b):
        """Test all four purposes using Sampler."""
        question = "What is 2+2?"
        
        # Purpose 1: Sampling & Metrics
        sampler = Sampler(mock_backend)
        prediction = sampler.sample(question, config=SamplingConfig(max_new_tokens=10))
        
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        metrics = registry.compute_all(
            predictions=[prediction],
            references=[["4"]]
        )
        assert "exact_match" in metrics
        
        # Purpose 2: Distribution Extraction
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(question=question, solution=prediction)
        assert isinstance(dists, np.ndarray)
        
        # Purpose 3: Divergence Calculation
        prediction_b = sampler.sample(question, config=SamplingConfig(max_new_tokens=10))
        extractor_b = DistributionExtractor(mock_backend_b)
        dists_b = extractor_b.extract_distributions(question=question, solution=prediction_b)
        
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
        
        # Can sample further from switched response
        further_prediction = sampler.sample(
            switched_response,
            config=SamplingConfig(max_new_tokens=5)
        )
        assert isinstance(further_prediction, str)
