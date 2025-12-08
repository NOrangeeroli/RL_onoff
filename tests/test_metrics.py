"""Tests for metrics module covering all four purposes:
1. Sampling & Metrics
2. Distribution Extraction
3. Divergence Calculation
4. Conditional Model Switching
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from rl_onoff.metrics.base import BaseMetric, MetricRegistry
from rl_onoff.metrics.builtin import (
    PerplexityMetric, BLEUMetric, ROUGEMetric, ExactMatchMetric
)
from rl_onoff.backends.base import BaseBackend
from rl_onoff.sampling import Sampler, SamplingConfig
from rl_onoff.distributions import DistributionExtractor
from rl_onoff.divergence import DivergenceCalculator
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
            return "AI is artificial intelligence."
        return ["AI is artificial intelligence." for _ in prompts]
    
    def get_logits(self, prompts, max_new_tokens=1, **kwargs):
        # Return logits that will produce reasonable probabilities
        if isinstance(prompts, str):
            logits = np.random.randn(max_new_tokens, self.vocab_size)
            return logits
        return [np.random.randn(max_new_tokens, self.vocab_size) for _ in prompts]
    
    def encode(self, text):
        return [1, 2, 3, 4, 5][:len(text.split())]
    
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


class TestMetricsSamplingAndMetrics:
    """Tests for Purpose 1: Sampling & Metrics."""
    
    def test_exact_match_metric_single(self):
        """Test ExactMatchMetric with single prediction."""
        metric = ExactMatchMetric()
        result = metric.compute(
            predictions="AI is artificial intelligence.",
            references="AI is artificial intelligence."
        )
        assert result == 1.0
    
    def test_exact_match_metric_multiple(self):
        """Test ExactMatchMetric with multiple predictions."""
        metric = ExactMatchMetric()
        results = metric.compute(
            predictions=["AI is AI.", "ML is ML."],
            references=["AI is AI.", "ML is machine learning."]
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0] == 1.0
        assert results[1] == 0.0
    
    def test_exact_match_with_normalization(self):
        """Test ExactMatchMetric with text normalization."""
        metric = ExactMatchMetric(normalize=True)
        result = metric.compute(
            predictions="  AI IS ARTIFICIAL INTELLIGENCE.  ",
            references="ai is artificial intelligence."
        )
        assert result == 1.0
    
    @pytest.mark.skipif(True, reason="Requires nltk")
    def test_bleu_metric(self):
        """Test BLEUMetric."""
        try:
            metric = BLEUMetric(n_gram=4)
            result = metric.compute(
                predictions="the cat is on the mat",
                references="the cat sat on the mat"
            )
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
        except ImportError:
            pytest.skip("nltk not available")
    
    @pytest.mark.skipif(True, reason="Requires rouge-score")
    def test_rouge_metric(self):
        """Test ROUGEMetric."""
        try:
            metric = ROUGEMetric()
            result = metric.compute(
                predictions="the cat is on the mat",
                references="the cat sat on the mat"
            )
            assert isinstance(result, dict)
            assert "rouge1" in result
            assert "rouge2" in result
            assert "rougeL" in result
        except ImportError:
            pytest.skip("rouge-score not available")
    
    def test_perplexity_metric(self, mock_backend):
        """Test PerplexityMetric."""
        metric = PerplexityMetric(backend=mock_backend)
        result = metric.compute(predictions="AI is artificial intelligence.")
        assert isinstance(result, float)
        assert result > 0
    
    def test_metric_registry_register(self):
        """Test MetricRegistry registration."""
        registry = MetricRegistry()
        metric = ExactMatchMetric()
        registry.register(metric)
        assert "exact_match" in registry.list_metrics()
    
    def test_metric_registry_compute_all(self):
        """Test MetricRegistry compute_all."""
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        
        results = registry.compute_all(
            predictions=["AI is AI."],
            references=[["AI is AI."]]
        )
        
        assert isinstance(results, dict)
        assert "exact_match" in results
    
    def test_metric_registry_get(self):
        """Test MetricRegistry get method."""
        registry = MetricRegistry()
        metric = ExactMatchMetric()
        registry.register(metric)
        
        retrieved = registry.get("exact_match")
        assert retrieved == metric
    
    def test_metric_registry_compute_subset(self):
        """Test computing subset of metrics."""
        registry = MetricRegistry()
        registry.register(ExactMatchMetric(name="em1"))
        registry.register(ExactMatchMetric(name="em2"))
        
        results = registry.compute_all(
            predictions=["AI is AI."],
            references=[["AI is AI."]],
            metric_names=["em1"]
        )
        
        assert "em1" in results
        assert "em2" not in results
    
    def test_sampler_with_metrics(self, mock_backend):
        """Test Sampler with metrics."""
        sampler = Sampler(mock_backend)
        prediction = sampler.sample("What is AI?", config=SamplingConfig(max_new_tokens=10))
        
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        metrics = registry.compute_all(
            predictions=[prediction],
            references=[["AI is artificial intelligence."]]
        )
        
        assert "exact_match" in metrics


class TestMetricsDistributionExtraction:
    """Tests for Purpose 2: Distribution Extraction."""
    
    def test_metrics_with_distribution_extraction(self, mock_backend):
        """Test metrics with distribution extraction."""
        # Extract distributions
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(
            question="What is 2+2?",
            solution="4"
        )
        assert isinstance(dists, np.ndarray)
        
        # Generate text and compute metrics
        sampler = Sampler(mock_backend)
        prediction = sampler.sample("What is 2+2?", config=SamplingConfig(max_new_tokens=5))
        
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        metrics = registry.compute_all(
            predictions=[prediction],
            references=[["4"]]
        )
        
        assert "exact_match" in metrics
    
    def test_perplexity_with_extracted_distributions(self, mock_backend):
        """Test perplexity metric using extracted distributions."""
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(
            question="What is AI?",
            solution="AI is artificial intelligence."
        )
        
        # Compute perplexity on generated text
        metric = PerplexityMetric(backend=mock_backend)
        perplexity = metric.compute(predictions="AI is artificial intelligence.")
        
        assert isinstance(perplexity, float)
        assert perplexity > 0


class TestMetricsDivergenceCalculation:
    """Tests for Purpose 3: Divergence Calculation."""
    
    def test_metrics_with_divergence_calculation(self, mock_backend, mock_backend_b):
        """Test metrics alongside divergence calculation."""
        question = "What is 2+2?"
        solution = "4"
        
        # Extract distributions from both models
        extractor_a = DistributionExtractor(mock_backend)
        extractor_b = DistributionExtractor(mock_backend_b)
        
        dist_a = extractor_a.extract_distributions(question=question, solution=solution)
        dist_b = extractor_b.extract_distributions(question=question, solution=solution)
        
        # Calculate divergence
        calculator = DivergenceCalculator()
        divergences = calculator.compute_token_divergences(dist_a, dist_b, divergence_type="both")
        assert "kl" in divergences
        assert "js" in divergences
        
        # Generate and compute metrics
        sampler = Sampler(mock_backend)
        prediction = sampler.sample(question, config=SamplingConfig(max_new_tokens=5))
        
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        metrics = registry.compute_all(
            predictions=[prediction],
            references=[[solution]]
        )
        
        assert "exact_match" in metrics
    
    def test_metrics_compare_models_with_divergence(self, mock_backend, mock_backend_b):
        """Test comparing model outputs with both metrics and divergence."""
        question = "What is AI?"
        
        # Generate from both models
        sampler_a = Sampler(mock_backend)
        sampler_b = Sampler(mock_backend_b)
        
        pred_a = sampler_a.sample(question, config=SamplingConfig(max_new_tokens=10))
        pred_b = sampler_b.sample(question, config=SamplingConfig(max_new_tokens=10))
        
        # Compute metrics
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        
        metrics_a = registry.compute_all(
            predictions=[pred_a],
            references=[["AI is artificial intelligence."]]
        )
        metrics_b = registry.compute_all(
            predictions=[pred_b],
            references=[["AI is artificial intelligence."]]
        )
        
        assert "exact_match" in metrics_a
        assert "exact_match" in metrics_b
        
        # Compute divergence
        dist_a = DistributionExtractor(mock_backend).extract_distributions(
            question=question, solution=pred_a
        )
        dist_b = DistributionExtractor(mock_backend_b).extract_distributions(
            question=question, solution=pred_b
        )
        
        calculator = DivergenceCalculator()
        divergences = calculator.compute_token_divergences(dist_a, dist_b, divergence_type="js")
        assert "js" in divergences


class TestMetricsModelSwitching:
    """Tests for Purpose 4: Conditional Model Switching."""
    
    def test_metrics_with_model_switching(self, mock_backend, mock_backend_b):
        """Test metrics with model switching."""
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
    
    def test_metrics_compare_switched_vs_normal(self, mock_backend, mock_backend_b):
        """Test comparing metrics between switched and normal generation."""
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
    
    def test_metrics_at_switch_points(self, mock_backend, mock_backend_b):
        """Test metrics at switch points."""
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
        
        # Compute metrics on full response
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        metrics = registry.compute_all(
            predictions=[response],
            references=[["AI is artificial intelligence."]]
        )
        
        assert "exact_match" in metrics
        assert isinstance(switch_points, list)


class TestMetricsIntegration:
    """Integration tests for all four purposes."""
    
    def test_all_purposes_with_metrics(self, mock_backend, mock_backend_b):
        """Test all four purposes with metrics."""
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
        extractor_b = DistributionExtractor(mock_backend_b)
        dists_b = extractor_b.extract_distributions(question=question, solution=prediction)
        
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
        
        # Compute metrics on switched response
        switched_metrics = registry.compute_all(
            predictions=[switched_response],
            references=[["4"]]
        )
        assert "exact_match" in switched_metrics
        assert isinstance(switch_points, list)
    
    def test_custom_metric_integration(self):
        """Test custom metric with all purposes."""
        class CustomMetric(BaseMetric):
            def compute(self, predictions, references, **kwargs):
                if isinstance(predictions, str):
                    return len(predictions.split())
                return [len(p.split()) for p in predictions]
        
        registry = MetricRegistry()
        registry.register(CustomMetric(name="word_count"))
        
        results = registry.compute_all(
            predictions="AI is artificial intelligence.",
            references="AI is AI."
        )
        
        assert "word_count" in results
        assert results["word_count"] == 4
