"""Tests for distributions module covering all four purposes:
1. Sampling & Metrics
2. Distribution Extraction
3. Divergence Calculation
4. Conditional Model Switching
"""

import pytest
import numpy as np
from unittest.mock import Mock

from rl_onoff.distributions import DistributionExtractor
from rl_onoff.backends.base import BaseBackend
from rl_onoff.sampling import Sampler, SamplingConfig
from rl_onoff.metrics import MetricRegistry
from rl_onoff.metrics.builtin import ExactMatchMetric
from rl_onoff.divergence import DivergenceCalculator
from rl_onoff.switching import ModelSwitcher


class MockBackend(BaseBackend):
    """Mock backend for testing."""
    
    def __init__(self, model_name: str = "mock_model", **kwargs):
        super().__init__(model_name, **kwargs)
        self.vocab_size = 1000
        self._mock_tokenizer = Mock()
        self._mock_tokenizer.encode = Mock(side_effect=self._encode_side_effect)
        self._mock_tokenizer.decode = Mock(side_effect=self._decode_side_effect)
        self._mock_tokenizer.vocab_size = self.vocab_size
        self._mock_tokenizer.convert_tokens_to_ids = Mock(return_value=1)
        self._mock_tokenizer.convert_ids_to_tokens = Mock(return_value="token")
    
    def _encode_side_effect(self, text, add_special_tokens=False):
        # Return token IDs based on text length
        words = text.split()
        return list(range(1, len(words) + 1))
    
    def _decode_side_effect(self, token_ids, skip_special_tokens=True):
        # Return decoded text
        if isinstance(token_ids, list):
            return " ".join(["token"] * len(token_ids))
        return "token"
    
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
        # Return logits with shape (max_new_tokens, vocab_size)
        if isinstance(prompts, str):
            # Create logits that favor certain tokens
            logits = np.random.randn(max_new_tokens, self.vocab_size)
            # Make first few tokens more likely
            logits[:, :10] += 2.0
            return logits
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


class TestDistributionsSamplingAndMetrics:
    """Tests for Purpose 1: Sampling & Metrics."""
    
    def test_extract_distributions_basic(self, mock_backend):
        """Test basic distribution extraction."""
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(
            question="What is 2+2?",
            solution="4"
        )
        
        assert isinstance(dists, np.ndarray)
        assert len(dists.shape) == 2
        assert dists.shape[1] == mock_backend.vocab_size
    
    def test_extract_distributions_with_sampler(self, mock_backend):
        """Test distribution extraction with sampler."""
        # Generate text using sampler
        sampler = Sampler(mock_backend)
        generated_text = sampler.sample("What is 2+2?", config=SamplingConfig(max_new_tokens=5))
        
        # Extract distributions for the generated text
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(
            question="What is 2+2?",
            solution=generated_text
        )
        
        assert isinstance(dists, np.ndarray)
        assert dists.shape[1] == mock_backend.vocab_size
    
    def test_extract_distributions_with_metrics(self, mock_backend):
        """Test distribution extraction with metrics."""
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(
            question="What is 2+2?",
            solution="4"
        )
        
        # Generate and compute metrics
        sampler = Sampler(mock_backend)
        prediction = sampler.sample("What is 2+2?", config=SamplingConfig(max_new_tokens=5))
        
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        metrics = registry.compute_all(
            predictions=[prediction],
            references=[["4"]]
        )
        
        assert "exact_match" in metrics
        assert isinstance(dists, np.ndarray)
    
    def test_extract_distributions_return_token_ids(self, mock_backend):
        """Test extraction with token IDs returned."""
        extractor = DistributionExtractor(mock_backend)
        dists, token_ids = extractor.extract_distributions(
            question="What is 2+2?",
            solution="4",
            return_token_ids=True
        )
        
        assert isinstance(dists, np.ndarray)
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
    
    def test_extract_distributions_use_logits(self, mock_backend):
        """Test extraction using logits instead of probabilities."""
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(
            question="What is 2+2?",
            solution="4",
            use_logits=True
        )
        
        assert isinstance(dists, np.ndarray)
        # Logits don't sum to 1, they can be any value
        assert dists.shape[1] == mock_backend.vocab_size
    
    def test_extract_distributions_batch(self, mock_backend):
        """Test batch distribution extraction."""
        extractor = DistributionExtractor(mock_backend)
        questions = ["What is 2+2?", "What is 3+3?"]
        solutions = ["4", "6"]
        
        results = extractor.extract_distributions_batch(
            questions=questions,
            solutions=solutions
        )
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)


class TestDistributionsDistributionExtraction:
    """Tests for Purpose 2: Distribution Extraction (core purpose)."""
    
    def test_extract_distributions_shape(self, mock_backend):
        """Test distribution array shape."""
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(
            question="What is 2+2?",
            solution="4"
        )
        
        # Should have shape (solution_length, vocab_size)
        assert len(dists.shape) == 2
        assert dists.shape[1] == mock_backend.vocab_size
        assert dists.shape[0] > 0
    
    def test_extract_distributions_probabilities_sum(self, mock_backend):
        """Test that probability distributions sum to 1."""
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(
            question="What is 2+2?",
            solution="4",
            use_logits=False
        )
        
        # Check that probabilities sum to approximately 1
        sums = np.sum(dists, axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-5)
    
    def test_extract_distributions_temperature(self, mock_backend):
        """Test distribution extraction with different temperatures."""
        extractor = DistributionExtractor(mock_backend)
        
        dists_t1 = extractor.extract_distributions(
            question="What is 2+2?",
            solution="4",
            temperature=1.0
        )
        
        dists_t2 = extractor.extract_distributions(
            question="What is 2+2?",
            solution="4",
            temperature=2.0
        )
        
        assert dists_t1.shape == dists_t2.shape
    
    def test_get_vocab_size(self, mock_backend):
        """Test getting vocabulary size."""
        extractor = DistributionExtractor(mock_backend)
        vocab_size = extractor.get_vocab_size()
        assert vocab_size == mock_backend.vocab_size
    
    def test_get_token_to_id(self, mock_backend):
        """Test getting token ID from token string."""
        extractor = DistributionExtractor(mock_backend)
        token_id = extractor.get_token_to_id("test")
        # Mock returns 1
        assert isinstance(token_id, int) or token_id is None
    
    def test_get_id_to_token(self, mock_backend):
        """Test getting token string from token ID."""
        extractor = DistributionExtractor(mock_backend)
        token = extractor.get_id_to_token(1)
        assert isinstance(token, str) or token is None


class TestDistributionsDivergenceCalculation:
    """Tests for Purpose 3: Divergence Calculation."""
    
    def test_extract_distributions_for_divergence(self, mock_backend, mock_backend_b):
        """Test extracting distributions for divergence calculation."""
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
    
    def test_extract_distributions_batch_for_divergence(self, mock_backend, mock_backend_b):
        """Test batch extraction for divergence calculation."""
        extractor_a = DistributionExtractor(mock_backend)
        extractor_b = DistributionExtractor(mock_backend_b)
        
        questions = ["What is 2+2?", "What is 3+3?"]
        solutions = ["4", "6"]
        
        dists_a = extractor_a.extract_distributions_batch(questions, solutions)
        dists_b = extractor_b.extract_distributions_batch(questions, solutions)
        
        calculator = DivergenceCalculator()
        for dist_a, dist_b in zip(dists_a, dists_b):
            divergences = calculator.compute_token_divergences(dist_a, dist_b, divergence_type="js")
            assert "js" in divergences
    
    def test_extract_distributions_with_calculator(self, mock_backend, mock_backend_b):
        """Test using DistributionExtractor with DivergenceCalculator."""
        calculator = DivergenceCalculator()
        divergences = calculator.compute_divergence_for_solutions(
            question="What is 2+2?",
            solution="4",
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="both"
        )
        
        assert "kl" in divergences or "js" in divergences


class TestDistributionsModelSwitching:
    """Tests for Purpose 4: Conditional Model Switching."""
    
    def test_extract_distributions_for_switching(self, mock_backend, mock_backend_b):
        """Test distribution extraction used in model switching."""
        # Model switching uses get_probabilities internally
        probs_a = mock_backend.get_probabilities("What is AI?", max_new_tokens=1)
        probs_b = mock_backend_b.get_probabilities("What is AI?", max_new_tokens=1)
        
        # Extract distributions can provide similar information
        extractor_a = DistributionExtractor(mock_backend)
        dists_a = extractor_a.extract_distributions(
            question="What is AI?",
            solution="AI is artificial intelligence."
        )
        
        assert isinstance(probs_a, np.ndarray)
        assert isinstance(dists_a, np.ndarray)
    
    def test_extract_distributions_with_switcher(self, mock_backend, mock_backend_b):
        """Test distribution extraction alongside model switching."""
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
        
        # Extract distributions from the switched response
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(
            question="What is AI?",
            solution=switched_response
        )
        
        assert isinstance(dists, np.ndarray)


class TestDistributionsIntegration:
    """Integration tests for all four purposes."""
    
    def test_all_purposes_with_distributions(self, mock_backend, mock_backend_b):
        """Test all four purposes using DistributionExtractor."""
        question = "What is 2+2?"
        solution = "4"
        
        # Purpose 2: Distribution Extraction (core)
        extractor = DistributionExtractor(mock_backend)
        dists = extractor.extract_distributions(question=question, solution=solution)
        assert isinstance(dists, np.ndarray)
        
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
        switched_response, switch_points = switcher.generate_with_switching(
            question=question,
            max_new_tokens=10,
            return_switch_points=True
        )
        assert isinstance(switched_response, str)
        assert isinstance(switch_points, list)
        
        # Extract distributions from switched response
        switched_dists = extractor.extract_distributions(
            question=question,
            solution=switched_response
        )
        assert isinstance(switched_dists, np.ndarray)
