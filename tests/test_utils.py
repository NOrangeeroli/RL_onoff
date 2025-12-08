"""Tests for utils module covering all four purposes:
1. Sampling & Metrics
2. Distribution Extraction
3. Divergence Calculation
4. Conditional Model Switching
"""

import pytest
import json
import tempfile
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import Mock

from rl_onoff.utils.config import Config
from rl_onoff.utils.data_loader import DataLoader, load_data
from rl_onoff.backends.base import BaseBackend
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
def sample_json_file():
    """Fixture for sample JSON file."""
    data = [
        {"question": "What is 2+2?", "solution": "4"},
        {"question": "What is 3+3?", "solution": "6"}
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_jsonl_file():
    """Fixture for sample JSONL file."""
    data = [
        {"question": "What is 2+2?", "solution": "4"},
        {"question": "What is 3+3?", "solution": "6"}
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_csv_file():
    """Fixture for sample CSV file."""
    import csv
    data = [
        {"question": "What is 2+2?", "solution": "4"},
        {"question": "What is 3+3?", "solution": "6"}
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=["question", "solution"])
        writer.writeheader()
        writer.writerows(data)
        yield f.name
    os.unlink(f.name)


class TestUtilsSamplingAndMetrics:
    """Tests for Purpose 1: Sampling & Metrics."""
    
    def test_config_for_sampling(self):
        """Test Config class for sampling configuration."""
        @dataclass
        class SamplingConfig(Config):
            max_new_tokens: int = 100
            temperature: float = 1.0
        
        config = SamplingConfig(max_new_tokens=50, temperature=0.8)
        assert config.max_new_tokens == 50
        assert config.temperature == 0.8
        
        # Test serialization
        config_dict = config.to_dict()
        assert config_dict["max_new_tokens"] == 50
        assert config_dict["temperature"] == 0.8
    
    def test_data_loader_with_sampling(self, sample_json_file, mock_backend):
        """Test loading data for sampling."""
        data = load_data(sample_json_file)
        
        assert isinstance(data, list)
        assert len(data) > 0
        assert "question" in data[0]
        
        # Use loaded data for sampling
        sampler = Sampler(mock_backend)
        for item in data[:2]:  # Test with first 2 items
            response = sampler.sample(
                item["question"],
                config=SamplingConfig(max_new_tokens=10)
            )
            assert isinstance(response, str)
    
    def test_data_loader_with_metrics(self, sample_json_file, mock_backend):
        """Test loading data for metrics calculation."""
        data = load_data(sample_json_file)
        
        sampler = Sampler(mock_backend)
        registry = MetricRegistry()
        registry.register(ExactMatchMetric())
        
        for item in data[:2]:
            prediction = sampler.sample(
                item["question"],
                config=SamplingConfig(max_new_tokens=10)
            )
            
            metrics = registry.compute_all(
                predictions=[prediction],
                references=[[item["solution"]]]
            )
            
            assert "exact_match" in metrics


class TestUtilsDistributionExtraction:
    """Tests for Purpose 2: Distribution Extraction."""
    
    def test_data_loader_for_distribution_extraction(self, sample_json_file, mock_backend):
        """Test loading data for distribution extraction."""
        data = load_data(sample_json_file)
        
        extractor = DistributionExtractor(mock_backend)
        
        for item in data[:2]:
            dists = extractor.extract_distributions(
                question=item["question"],
                solution=item["solution"]
            )
            
            assert isinstance(dists, np.ndarray)
            assert len(dists.shape) == 2
    
    def test_config_for_extraction(self):
        """Test Config for distribution extraction settings."""
        @dataclass
        class ExtractionConfig(Config):
            use_logits: bool = False
            temperature: float = 1.0
        
        config = ExtractionConfig(use_logits=True, temperature=2.0)
        assert config.use_logits is True
        assert config.temperature == 2.0


class TestUtilsDivergenceCalculation:
    """Tests for Purpose 3: Divergence Calculation."""
    
    def test_data_loader_for_divergence(self, sample_json_file, mock_backend, mock_backend_b):
        """Test loading data for divergence calculation."""
        data = load_data(sample_json_file)
        
        calculator = DivergenceCalculator()
        
        for item in data[:2]:
            divergences = calculator.compute_divergence_for_solutions(
                question=item["question"],
                solution=item["solution"],
                backend_a=mock_backend,
                backend_b=mock_backend_b,
                divergence_type="both"
            )
            
            assert "kl" in divergences or "js" in divergences
    
    def test_config_for_divergence(self):
        """Test Config for divergence settings."""
        @dataclass
        class DivergenceConfig(Config):
            divergence_type: str = "both"
            epsilon: float = 1e-10
        
        config = DivergenceConfig(divergence_type="js", epsilon=1e-8)
        assert config.divergence_type == "js"
        assert config.epsilon == 1e-8


class TestUtilsModelSwitching:
    """Tests for Purpose 4: Conditional Model Switching."""
    
    def test_data_loader_for_switching(self, sample_json_file, mock_backend, mock_backend_b):
        """Test loading data for model switching."""
        data = load_data(sample_json_file)
        
        switcher = ModelSwitcher(
            backend_a=mock_backend,
            backend_b=mock_backend_b,
            divergence_type="js",
            threshold=0.5
        )
        
        for item in data[:2]:
            response = switcher.generate_with_switching(
                question=item["question"],
                max_new_tokens=10
            )
            
            assert isinstance(response, str)
    
    def test_config_for_switching(self):
        """Test Config for switching settings."""
        @dataclass
        class SwitchingConfig(Config):
            divergence_type: str = "js"
            threshold: float = 0.5
            switch_back_threshold: float = 0.25
        
        config = SwitchingConfig(
            divergence_type="kl",
            threshold=0.7,
            switch_back_threshold=0.3
        )
        assert config.divergence_type == "kl"
        assert config.threshold == 0.7
        assert config.switch_back_threshold == 0.3


class TestUtilsConfig:
    """Tests for Config utility."""
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        @dataclass
        class TestConfig(Config):
            value1: int = 1
            value2: str = "test"
        
        config = TestConfig(value1=42, value2="hello")
        config_dict = config.to_dict()
        
        assert config_dict["value1"] == 42
        assert config_dict["value2"] == "hello"
    
    def test_config_to_json(self):
        """Test saving config to JSON."""
        @dataclass
        class TestConfig(Config):
            value: int = 1
        
        config = TestConfig(value=42)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.to_json(temp_path)
            
            # Read it back
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded["value"] == 42
        finally:
            os.unlink(temp_path)
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        @dataclass
        class TestConfig(Config):
            value: int = 1
        
        data = {"value": 42}
        config = TestConfig.from_dict(data)
        
        assert config.value == 42
    
    def test_config_from_json(self):
        """Test loading config from JSON."""
        @dataclass
        class TestConfig(Config):
            value: int = 1
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"value": 42}, f)
            temp_path = f.name
        
        try:
            config = TestConfig.from_json(temp_path)
            assert config.value == 42
        finally:
            os.unlink(temp_path)


class TestUtilsDataLoader:
    """Tests for DataLoader utility."""
    
    def test_load_json(self, sample_json_file):
        """Test loading JSON file."""
        data = DataLoader.load_json(sample_json_file)
        assert isinstance(data, list)
        assert len(data) == 2
        assert "question" in data[0]
    
    def test_load_jsonl(self, sample_jsonl_file):
        """Test loading JSONL file."""
        data = DataLoader.load_jsonl(sample_jsonl_file)
        assert isinstance(data, list)
        assert len(data) == 2
        assert "question" in data[0]
    
    def test_load_csv(self, sample_csv_file):
        """Test loading CSV file."""
        data = DataLoader.load_csv(sample_csv_file)
        assert isinstance(data, list)
        assert len(data) == 2
        assert "question" in data[0]
        assert "solution" in data[0]
    
    def test_load_csv_with_columns(self, sample_csv_file):
        """Test loading CSV with specified columns."""
        data = DataLoader.load_csv(
            sample_csv_file,
            question_column="question",
            solution_column="solution"
        )
        assert len(data) == 2
        assert "question" in data[0]
        assert "solution" in data[0]
    
    def test_load_data_auto_detect_json(self, sample_json_file):
        """Test auto-detection of JSON format."""
        data = load_data(sample_json_file)
        assert isinstance(data, list)
        assert len(data) == 2
    
    def test_load_data_auto_detect_jsonl(self, sample_jsonl_file):
        """Test auto-detection of JSONL format."""
        data = load_data(sample_jsonl_file)
        assert isinstance(data, list)
        assert len(data) == 2
    
    def test_load_data_auto_detect_csv(self, sample_csv_file):
        """Test auto-detection of CSV format."""
        data = load_data(sample_csv_file)
        assert isinstance(data, list)
        assert len(data) == 2
    
    def test_load_data_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.json")
    
    def test_load_data_unsupported_format(self):
        """Test loading unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_data(temp_path)
        finally:
            os.unlink(temp_path)


class TestUtilsIntegration:
    """Integration tests for all four purposes."""
    
    def test_all_purposes_with_utils(self, sample_json_file, mock_backend, mock_backend_b):
        """Test all four purposes with utils (data loading and config)."""
        # Load data using utils
        data = load_data(sample_json_file)
        assert len(data) > 0
        
        item = data[0]
        question = item["question"]
        solution = item["solution"]
        
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
        switched_response, switch_points = switcher.generate_with_switching(
            question=question,
            max_new_tokens=10,
            return_switch_points=True
        )
        assert isinstance(switched_response, str)
        assert isinstance(switch_points, list)
