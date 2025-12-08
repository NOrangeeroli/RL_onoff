"""Tests for VLLMBackend."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from rl_onoff.backends.vllm import VLLMBackend, VLLM_AVAILABLE


class TestVLLMBackend:
    """Test cases for VLLMBackend."""
    
    def test_init_without_vllm(self):
        """Test initialization when vLLM is not available."""
        if not VLLM_AVAILABLE:
            with pytest.raises(ImportError, match="vLLM is not installed"):
                VLLMBackend(model_name="test_model")
    
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
    def test_init(self):
        """Test vLLM backend initialization."""
        backend = VLLMBackend(
            model_name="test_model",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        )
        assert backend.model_name == "test_model"
        assert backend.model is None
        assert backend.tensor_parallel_size == 1
        assert backend.gpu_memory_utilization == 0.9
        assert not backend._is_loaded
    
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
    @patch('rl_onoff.backends.vllm.LLM')
    def test_load(self, mock_llm_class):
        """Test loading the vLLM model."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        
        backend = VLLMBackend(model_name="test_model")
        backend.load()
        
        mock_llm_class.assert_called_once()
        assert backend.model == mock_llm
        assert backend._is_loaded
    
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
    @patch('rl_onoff.backends.vllm.LLM')
    def test_load_idempotent(self, mock_llm_class):
        """Test that load can be called multiple times safely."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        
        backend = VLLMBackend(model_name="test_model")
        backend.load()
        backend.load()  # Call again
        
        # Should only be called once
        assert mock_llm_class.call_count == 1
    
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
    @patch('rl_onoff.backends.vllm.LLM')
    @patch('rl_onoff.backends.vllm.SamplingParams')
    def test_generate_single(self, mock_sampling_params_class, mock_llm_class):
        """Test generating text for a single prompt."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="generated_text")]
        mock_llm.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm
        
        mock_sampling_params = MagicMock()
        mock_sampling_params_class.return_value = mock_sampling_params
        
        backend = VLLMBackend(model_name="test_model")
        result = backend.generate("test prompt", max_new_tokens=10)
        
        assert isinstance(result, str)
        assert result == "generated_text"
        mock_llm.generate.assert_called_once()
    
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
    @patch('rl_onoff.backends.vllm.LLM')
    @patch('rl_onoff.backends.vllm.SamplingParams')
    def test_generate_multiple(self, mock_sampling_params_class, mock_llm_class):
        """Test generating text for multiple prompts."""
        mock_llm = MagicMock()
        mock_output1 = MagicMock()
        mock_output1.outputs = [MagicMock(text="text1")]
        mock_output2 = MagicMock()
        mock_output2.outputs = [MagicMock(text="text2")]
        mock_llm.generate.return_value = [mock_output1, mock_output2]
        mock_llm_class.return_value = mock_llm
        
        mock_sampling_params = MagicMock()
        mock_sampling_params_class.return_value = mock_sampling_params
        
        backend = VLLMBackend(model_name="test_model")
        result = backend.generate(["prompt1", "prompt2"], max_new_tokens=10)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == ["text1", "text2"]
    
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
    @patch('rl_onoff.backends.vllm.LLM')
    @patch('rl_onoff.backends.vllm.SamplingParams')
    def test_generate_with_sampling_params(self, mock_sampling_params_class, mock_llm_class):
        """Test generation with sampling parameters."""
        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="generated_text")]
        mock_llm.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm
        
        mock_sampling_params = MagicMock()
        mock_sampling_params_class.return_value = mock_sampling_params
        
        backend = VLLMBackend(model_name="test_model")
        backend.generate(
            "test",
            max_new_tokens=10,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        # Check that SamplingParams was created with correct parameters
        call_kwargs = mock_sampling_params_class.call_args[1]
        assert call_kwargs["max_tokens"] == 10
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["top_p"] == 0.9
    
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
    @patch('rl_onoff.backends.vllm.LLM')
    def test_get_logits_not_implemented(self, mock_llm_class):
        """Test that get_logits raises NotImplementedError."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        
        backend = VLLMBackend(model_name="test_model")
        backend.load()
        
        with pytest.raises(NotImplementedError, match="vLLM logit extraction"):
            backend.get_logits("test prompt", max_new_tokens=1)
    
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
    @patch('rl_onoff.backends.vllm.LLM')
    def test_get_tokenizer(self, mock_llm_class):
        """Test getting the tokenizer."""
        mock_tokenizer = MagicMock()
        mock_llm_engine = MagicMock()
        mock_llm_engine.tokenizer.tokenizer = mock_tokenizer
        mock_llm = MagicMock()
        mock_llm.llm_engine = mock_llm_engine
        mock_llm_class.return_value = mock_llm
        
        backend = VLLMBackend(model_name="test_model")
        backend.load()
        
        tokenizer = backend.get_tokenizer()
        assert tokenizer == mock_tokenizer
    
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
    @patch('rl_onoff.backends.vllm.LLM')
    def test_get_tokenizer_no_engine(self, mock_llm_class):
        """Test get_tokenizer when llm_engine is not available."""
        mock_llm = MagicMock()
        mock_llm.llm_engine = None
        mock_llm_class.return_value = mock_llm
        
        backend = VLLMBackend(model_name="test_model")
        backend.load()
        
        tokenizer = backend.get_tokenizer()
        assert tokenizer is None

