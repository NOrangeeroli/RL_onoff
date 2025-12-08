"""Tests for SGLangBackend."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from rl_onoff.backends.sglang import SGLangBackend, SGLANG_AVAILABLE


class TestSGLangBackend:
    """Test cases for SGLangBackend."""
    
    def test_init_without_sglang(self):
        """Test initialization when SGLang is not available."""
        if not SGLANG_AVAILABLE:
            with pytest.raises(ImportError, match="SGLang is not installed"):
                SGLangBackend(model_name="test_model")
    
    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
    def test_init(self):
        """Test SGLang backend initialization."""
        backend = SGLangBackend(
            model_name="test_model",
            tp_size=1,
            mem_fraction_static=0.85
        )
        assert backend.model_name == "test_model"
        assert backend.runtime is None
        assert backend.tokenizer is None
        assert backend.tp_size == 1
        assert backend.mem_fraction_static == 0.85
        assert not backend._is_loaded
    
    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
    @patch('rl_onoff.backends.sglang.sgl')
    @patch('rl_onoff.backends.sglang.get_tokenizer')
    def test_load(self, mock_get_tokenizer, mock_sgl):
        """Test loading the SGLang runtime."""
        mock_runtime = MagicMock()
        mock_sgl.Runtime.return_value = mock_runtime
        
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        backend = SGLangBackend(model_name="test_model")
        backend.load()
        
        mock_sgl.Runtime.assert_called_once()
        mock_get_tokenizer.assert_called_once()
        assert backend.runtime == mock_runtime
        assert backend.tokenizer == mock_tokenizer
        assert backend._is_loaded
    
    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
    @patch('rl_onoff.backends.sglang.sgl')
    @patch('rl_onoff.backends.sglang.get_tokenizer')
    def test_load_idempotent(self, mock_get_tokenizer, mock_sgl):
        """Test that load can be called multiple times safely."""
        mock_runtime = MagicMock()
        mock_sgl.Runtime.return_value = mock_runtime
        
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        backend = SGLangBackend(model_name="test_model")
        backend.load()
        backend.load()  # Call again
        
        # Should only be called once
        assert mock_sgl.Runtime.call_count == 1
        assert mock_get_tokenizer.call_count == 1
    
    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
    @patch('rl_onoff.backends.sglang.sgl')
    @patch('rl_onoff.backends.sglang.get_tokenizer')
    def test_generate_single(self, mock_get_tokenizer, mock_sgl):
        """Test generating text for a single prompt."""
        mock_runtime = MagicMock()
        mock_state = MagicMock()
        mock_state.text = "generated_text"
        mock_runtime.run.return_value = mock_state
        mock_sgl.Runtime.return_value = mock_runtime
        
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        backend = SGLangBackend(model_name="test_model")
        result = backend.generate("test prompt", max_new_tokens=10)
        
        assert isinstance(result, str)
        assert result == "generated_text"
        mock_runtime.run.assert_called_once()
    
    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
    @patch('rl_onoff.backends.sglang.sgl')
    @patch('rl_onoff.backends.sglang.get_tokenizer')
    def test_generate_multiple(self, mock_get_tokenizer, mock_sgl):
        """Test generating text for multiple prompts."""
        mock_runtime = MagicMock()
        mock_state1 = MagicMock()
        mock_state1.text = "text1"
        mock_state2 = MagicMock()
        mock_state2.text = "text2"
        mock_runtime.run.side_effect = [mock_state1, mock_state2]
        mock_sgl.Runtime.return_value = mock_runtime
        
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        backend = SGLangBackend(model_name="test_model")
        result = backend.generate(["prompt1", "prompt2"], max_new_tokens=10)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == ["text1", "text2"]
        assert mock_runtime.run.call_count == 2
    
    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
    @patch('rl_onoff.backends.sglang.sgl')
    @patch('rl_onoff.backends.sglang.get_tokenizer')
    def test_generate_with_sampling_params(self, mock_get_tokenizer, mock_sgl):
        """Test generation with sampling parameters."""
        mock_runtime = MagicMock()
        mock_state = MagicMock()
        mock_state.text = "generated_text"
        mock_runtime.run.return_value = mock_state
        mock_sgl.Runtime.return_value = mock_runtime
        
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        backend = SGLangBackend(model_name="test_model")
        backend.generate(
            "test",
            max_new_tokens=10,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        # Check that run was called with correct parameters
        call_kwargs = mock_runtime.run.call_args[1]
        assert call_kwargs["max_new_tokens"] == 10
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["top_p"] == 0.9
    
    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
    @patch('rl_onoff.backends.sglang.sgl')
    @patch('rl_onoff.backends.sglang.get_tokenizer')
    def test_get_logits_not_implemented(self, mock_get_tokenizer, mock_sgl):
        """Test that get_logits raises NotImplementedError."""
        mock_runtime = MagicMock()
        mock_sgl.Runtime.return_value = mock_runtime
        
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        backend = SGLangBackend(model_name="test_model")
        backend.load()
        
        with pytest.raises(NotImplementedError, match="SGLang logit extraction"):
            backend.get_logits("test prompt", max_new_tokens=1)
    
    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
    @patch('rl_onoff.backends.sglang.sgl')
    @patch('rl_onoff.backends.sglang.get_tokenizer')
    def test_get_tokenizer(self, mock_get_tokenizer, mock_sgl):
        """Test getting the tokenizer."""
        mock_runtime = MagicMock()
        mock_sgl.Runtime.return_value = mock_runtime
        
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        backend = SGLangBackend(model_name="test_model")
        backend.load()
        
        tokenizer = backend.get_tokenizer()
        assert tokenizer == mock_tokenizer
        # Should auto-load if not loaded
        assert backend._is_loaded

