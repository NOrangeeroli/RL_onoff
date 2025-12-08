"""Tests for HuggingFaceBackend."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from rl_onoff.backends.huggingface import HuggingFaceBackend


class TestHuggingFaceBackend:
    """Test cases for HuggingFaceBackend."""
    
    def test_init(self):
        """Test HuggingFace backend initialization."""
        backend = HuggingFaceBackend(model_name="test_model")
        assert backend.model_name == "test_model"
        assert backend.model is None
        assert backend.tokenizer is None
        assert not backend._is_loaded
        assert backend.device in ["cpu", "cuda"]
    
    def test_init_with_device(self):
        """Test initialization with specific device."""
        backend = HuggingFaceBackend(model_name="test_model", device="cpu")
        assert backend.device == "cpu"
    
    @patch('rl_onoff.backends.huggingface.AutoTokenizer')
    @patch('rl_onoff.backends.huggingface.AutoModelForCausalLM')
    def test_load(self, mock_model_class, mock_tokenizer_class):
        """Test loading the model and tokenizer."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "eos"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        backend = HuggingFaceBackend(model_name="test_model", device="cpu")
        backend.load()
        
        # Verify tokenizer was loaded
        mock_tokenizer_class.from_pretrained.assert_called_once()
        assert backend.tokenizer == mock_tokenizer
        assert mock_tokenizer.pad_token == "eos"  # Should be set to eos_token
        
        # Verify model was loaded
        mock_model_class.from_pretrained.assert_called_once()
        assert backend.model == mock_model
        assert backend._is_loaded
    
    @patch('rl_onoff.backends.huggingface.AutoTokenizer')
    @patch('rl_onoff.backends.huggingface.AutoModelForCausalLM')
    def test_load_idempotent(self, mock_model_class, mock_tokenizer_class):
        """Test that load can be called multiple times safely."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        backend = HuggingFaceBackend(model_name="test_model", device="cpu")
        backend.load()
        backend.load()  # Call again
        
        # Should only be called once
        assert mock_tokenizer_class.from_pretrained.call_count == 1
        assert mock_model_class.from_pretrained.call_count == 1
    
    @patch('rl_onoff.backends.huggingface.AutoTokenizer')
    @patch('rl_onoff.backends.huggingface.AutoModelForCausalLM')
    def test_generate_single(self, mock_model_class, mock_tokenizer_class):
        """Test generating text for a single prompt."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.batch_decode.return_value = ["generated_text"]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_class.from_pretrained.return_value = mock_model
        
        backend = HuggingFaceBackend(model_name="test_model", device="cpu")
        result = backend.generate("test prompt", max_new_tokens=10)
        
        assert isinstance(result, str)
        assert result == "generated_text"
        mock_model.generate.assert_called_once()
    
    @patch('rl_onoff.backends.huggingface.AutoTokenizer')
    @patch('rl_onoff.backends.huggingface.AutoModelForCausalLM')
    def test_generate_multiple(self, mock_model_class, mock_tokenizer_class):
        """Test generating text for multiple prompts."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        mock_tokenizer.batch_decode.return_value = ["text1", "text2"]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        mock_model_class.from_pretrained.return_value = mock_model
        
        backend = HuggingFaceBackend(model_name="test_model", device="cpu")
        result = backend.generate(["prompt1", "prompt2"], max_new_tokens=10)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == ["text1", "text2"]
    
    @patch('rl_onoff.backends.huggingface.AutoTokenizer')
    @patch('rl_onoff.backends.huggingface.AutoModelForCausalLM')
    def test_generate_with_sampling_params(self, mock_model_class, mock_tokenizer_class):
        """Test generation with sampling parameters."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.batch_decode.return_value = ["generated_text"]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_class.from_pretrained.return_value = mock_model
        
        backend = HuggingFaceBackend(model_name="test_model", device="cpu")
        backend.generate(
            "test",
            max_new_tokens=10,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        # Check that generate was called with correct parameters
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 10
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["do_sample"] is True
    
    @patch('rl_onoff.backends.huggingface.AutoTokenizer')
    @patch('rl_onoff.backends.huggingface.AutoModelForCausalLM')
    def test_get_logits_single(self, mock_model_class, mock_tokenizer_class):
        """Test getting logits for a single prompt."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model outputs
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(1, 3, 1000)  # (batch, seq, vocab)
        
        mock_model = MagicMock()
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        backend = HuggingFaceBackend(model_name="test_model", device="cpu")
        logits = backend.get_logits("test prompt", max_new_tokens=2)
        
        assert isinstance(logits, np.ndarray)
        assert logits.shape == (2, 1000)  # (max_new_tokens, vocab_size)
    
    @patch('rl_onoff.backends.huggingface.AutoTokenizer')
    @patch('rl_onoff.backends.huggingface.AutoModelForCausalLM')
    def test_get_logits_multiple(self, mock_model_class, mock_tokenizer_class):
        """Test getting logits for multiple prompts."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.randn(2, 3, 1000)
        
        mock_model = MagicMock()
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        backend = HuggingFaceBackend(model_name="test_model", device="cpu")
        logits = backend.get_logits(["prompt1", "prompt2"], max_new_tokens=2)
        
        assert isinstance(logits, list)
        assert len(logits) == 2
        for logit_array in logits:
            assert isinstance(logit_array, np.ndarray)
            assert logit_array.shape == (2, 1000)
    
    @patch('rl_onoff.backends.huggingface.AutoTokenizer')
    @patch('rl_onoff.backends.huggingface.AutoModelForCausalLM')
    def test_get_tokenizer(self, mock_model_class, mock_tokenizer_class):
        """Test getting the tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        backend = HuggingFaceBackend(model_name="test_model", device="cpu")
        tokenizer = backend.get_tokenizer()
        
        assert tokenizer == mock_tokenizer
        # Should auto-load if not loaded
        assert backend._is_loaded
    
    @patch('rl_onoff.backends.huggingface.AutoTokenizer')
    @patch('rl_onoff.backends.huggingface.AutoModelForCausalLM')
    def test_encode_decode(self, mock_model_class, mock_tokenizer_class):
        """Test encode and decode methods."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "decoded_text"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        backend = HuggingFaceBackend(model_name="test_model", device="cpu")
        backend.load()
        
        # Test encode
        result = backend.encode("test")
        mock_tokenizer.encode.assert_called_with("test", add_special_tokens=False)
        
        # Test decode
        result = backend.decode([1, 2, 3])
        mock_tokenizer.decode.assert_called_with([1, 2, 3], skip_special_tokens=True)

