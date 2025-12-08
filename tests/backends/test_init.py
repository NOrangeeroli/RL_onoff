"""Tests for backend __init__.py (get_backend function)."""

import pytest
from unittest.mock import patch, MagicMock

from rl_onoff.backends import get_backend, BaseBackend, HuggingFaceBackend


class TestGetBackend:
    """Test cases for get_backend factory function."""
    
    def test_get_backend_huggingface(self):
        """Test getting HuggingFace backend."""
        backend = get_backend("huggingface", model_name="test_model")
        assert isinstance(backend, HuggingFaceBackend)
        assert backend.model_name == "test_model"
    
    def test_get_backend_huggingface_case_insensitive(self):
        """Test that backend type is case-insensitive."""
        backend1 = get_backend("huggingface", model_name="test_model")
        backend2 = get_backend("HuggingFace", model_name="test_model")
        backend3 = get_backend("HUGGINGFACE", model_name="test_model")
        
        assert isinstance(backend1, HuggingFaceBackend)
        assert isinstance(backend2, HuggingFaceBackend)
        assert isinstance(backend3, HuggingFaceBackend)
    
    def test_get_backend_vllm(self):
        """Test getting vLLM backend."""
        try:
            backend = get_backend("vllm", model_name="test_model")
            # If vLLM is available, check it's the right type
            from rl_onoff.backends.vllm import VLLMBackend
            assert isinstance(backend, VLLMBackend)
        except ImportError:
            # vLLM not available, skip
            pytest.skip("vLLM not available")
    
    def test_get_backend_sglang(self):
        """Test getting SGLang backend."""
        try:
            backend = get_backend("sglang", model_name="test_model")
            # If SGLang is available, check it's the right type
            from rl_onoff.backends.sglang import SGLangBackend
            assert isinstance(backend, SGLangBackend)
        except ImportError:
            # SGLang not available, skip
            pytest.skip("SGLang not available")
    
    def test_get_backend_invalid(self):
        """Test that invalid backend type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend type"):
            get_backend("invalid_backend", model_name="test_model")
    
    def test_get_backend_passes_kwargs(self):
        """Test that kwargs are passed to backend constructor."""
        backend = get_backend(
            "huggingface",
            model_name="test_model",
            device="cpu",
            torch_dtype="float32"
        )
        assert isinstance(backend, HuggingFaceBackend)
        assert backend.model_name == "test_model"
        assert backend.device == "cpu"
    
    def test_backend_exports(self):
        """Test that expected classes are exported from backends module."""
        from rl_onoff.backends import (
            BaseBackend,
            HuggingFaceBackend,
            VLLMBackend,
            SGLangBackend,
            get_backend
        )
        
        assert BaseBackend is not None
        assert HuggingFaceBackend is not None
        assert VLLMBackend is not None
        assert SGLangBackend is not None
        assert callable(get_backend)

