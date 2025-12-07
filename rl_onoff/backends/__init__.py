"""Backend implementations for different LLM inference engines."""

from rl_onoff.backends.base import BaseBackend
from rl_onoff.backends.huggingface import HuggingFaceBackend
from rl_onoff.backends.vllm import VLLMBackend
from rl_onoff.backends.sglang import SGLangBackend

__all__ = [
    "BaseBackend",
    "HuggingFaceBackend",
    "VLLMBackend",
    "SGLangBackend",
    "get_backend",
]


def get_backend(backend_type: str, **kwargs):
    """Get a backend instance by type."""
    backend_map = {
        "huggingface": HuggingFaceBackend,
        "vllm": VLLMBackend,
        "sglang": SGLangBackend,
    }
    
    if backend_type.lower() not in backend_map:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Supported: {list(backend_map.keys())}"
        )
    
    return backend_map[backend_type.lower()](**kwargs)

