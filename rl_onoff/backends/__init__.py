"""Backend implementations for different LLM inference engines."""

from rl_onoff.backends.base import BaseBackend
from rl_onoff.backends.huggingface import HuggingFaceBackend
from rl_onoff.backends.vllm import VLLMBackend
from rl_onoff.backends.sglang import SGLangBackend
from rl_onoff.backends.config import BackendConfig

__all__ = [
    "BaseBackend",
    "HuggingFaceBackend",
    "VLLMBackend",
    "SGLangBackend",
    "BackendConfig",
    "create_backend",
]


def create_backend(config: 'BackendConfig'):
    """Get a backend instance from a BackendConfig.
    
    Args:
        config: BackendConfig instance
    
    Returns:
        Backend instance
    """
    backend_type = config.backend_type
    
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
    
    return backend_map[backend_type.lower()](config)


if __name__ == "__main__":
    # Simple usage example for create_backend
    print("create_backend Usage Example")
    print("=" * 50)
    
    # Example 1: Create backend from config file
    print("\nExample 1: Create backend from config file")
    try:
        from rl_onoff.backends.config import BackendConfig
        config = BackendConfig.from_file("rl_onoff/backends/configs/huggingface_default.yaml")
        backend = create_backend(config)
        print(f"Created backend from config: {backend.__class__.__name__}")
        print(f"Model name: {backend.model_name}")
        print(f"Backend type: {config.backend_type}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Create backend from BackendConfig instance
    print("\nExample 2: Create backend from BackendConfig instance")
    try:
        from rl_onoff.backends.config import BackendConfig
        config = BackendConfig(
            backend_type="huggingface",
            model_name="gpt2",
            device="cpu"
        )
        backend = create_backend(config)
        print(f"Created backend: {backend.__class__.__name__}")
        print(f"Model name: {backend.model_name}")
        print(f"Device: {backend.device if hasattr(backend, 'device') else 'N/A'}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Note: Backends need to be loaded before use:")
    print("  backend.load()")
    print("  response = backend.generate('Hello')")

