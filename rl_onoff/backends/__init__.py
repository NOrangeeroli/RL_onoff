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


def create_backend(backend_type: str = None, config: 'BackendConfig' = None, model_name: str = None, **kwargs):
    """Get a backend instance by type.
    
    Args:
        backend_type: Type of backend ("huggingface", "vllm", "sglang")
        config: BackendConfig instance (if provided, overrides backend_type and kwargs)
        model_name: Model name or path (required if config not provided)
        **kwargs: Additional backend-specific arguments
    
    Returns:
        Backend instance
    """
    # If config is provided, use it
    if config is not None:
        backend_type = config.backend_type
        backend_kwargs = config.get_backend_kwargs()
        backend_kwargs.update(kwargs)  # kwargs override config
        kwargs = backend_kwargs
        model_name = config.model_name
    else:
        # Use provided model_name or extract from kwargs
        if model_name is None:
            model_name = kwargs.pop('model_name', None)
        if model_name is None:
            raise ValueError("model_name must be provided either in config or as argument")
    
    backend_map = {
        "huggingface": HuggingFaceBackend,
        "vllm": VLLMBackend,
        "sglang": SGLangBackend,
    }
    
    if backend_type is None:
        raise ValueError("backend_type must be provided")
    
    if backend_type.lower() not in backend_map:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Supported: {list(backend_map.keys())}"
        )
    
    return backend_map[backend_type.lower()](model_name=model_name, **kwargs)


if __name__ == "__main__":
    # Simple usage example for create_backend
    print("create_backend Usage Example")
    print("=" * 50)
    
    # Example 1: Create backend from type and model name
    print("\nExample 1: Create backend from type and model name")
    try:
        backend = create_backend(backend_type="huggingface", model_name="gpt2")
        print(f"Created backend: {backend.__class__.__name__}")
        print(f"Model name: {backend.model_name}")
        print(f"Loaded: {backend.is_loaded()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Create backend from config file
    print("\nExample 2: Create backend from config file")
    try:
        from rl_onoff.backends.config import BackendConfig
        config = BackendConfig.from_json("rl_onoff/backends/configs/huggingface_default.json")
        backend = create_backend(config=config)
        print(f"Created backend from config: {backend.__class__.__name__}")
        print(f"Model name: {backend.model_name}")
        print(f"Backend type: {config.backend_type}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Create backend with custom parameters
    print("\nExample 3: Create backend with custom parameters")
    try:
        backend = create_backend(
            backend_type="huggingface",
            model_name="gpt2",
            device="cpu"  # Custom parameter
        )
        print(f"Created backend with custom device: {backend.device if hasattr(backend, 'device') else 'N/A'}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Note: Backends need to be loaded before use:")
    print("  backend.load()")
    print("  response = backend.generate('Hello')")

