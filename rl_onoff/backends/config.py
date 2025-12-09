"""Backend configuration management."""

from typing import Optional, Dict, Any, Union, Literal, List
from dataclasses import dataclass
from pathlib import Path

from rl_onoff.utils.config import Config

# Type aliases for valid values
BackendType = Literal["huggingface", "vllm", "sglang"]


@dataclass
class BackendConfig(Config):
    """Configuration for a backend.
    
    Specifies:
    - backend_type: Type of backend to use
    - model_name: Model name or path
    - Additional backend-specific parameters
    """
    
    backend_type: BackendType = "huggingface"
    model_name: str = "gpt2"
    
    # Backend-specific parameters
    # HuggingFace parameters
    device: Optional[str] = None  # "cpu", "cuda", "cuda:0", etc.
    torch_dtype: Optional[str] = None  # "float32", "float16", "bfloat16"
    device_map: Optional[str] = None  # "auto", "balanced", etc.
    
    # vLLM parameters
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    
    # SGLang parameters
    tp_size: Optional[int] = None  # Tensor parallelism size
    mem_fraction_static: Optional[float] = None
    context_length: Optional[int] = None
    
    # Additional kwargs for backend initialization
    backend_kwargs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values for optional fields and validate types."""
        if self.backend_kwargs is None:
            self.backend_kwargs = {}
        
        # Validate backend_type
        valid_backends = ["huggingface", "vllm", "sglang"]
        if self.backend_type not in valid_backends:
            raise ValueError(
                f"Invalid backend_type: '{self.backend_type}'. "
                f"Must be one of: {valid_backends}"
            )
    
    def get_backend_kwargs(self) -> Dict[str, Any]:
        """Get backend-specific kwargs for initialization.
        
        Returns:
            Dictionary of kwargs specific to the backend type
        """
        kwargs = {}
        
        if self.backend_type == "huggingface":
            if self.device is not None:
                kwargs["device"] = self.device
            if self.torch_dtype is not None:
                # Convert string to torch dtype if needed
                try:
                    import torch
                    dtype_map = {
                        "float32": torch.float32,
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                    }
                    if self.torch_dtype in dtype_map:
                        kwargs["torch_dtype"] = dtype_map[self.torch_dtype]
                    else:
                        # Try to use as-is (might already be a torch dtype)
                        kwargs["torch_dtype"] = self.torch_dtype
                except ImportError:
                    # If torch not available, pass as string
                    kwargs["torch_dtype"] = self.torch_dtype
            if self.device_map is not None:
                kwargs["device_map"] = self.device_map
        
        elif self.backend_type == "vllm":
            if self.tensor_parallel_size is not None:
                kwargs["tensor_parallel_size"] = self.tensor_parallel_size
            if self.gpu_memory_utilization is not None:
                kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
            if self.max_model_len is not None:
                kwargs["max_model_len"] = self.max_model_len
        
        elif self.backend_type == "sglang":
            if self.tp_size is not None:
                kwargs["tp_size"] = self.tp_size
            if self.mem_fraction_static is not None:
                kwargs["mem_fraction_static"] = self.mem_fraction_static
            if self.context_length is not None:
                kwargs["context_length"] = self.context_length
        
        # Merge with additional kwargs
        kwargs.update(self.backend_kwargs)
        
        return kwargs
    
    def create_backend(self):
        """Create a backend instance from this config.
        
        Returns:
            Backend instance
        """
        from rl_onoff.backends import create_backend
        kwargs = self.get_backend_kwargs()
        return create_backend(self.backend_type, model_name=self.model_name, **kwargs)

