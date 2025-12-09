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

