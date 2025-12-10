"""Backend configuration management."""

from typing import Optional, Dict, Any, Union, Literal
from dataclasses import dataclass
from rl_onoff.utils.config import Config

# Type aliases for valid values
BackendType = Literal["huggingface", "vllm", "sglang"]


@dataclass
class HuggingFaceBackendConfig(Config):
    """HuggingFace-specific backend configuration."""
    device: Optional[str] = None  # "cpu", "cuda", "cuda:0", etc.
    torch_dtype: Optional[str] = None  # "float32", "float16", "bfloat16"
    device_map: Optional[str] = None  # "auto", "balanced", etc.
    tp_size: Optional[int] = None  # Tensor parallelism size (number of GPUs per model replica)
    token: Optional[str] = None  # HuggingFace access token for gated models (or use HF_TOKEN env var)


@dataclass
class VLLMBackendConfig(Config):
    """vLLM-specific backend configuration."""
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None


@dataclass
class SGLangBackendConfig(Config):
    """SGLang-specific backend configuration."""
    tp_size: Optional[int] = None  # Tensor parallelism size
    mem_fraction_static: Optional[float] = None
    context_length: Optional[int] = None


@dataclass
class BackendConfig(Config):
    """Configuration for a backend.
    
    Automatically switches between backend-specific configs based on backend_type.
    """
    backend_type: BackendType = "huggingface"
    model_name: str = "gpt2"
    backend_config: Optional[Union[HuggingFaceBackendConfig, VLLMBackendConfig, SGLangBackendConfig]] = None
    
    def __post_init__(self):
        """Initialize default backend_config if not provided."""
        if self.backend_config is None:
            if self.backend_type == "huggingface":
                self.backend_config = HuggingFaceBackendConfig()
            elif self.backend_type == "vllm":
                self.backend_config = VLLMBackendConfig()
            elif self.backend_type == "sglang":
                self.backend_config = SGLangBackendConfig()
        
        # Validate backend_type
        valid_backends = ["huggingface", "vllm", "sglang"]
        if self.backend_type not in valid_backends:
            raise ValueError(
                f"Invalid backend_type: '{self.backend_type}'. "
                f"Must be one of: {valid_backends}"
            )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackendConfig':
        """Create BackendConfig from dictionary, auto-detecting backend type.
        
        Expected structure:
        {
            "backend_type": "huggingface",
            "model_name": "gpt2",
            "backend_specific": {
                "device": "cuda",
                "torch_dtype": "float16",
                ...
            }
        }
        
        Also supports backward compatibility with top-level parameters.
        """
        backend_type = data.get("backend_type", "huggingface")
        model_name = data.get("model_name", "gpt2")
        
        # Extract backend_specific parameters (preferred structure)
        backend_specific = data.get("backend_specific", {})
        
        # If backend_specific is not present, try to extract from top level (backward compatibility)
        if not backend_specific:
            # For HuggingFace
            if any(k in data for k in ["device", "torch_dtype", "device_map", "tp_size", "num_process", "token"]):
                backend_specific = {
                    "device": data.get("device"),
                    "torch_dtype": data.get("torch_dtype"),
                    "device_map": data.get("device_map"),
                    "tp_size": data.get("tp_size") or data.get("num_process"),  # Backward compatibility
                    "token": data.get("token"),
                }
            # For vLLM
            elif any(k in data for k in ["tensor_parallel_size", "gpu_memory_utilization", "max_model_len"]):
                backend_specific = {
                    "tensor_parallel_size": data.get("tensor_parallel_size"),
                    "gpu_memory_utilization": data.get("gpu_memory_utilization"),
                    "max_model_len": data.get("max_model_len"),
                }
            # For SGLang
            elif any(k in data for k in ["tp_size", "mem_fraction_static", "context_length"]):
                backend_specific = {
                    "tp_size": data.get("tp_size"),
                    "mem_fraction_static": data.get("mem_fraction_static"),
                    "context_length": data.get("context_length"),
                }
        
        # Create backend-specific config based on type
        if backend_type == "huggingface":
            backend_config = HuggingFaceBackendConfig.from_dict({
                "device": backend_specific.get("device"),
                "torch_dtype": backend_specific.get("torch_dtype"),
                "device_map": backend_specific.get("device_map"),
                "tp_size": backend_specific.get("tp_size") or backend_specific.get("num_process"),  # Backward compatibility
                "token": backend_specific.get("token"),
            })
        elif backend_type == "vllm":
            backend_config = VLLMBackendConfig.from_dict({
                "tensor_parallel_size": backend_specific.get("tensor_parallel_size"),
                "gpu_memory_utilization": backend_specific.get("gpu_memory_utilization"),
                "max_model_len": backend_specific.get("max_model_len"),
            })
        elif backend_type == "sglang":
            backend_config = SGLangBackendConfig.from_dict({
                "tp_size": backend_specific.get("tp_size"),
                "mem_fraction_static": backend_specific.get("mem_fraction_static"),
                "context_length": backend_specific.get("context_length"),
            })
        else:
            raise ValueError(
                f"Invalid backend_type: '{backend_type}'. "
                f"Must be one of: ['huggingface', 'vllm', 'sglang']"
            )
        
        return cls(
            backend_type=backend_type,
            model_name=model_name,
            backend_config=backend_config
        )
