# Backends Module

The backends module provides a unified interface for running LLM inference using different inference engines. It supports HuggingFace Transformers, vLLM, and SGLang backends, each optimized for different use cases.

## Overview

Backends handle model loading, text generation, and token-level operations (encoding, decoding, logits extraction). All backends implement the same `BaseBackend` interface, making it easy to switch between different inference engines without changing your code.

## Available Backends

### 1. HuggingFace Backend

The HuggingFace backend uses the `transformers` library for inference.

**Features:**
- Full PyTorch integration
- Support for all HuggingFace models
- Multi-GPU data parallelism via `torch.nn.DataParallel`
- Model parallelism via `device_map`
- Support for logits/probabilities extraction
- Flexible device placement (CPU, CUDA, device_map)

**Best for:**
- Development and prototyping
- Models not optimized by vLLM/SGLang
- When you need full control over model loading
- Multi-GPU setups using data parallelism

**Dependencies:** `torch`, `transformers`

### 2. vLLM Backend

The vLLM backend uses the vLLM library for fast inference.

**Features:**
- Optimized for high-throughput inference
- Tensor parallelism support
- Efficient memory management
- Continuous batching
- PagedAttention for memory efficiency

**Best for:**
- Production deployments requiring high throughput
- Large batch inference
- Multi-GPU tensor parallelism

**Dependencies:** `vllm`

**Limitations:**
- Logits/probabilities extraction not yet implemented

### 3. SGLang Backend

The SGLang backend uses SGLang for structured generation.

**Features:**
- Optimized for structured generation
- Tensor parallelism support
- Efficient memory allocation

**Best for:**
- Structured generation tasks
- High-throughput structured outputs

**Dependencies:** `sglang`

**Limitations:**
- Logits/probabilities extraction not yet implemented

## Quick Start

### Basic Usage

The simplest way to create a backend is using a dictionary config:

```python
from rl_onoff.backends import create_backend

# Create backend from dict
backend = create_backend({
    "backend_type": "huggingface",
    "model_name": "gpt2",
    "backend_specific": {
        "device": "cuda"
    }
})

# Load the model
backend.load()

# Generate text
response = backend.generate("The future of AI is")
print(response)
```

### Using Config Files

You can also use YAML config files:

```python
from rl_onoff.backends.config import BackendConfig
from rl_onoff.backends import create_backend

# Load config from file
config = BackendConfig.from_file("rl_onoff/backends/configs/huggingface_default.yaml")
backend = create_backend(config)
backend.load()
```

### Using in Experiment Scripts

In experiment scripts, you can pass the backend config directly from your experiment config:

```python
import yaml
from rl_onoff.backends import create_backend

# Load experiment config
with open("exp/experiment_config.yaml") as f:
    exp_config = yaml.safe_load(f)

# Get backend config section
backend_config_dict = exp_config.get("backend", {})

# Create backend - it automatically handles the backend type
backend = create_backend(backend_config_dict)
backend.load()
```

## Configuration

### Configuration Structure

All backend configurations use the same top-level structure:

```yaml
backend_type: "huggingface"  # or "vllm" or "sglang"
model_name: "gpt2"  # Model name or path
backend_specific:
  # Backend-specific parameters (see below)
```

### HuggingFace Configuration

```yaml
backend_type: "huggingface"
model_name: "gpt2"
backend_specific:
  device: "cuda"  # "cpu", "cuda", "cuda:0", etc. (null = auto-detect)
  torch_dtype: "float16"  # "float32", "float16", "bfloat16" (null = auto)
  device_map: "auto"  # "auto", "balanced", etc. (null = use device)
```

**Parameters:**
- `device`: Target device for the model. If `null`, auto-detects CUDA availability.
- `torch_dtype`: Data type for model weights. Useful for memory efficiency (`float16`, `bfloat16`).
- `device_map`: Enables model parallelism. Options include `"auto"`, `"balanced"`, or custom mappings. If set, overrides `device`.

**Multi-GPU Notes:**
- If `device_map` is not set and multiple GPUs are available, the backend automatically uses `DataParallel` for data parallelism (splits batches across GPUs).
- If `device_map` is set, it uses model parallelism (splits model layers across GPUs).

### vLLM Configuration

```yaml
backend_type: "vllm"
model_name: "gpt2"
backend_specific:
  tensor_parallel_size: 2  # Number of GPUs for tensor parallelism (null = 1)
  gpu_memory_utilization: 0.9  # GPU memory utilization (null = 0.9)
  max_model_len: 2048  # Maximum model length (null = auto)
```

**Parameters:**
- `tensor_parallel_size`: Number of GPUs to use for tensor parallelism.
- `gpu_memory_utilization`: Fraction of GPU memory to use (0.0-1.0).
- `max_model_len`: Maximum sequence length the model can handle.

### SGLang Configuration

```yaml
backend_type: "sglang"
model_name: "gpt2"
backend_specific:
  tp_size: 2  # Tensor parallelism size (null = 1)
  mem_fraction_static: 0.85  # Memory fraction for static allocation (null = 0.85)
  context_length: 2048  # Context length (null = auto)
```

**Parameters:**
- `tp_size`: Number of GPUs for tensor parallelism.
- `mem_fraction_static`: Fraction of GPU memory for static allocation.
- `context_length`: Maximum context length.

## API Reference

### BaseBackend Interface

All backends implement the `BaseBackend` interface:

```python
class BaseBackend:
    def load(self) -> None:
        """Load the model and tokenizer."""
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        stop_strings: Optional[List[str]] = None,
        return_logits: bool = False,
        return_probs: bool = False,
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """Generate text from prompts."""
    
    def get_logits(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get token logits for response tokens given prompts."""
    
    def get_probabilities(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
        temperature: float = 1.0,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get token probability distributions."""
    
    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """Encode text to token IDs."""
    
    def decode(self, token_ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """Decode token IDs to text."""
    
    def get_tokenizer(self):
        """Get the tokenizer instance."""
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
```

## Usage Examples

### Text Generation

```python
from rl_onoff.backends import create_backend

backend = create_backend({
    "backend_type": "huggingface",
    "model_name": "gpt2",
    "backend_specific": {"device": "cuda"}
})
backend.load()

# Single prompt
response = backend.generate("The capital of France is")
print(response)

# Multiple prompts
responses = backend.generate([
    "The capital of France is",
    "The capital of Germany is"
])
print(responses)
```

### Generation Parameters

```python
# Sampling with temperature and top-k
response = backend.generate(
    "Once upon a time",
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    do_sample=True
)

# Greedy decoding
response = backend.generate(
    "The answer is",
    max_length=50,
    do_sample=False
)

# With stop strings
response = backend.generate(
    "Write a story:",
    max_length=200,
    stop_strings=["\n\n", "THE END"]
)
```

### Logits and Probabilities

```python
# Get logits during generation
result = backend.generate(
    "The capital of France is",
    max_length=10,
    return_logits=True
)
print(result["text"])  # Generated text
print(result["logits"].shape)  # (num_tokens, vocab_size)

# Get logits for existing text
prompt = "The capital of France is"
response = "Paris"
logits = backend.get_logits(prompt, response)
# Shape: (len(response_tokens), vocab_size)

# Get probabilities
probs = backend.get_probabilities(prompt, response, temperature=1.0)
# Shape: (len(response_tokens), vocab_size)
```

### Tokenization

```python
# Encode text to token IDs
token_ids = backend.encode("Hello, world!")
print(token_ids)  # [15496, 11, 995, 0]

# Decode token IDs to text
text = backend.decode([15496, 11, 995, 0])
print(text)  # "Hello, world!"

# Batch operations
token_ids_list = backend.encode(["Hello", "World"])
texts = backend.decode(token_ids_list)
```

### Backend Switching

Since all backends share the same interface, you can easily switch between them:

```python
# HuggingFace backend
hf_backend = create_backend({
    "backend_type": "huggingface",
    "model_name": "gpt2",
    "backend_specific": {"device": "cuda"}
})

# vLLM backend (same API)
vllm_backend = create_backend({
    "backend_type": "vllm",
    "model_name": "gpt2",
    "backend_specific": {"tensor_parallel_size": 1}
})

# Both backends work the same way
response1 = hf_backend.generate("Hello")
response2 = vllm_backend.generate("Hello")
```

## Backend-Specific Features

### HuggingFace: Multi-GPU Data Parallelism

The HuggingFace backend automatically uses `DataParallel` when multiple GPUs are available (unless `device_map` is set):

```python
# Automatically uses all available GPUs for data parallelism
backend = create_backend({
    "backend_type": "huggingface",
    "model_name": "gpt2",
    "backend_specific": {"device": "cuda"}  # No device_map
})
backend.load()  # Will print: "Wrapping model with DataParallel for N GPUs"
```

### HuggingFace: Model Parallelism

Use `device_map` for model parallelism (splits model layers across GPUs):

```python
backend = create_backend({
    "backend_type": "huggingface",
    "model_name": "gpt2",
    "backend_specific": {
        "device_map": "auto"  # Automatically distributes layers
    }
})
backend.load()
```

### vLLM: Tensor Parallelism

```python
backend = create_backend({
    "backend_type": "vllm",
    "model_name": "gpt2",
    "backend_specific": {
        "tensor_parallel_size": 2  # Use 2 GPUs
    }
})
backend.load()
```

### SGLang: Tensor Parallelism

```python
backend = create_backend({
    "backend_type": "sglang",
    "model_name": "gpt2",
    "backend_specific": {
        "tp_size": 2  # Use 2 GPUs
    }
})
backend.load()
```

## Configuration Files

Default configuration files are available in `rl_onoff/backends/configs/`:

- `huggingface_default.yaml`: Default HuggingFace configuration
- `vllm_default.yaml`: Default vLLM configuration
- `sglang_default.yaml`: Default SGLang configuration

You can use these as templates or load them directly:

```python
from rl_onoff.backends.config import BackendConfig
from rl_onoff.backends import create_backend

config = BackendConfig.from_file("rl_onoff/backends/configs/huggingface_default.yaml")
backend = create_backend(config)
```

## Backend Selection Guide

Choose a backend based on your needs:

| Feature | HuggingFace | vLLM | SGLang |
|---------|-------------|------|--------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Throughput** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Memory Efficiency** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Logits Extraction** | ✅ | ❌ | ❌ |
| **Model Support** | All HF models | Many models | Many models |
| **Multi-GPU** | Data + Model parallel | Tensor parallel | Tensor parallel |
| **Best For** | Development, prototyping | Production, high throughput | Structured generation |

## Implementation Details

### Backend Architecture

The backends module uses a factory pattern with type-specific configuration classes:

1. **BackendConfig**: Main configuration class that holds `backend_type`, `model_name`, and a nested `backend_config`
2. **Backend-Specific Configs**: `HuggingFaceBackendConfig`, `VLLMBackendConfig`, `SGLangBackendConfig`
3. **create_backend()**: Factory function that automatically selects and instantiates the correct backend based on config

### Configuration Structure

The configuration system automatically:
- Detects backend type from `backend_type` field
- Creates appropriate backend-specific config class
- Validates configuration parameters
- Handles nested `backend_specific` parameters

### Error Handling

- Backends validate their configuration on initialization
- Missing dependencies raise clear `ImportError` messages
- Invalid backend types raise `ValueError` with available options
- Configuration mismatches (e.g., HuggingFace config for vLLM backend) are caught early

## Dependencies

- **HuggingFace**: `torch`, `transformers`
- **vLLM**: `vllm`
- **SGLang**: `sglang`

## Notes

- Backends must be loaded (call `backend.load()`) before generating text
- HuggingFace backend supports both data parallelism (via DataParallel) and model parallelism (via device_map), but not simultaneously
- Logits/probabilities extraction is currently only supported by HuggingFace backend
- All backends use left padding for tokenization to support batched inference
- Stop strings are handled by the underlying inference engines

