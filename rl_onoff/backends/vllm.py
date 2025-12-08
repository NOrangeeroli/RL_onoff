"""vLLM backend implementation."""

from typing import List, Dict, Optional, Union
import numpy as np

from rl_onoff.backends.base import BaseBackend

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None


class VLLMBackend(BaseBackend):
    """Backend using vLLM for fast inference."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        """Initialize vLLM backend.
        
        Args:
            model_name: Model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            **kwargs: Additional vLLM arguments
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            )
        
        super().__init__(model_name, **kwargs)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.vllm_kwargs = kwargs

    def load(self, **kwargs) -> None:
        """Load the vLLM model."""
        if self._is_loaded:
            return

        load_kwargs = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            **self.vllm_kwargs,
            **kwargs
        }

        self.model = LLM(model=self.model_name, **load_kwargs)
        # vLLM doesn't expose tokenizer directly, but we can access it
        # through the model's tokenizer attribute
        self._is_loaded = True

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate text from prompts."""
        if not self._is_loaded:
            self.load()

        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]

        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )

        # Generate
        outputs = self.model.generate(prompts, sampling_params)
        
        # Extract generated text
        texts = [output.outputs[0].text for output in outputs]

        return texts[0] if is_single else texts

    def get_logits(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 1,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get token logits for given prompts."""
        if not self._is_loaded:
            self.load()

        # Note: vLLM doesn't directly expose logits in a straightforward way
        # This is a simplified implementation that may need adjustment
        # based on vLLM's actual API for logit extraction
        
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]

        # vLLM API for logits may vary - this is a placeholder
        # In practice, you might need to use vLLM's internal APIs
        # or modify the model wrapper to expose logits
        
        # For now, raise a NotImplementedError with guidance
        raise NotImplementedError(
            "vLLM logit extraction requires special handling. "
            "You may need to use vLLM's internal APIs or modify the backend. "
            "Consider using HuggingFace backend for logit extraction, "
            "or use vLLM's model runner directly."
        )

    def get_tokenizer(self):
        """Get the tokenizer instance."""
        if not self._is_loaded:
            self.load()
        
        # vLLM models have a tokenizer attribute
        if hasattr(self.model, "llm_engine") and self.model.llm_engine is not None:
            return self.model.llm_engine.tokenizer.tokenizer
        return None

