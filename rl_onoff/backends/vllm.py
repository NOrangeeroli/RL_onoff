"""vLLM backend implementation."""

from typing import List, Dict, Optional, Union, Any
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
        config: 'BackendConfig'
    ):
        """Initialize vLLM backend.
        
        Args:
            config: BackendConfig instance with backend configuration
        """
        from rl_onoff.backends.config import BackendConfig
        
        if not isinstance(config, BackendConfig):
            raise TypeError(f"config must be a BackendConfig instance, got {type(config)}")
        
        if config.backend_type != "vllm":
            raise ValueError(f"BackendConfig backend_type must be 'vllm', got '{config.backend_type}'")
        
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            )
        
        from rl_onoff.backends.config import VLLMBackendConfig
        
        if not isinstance(config.backend_config, VLLMBackendConfig):
            raise TypeError("config.backend_config must be VLLMBackendConfig")
        
        super().__init__(config.model_name)
        
        # Extract vLLM-specific parameters from nested config
        vllm_config = config.backend_config
        self.tensor_parallel_size = vllm_config.tensor_parallel_size or 1
        self.gpu_memory_utilization = vllm_config.gpu_memory_utilization or 0.9
        self.max_model_len = vllm_config.max_model_len

    def load(self) -> None:
        """Load the vLLM model."""
        if self._is_loaded:
            return

        print(f"Loading vLLM model: {self.model_name}")
        
        load_kwargs = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        
        if self.max_model_len is not None:
            load_kwargs["max_model_len"] = self.max_model_len
        
        print(f"Configuration:")
        print(f"  Tensor parallel size: {self.tensor_parallel_size}")
        print(f"  GPU memory utilization: {self.gpu_memory_utilization}")
        if self.max_model_len is not None:
            print(f"  Max model length: {self.max_model_len}")
        
        print("Loading model...")
        self.model = LLM(model=self.model_name, **load_kwargs)
        # vLLM doesn't expose tokenizer directly, but we can access it
        # through the model's tokenizer attribute
        self._is_loaded = True
        print("Model loaded successfully!")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 0.9,
        top_k: Optional[int] = -1,
        top_p: Optional[float] = 1,
        do_sample: bool = True,
        stop_strings: Optional[List[str]] = None,
        return_logits: bool = False,
        return_probs: bool = False,
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """Generate text from prompts."""
        if not self._is_loaded:
            self.load()

        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]

        # Create sampling parameters
        sampling_params_kwargs = {
            "max_tokens": max_length,
            "temperature": temperature if do_sample else 0.0,
            "top_k": top_k,
            "top_p": top_p,
        }
        
        # Handle stop strings
        if stop_strings is not None:
            sampling_params_kwargs["stop"] = stop_strings
        
        sampling_params = SamplingParams(**sampling_params_kwargs)

        # Check if logits/probs are requested (not yet supported for vLLM)
        if return_logits or return_probs:
            raise NotImplementedError(
                "return_logits and return_probs are not yet supported for vLLM backend. "
                "Please use HuggingFace backend for this functionality."
            )
        
        # Generate
        outputs = self.model.generate(prompts, sampling_params)
        
        # Extract generated text
        texts = [output.outputs[0].text for output in outputs]

        return texts[0] if is_single else texts

    def get_logits(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get token logits for predicting response tokens given prompts."""
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


if __name__ == "__main__":
    """Simple use cases for VLLMBackend."""
    
    if not VLLM_AVAILABLE:
        print("vLLM is not installed. Install it with: pip install vllm")
        print("Skipping examples.")
    else:
        # Example 1: Basic text generation
        print("=" * 60)
        print("Example 1: Basic text generation")
        print("=" * 60)
        
        # Initialize backend (replace with your preferred model)
        from rl_onoff.backends.config import BackendConfig
        from rl_onoff.backends import create_backend
        config = BackendConfig(
            backend_type="vllm",
            model_name="meta-llama/Llama-3.2-1B",  # Replace with your model
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        )
        backend = create_backend(config)
        
        # Generate text from a single prompt
        prompt = "The future of AI is"
        generated = backend.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True
        )
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}\n")
        
        # Example 2: Generate from multiple prompts
        print("=" * 60)
        print("Example 2: Generate from multiple prompts")
        print("=" * 60)
        
        prompts = [
            "Python is",
            "Machine learning is"
        ]
        generated_texts = backend.generate(
            prompts,
            max_new_tokens=15,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        for prompt, text in zip(prompts, generated_texts):
            print(f"Prompt: {prompt}")
            print(f"Generated: {text}\n")
        
        # Example 3: Encode and decode text
        print("=" * 60)
        print("Example 3: Encode and decode text")
        print("=" * 60)
        
        tokenizer = backend.get_tokenizer()
        if tokenizer is not None:
            text = "Hello, world!"
            token_ids = backend.encode(text)
            print(f"Text: {text}")
            print(f"Token IDs: {token_ids}")
            
            decoded = backend.decode(token_ids)
            print(f"Decoded: {decoded}\n")
        else:
            print("Tokenizer not available (llm_engine not accessible)\n")
        
        # Example 4: Get logits (Note: Not implemented for vLLM)
        print("=" * 60)
        print("Example 4: Get logits (Not implemented for vLLM)")
        print("=" * 60)
        
        try:
            logits = backend.get_logits("The answer is", " 42.")
            print(f"Logits shape: {logits.shape}")
        except NotImplementedError as e:
            print(f"Note: {e}\n")
        
        # Example 5: Access tokenizer directly
        print("=" * 60)
        print("Example 5: Access tokenizer directly")
        print("=" * 60)
        
        tokenizer = backend.get_tokenizer()
        if tokenizer is not None:
            print(f"Tokenizer: {type(tokenizer).__name__}")
            if hasattr(tokenizer, 'vocab_size'):
                print(f"Vocabulary size: {tokenizer.vocab_size}\n")
        else:
            print("Tokenizer not available (llm_engine not accessible)\n")

