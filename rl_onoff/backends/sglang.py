"""SGLang backend implementation."""

from typing import List, Dict, Optional, Union
import numpy as np

from rl_onoff.backends.base import BaseBackend

try:
    import sglang as sgl
    from sglang.srt.hf_transformers_utils import get_tokenizer
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    sgl = None
    get_tokenizer = None


class SGLangBackend(BaseBackend):
    """Backend using SGLang for structured generation."""

    def __init__(
        self,
        model_name: str,
        tp_size: int = 1,
        mem_fraction_static: float = 0.85,
        **kwargs
    ):
        """Initialize SGLang backend.
        
        Args:
            model_name: Model name or path
            tp_size: Tensor parallelism size
            mem_fraction_static: Memory fraction for static allocation
            **kwargs: Additional SGLang arguments
        """
        if not SGLANG_AVAILABLE:
            raise ImportError(
                "SGLang is not installed. Install it with: pip install 'sglang[all]'"
            )
        
        super().__init__(model_name, **kwargs)
        self.tp_size = tp_size
        self.mem_fraction_static = mem_fraction_static
        self.sglang_kwargs = kwargs

    def load(self, **kwargs) -> None:
        """Load the SGLang runtime."""
        if self._is_loaded:
            return

        load_kwargs = {
            "tp_size": self.tp_size,
            "mem_fraction_static": self.mem_fraction_static,
            **self.sglang_kwargs,
            **kwargs
        }

        # Start SGLang runtime
        self.runtime = sgl.Runtime(
            model_path=self.model_name,
            **load_kwargs
        )
        
        # Get tokenizer
        self.tokenizer = get_tokenizer(self.model_name, trust_remote_code=True)
        
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

        # Prepare sampling parameters
        sampling_params = {
            "temperature": temperature if do_sample else 0.0,
            "max_new_tokens": max_new_tokens,
            **kwargs
        }
        
        if top_k is not None:
            sampling_params["top_k"] = top_k
        if top_p is not None:
            sampling_params["top_p"] = top_p

        # Generate using SGLang
        # Note: SGLang API may vary - this is a simplified version
        outputs = []
        for prompt in prompts:
            state = self.runtime.run(prompt, **sampling_params)
            outputs.append(state.text)

        return outputs[0] if is_single else outputs

    def get_logits(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 1,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get token logits for given prompts."""
        if not self._is_loaded:
            self.load()

        # Note: SGLang's logit extraction API may require special handling
        # This is a placeholder implementation
        
        raise NotImplementedError(
            "SGLang logit extraction requires special handling. "
            "You may need to use SGLang's internal APIs or modify the backend. "
            "Consider using HuggingFace backend for logit extraction."
        )

    def get_tokenizer(self):
        """Get the tokenizer instance."""
        if not self._is_loaded:
            self.load()
        return self.tokenizer

