"""HuggingFace Transformers backend implementation."""

from typing import List, Dict, Optional, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_onoff.backends.base import BaseBackend


class HuggingFaceBackend(BaseBackend):
    """Backend using HuggingFace Transformers."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """Initialize HuggingFace backend.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on (default: auto-detect)
            torch_dtype: Torch dtype for model (default: auto)
            **kwargs: Additional arguments passed to model/tokenizer
        """
        super().__init__(model_name, **kwargs)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.model_kwargs = kwargs

    def load(self, **kwargs) -> None:
        """Load the HuggingFace model and tokenizer."""
        if self._is_loaded:
            return

        # Merge initialization kwargs with load kwargs
        load_kwargs = {**self.model_kwargs, **kwargs}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **{k: v for k, v in load_kwargs.items() if k not in ["device_map", "torch_dtype"]}
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model loading kwargs
        model_kwargs = {}
        if "device_map" in load_kwargs:
            model_kwargs["device_map"] = load_kwargs["device_map"]
        else:
            model_kwargs["device_map"] = self.device

        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        elif "torch_dtype" in load_kwargs:
            model_kwargs["torch_dtype"] = load_kwargs["torch_dtype"]

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
            **{k: v for k, v in load_kwargs.items() if k not in ["device_map", "torch_dtype"]}
        )
        
        # If not using device_map, move model to device
        if "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)
        
        self.model.eval()
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

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            **kwargs
        }
        
        if do_sample:
            gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode outputs (remove input tokens)
        input_lengths = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_lengths:]
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

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

        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        all_logits = []
        
        with torch.no_grad():
            # Get initial logits
            outputs = self.model(**inputs)
            current_logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            
            # Get logits for each new token position
            generated_ids = inputs["input_ids"].clone()
            
            for _ in range(max_new_tokens):
                # Get logits for the last token position
                next_token_logits = current_logits[:, -1:, :]  # (batch_size, 1, vocab_size)
                all_logits.append(next_token_logits.cpu().numpy())
                
                # Sample next token (greedy for simplicity)
                next_token_ids = torch.argmax(next_token_logits, dim=-1)
                generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
                
                # Get logits for next position
                if _ < max_new_tokens - 1:
                    outputs = self.model(generated_ids)
                    current_logits = outputs.logits

        # Concatenate logits along sequence dimension
        # all_logits is list of (batch_size, 1, vocab_size)
        # We want (batch_size, max_new_tokens, vocab_size)
        batch_logits = np.concatenate(all_logits, axis=1)  # (batch_size, max_new_tokens, vocab_size)
        
        if is_single:
            return batch_logits[0]  # (max_new_tokens, vocab_size)
        return [batch_logits[i] for i in range(batch_logits.shape[0])]

    def get_tokenizer(self):
        """Get the tokenizer instance."""
        if not self._is_loaded:
            self.load()
        return self.tokenizer


if __name__ == "__main__":
    """Simple use cases for HuggingFaceBackend."""
    
    # Example 1: Basic text generation
    print("=" * 60)
    print("Example 1: Basic text generation")
    print("=" * 60)
    
    # Initialize backend (use a small model for testing, e.g., "gpt2")
    # Replace with your preferred model name
    backend = HuggingFaceBackend(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",  # Replace with your model
        device="cuda"  # Use "cuda" if GPU is available
    )
    
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
        temperature=0.8
    )
    for prompt, text in zip(prompts, generated_texts):
        print(f"Prompt: {prompt}")
        print(f"Generated: {text}\n")
    
    # Example 3: Encode and decode text
    print("=" * 60)
    print("Example 3: Encode and decode text")
    print("=" * 60)
    
    text = "Hello, world!"
    token_ids = backend.encode(text)
    print(f"Text: {text}")
    print(f"Token IDs: {token_ids}")
    
    decoded = backend.decode(token_ids)
    print(f"Decoded: {decoded}\n")
    
    # Example 4: Get logits for a prompt
    print("=" * 60)
    print("Example 4: Get logits for a prompt")
    print("=" * 60)
    
    logits = backend.get_logits("The answer is", max_new_tokens=1)
    print(f"Logits shape: {logits.shape}")
    print(f"Vocabulary size: {logits.shape[-1]}")
    print(f"Top 5 token logits: {logits[0][:5]}\n")
    
    # Example 5: Get probability distributions
    print("=" * 60)
    print("Example 5: Get probability distributions")
    print("=" * 60)
    
    probs = backend.get_probabilities("The answer is", max_new_tokens=1, temperature=1.0)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sum of probabilities: {probs.sum():.4f}")  # Should be ~1.0
    print(f"Top 5 probabilities: {probs[0][:5]}\n")
    
    # Example 6: Access tokenizer directly
    print("=" * 60)
    print("Example 6: Access tokenizer directly")
    print("=" * 60)
    
    tokenizer = backend.get_tokenizer()
    print(f"Tokenizer: {type(tokenizer).__name__}")
    print(f"Vocabulary size: {len(tokenizer)}\n")


