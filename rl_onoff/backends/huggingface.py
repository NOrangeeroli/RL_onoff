"""HuggingFace Transformers backend implementation."""

from typing import List, Dict, Optional, Union, Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_onoff.backends.base import BaseBackend


class HuggingFaceBackend(BaseBackend):
    """Backend using HuggingFace Transformers."""

    def __init__(
        self,
        config: 'BackendConfig'
    ):
        """Initialize HuggingFace backend.
        
        Args:
            config: BackendConfig instance with backend configuration
        """
        from rl_onoff.backends.config import BackendConfig
        
        if not isinstance(config, BackendConfig):
            raise TypeError(f"config must be a BackendConfig instance, got {type(config)}")
        
        if config.backend_type != "huggingface":
            raise ValueError(f"BackendConfig backend_type must be 'huggingface', got '{config.backend_type}'")
        
        from rl_onoff.backends.config import HuggingFaceBackendConfig
        
        if not isinstance(config.backend_config, HuggingFaceBackendConfig):
            raise TypeError("config.backend_config must be HuggingFaceBackendConfig")
        
        super().__init__(config.model_name)
        
        # Extract HuggingFace-specific parameters from nested config
        hf_config = config.backend_config
        self.device = hf_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert torch_dtype from string if needed
        if hf_config.torch_dtype is not None:
            if isinstance(hf_config.torch_dtype, str):
                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                self.torch_dtype = dtype_map.get(hf_config.torch_dtype, hf_config.torch_dtype)
            else:
                self.torch_dtype = hf_config.torch_dtype
        else:
            self.torch_dtype = None
        
        self.device_map = hf_config.device_map

    def load(self) -> None:
        """Load the HuggingFace model and tokenizer."""
        if self._is_loaded:
            return

        print(f"Loading HuggingFace model: {self.model_name}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model loading kwargs
        model_kwargs = {}
        
        # Determine device_map: use "auto" if not specified and multiple GPUs available
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
            print(f"Using device_map: {self.device_map}")
        elif torch.cuda.device_count() > 1:
            # Use "auto" for multi-GPU setups
            model_kwargs["device_map"] = "auto"
            print(f"Using device_map: auto (for {torch.cuda.device_count()} GPUs)")
        else:
            # Single GPU or CPU: will set device manually after loading
            print(f"Using device: {self.device}")

        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
            print(f"Using torch_dtype: {self.torch_dtype}")

        # Load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # If not using device_map, move model to device
        if "device_map" not in model_kwargs:
            print(f"Moving model to device: {self.device}")
            self.model = self.model.to(self.device)
        
        # Print device information
        if "device_map" in model_kwargs:
            # When using device_map, print the actual device mapping
            if hasattr(self.model, 'hf_device_map'):
                print(f"  Model device mapping: {self.model.hf_device_map}")
            else:
                print(f"  Using device_map: {model_kwargs['device_map']}")
        else:
            # Single device case
            print(f"  Device used for generation: {self.device}")
        
        self.model.eval()
        self._is_loaded = True
        print("Model loaded successfully!")

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
        return_probs: bool = False
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
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
        # Determine target device for inputs
        # If using device_map, the model will automatically handle device placement
        # Otherwise, move inputs to the specified device
        if not hasattr(self.model, 'hf_device_map'):
            # Not using device_map: move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # If using device_map, inputs will be automatically moved by the model's generate() method

        # Prepare generation kwargs
        gen_kwargs = {
            "max_length": max_length,
            "do_sample": do_sample,
        }
        
        # Handle stop strings - need to pass tokenizer when using stop_strings
        if stop_strings is not None:
            gen_kwargs["stop_strings"] = stop_strings
            gen_kwargs["tokenizer"] = self.tokenizer
        
        # Request scores (logits) if needed
        if return_logits or return_probs:
            gen_kwargs["return_dict_in_generate"] = True
            gen_kwargs["output_scores"] = True
        
        if do_sample:
            gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

        # Generate
        # Note: device_map handles multi-GPU automatically, no need to unwrap DataParallel
        with torch.no_grad():
            if return_logits or return_probs:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                # outputs is a GenerateDecoderOnlyOutput object
                generated_ids = outputs.sequences
                scores = outputs.scores  # List of tensors, one per generated token
            else:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                generated_ids = outputs

        # Decode outputs (remove input tokens)
        input_lengths = inputs["input_ids"].shape[1]
        generated_token_ids = generated_ids[:, input_lengths:]
        texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

        # If not returning logits/probs, return texts as before
        if not return_logits and not return_probs:
            return texts[0] if is_single else texts

        # Prepare results with logits/probs
        results = []
        for i, text in enumerate(texts):
            result = {"text": text}
            
            if return_logits or return_probs:
                # Stack scores: (num_tokens, batch_size, vocab_size) -> (batch_size, num_tokens, vocab_size)
                if scores:
                    # scores is a tuple of tensors, one per generated token position
                    # Each tensor has shape (batch_size, vocab_size)
                    batch_logits = torch.stack(scores, dim=1)  # (batch_size, num_tokens, vocab_size)
                    logits_array = batch_logits[i].cpu().numpy()  # (num_tokens, vocab_size)
                    
                    if return_logits:
                        result["logits"] = logits_array
                    
                    if return_probs:
                        # Convert logits to probabilities
                        scaled_logits = logits_array / temperature
                        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
                        probs_array = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                        result["probs"] = probs_array
            
            results.append(result)

        return results[0] if is_single else results

    def get_logits(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get token logits for predicting response tokens given prompts."""
        if not self._is_loaded:
            self.load()

        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
            responses = [responses]
        
        if len(prompts) != len(responses):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of responses ({len(responses)})")

        # Concatenate prompts and responses
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]
        
        # Tokenize full texts (prompt + solution)
        full_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # Determine target device for inputs
        # If using device_map, the model will automatically handle device placement
        # Otherwise, move inputs to the specified device
        if not hasattr(self.model, 'hf_device_map'):
            # Not using device_map: move inputs to device
            full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}
        # If using device_map, inputs will be automatically moved by the model

        # Tokenize prompts separately to get prompt lengths
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # Move prompt_inputs to same device as full_inputs
        if not hasattr(self.model, 'hf_device_map'):
            # Not using device_map: move to device
            prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
        # If using device_map, inputs will be automatically moved by the model
        prompt_lengths = prompt_inputs["input_ids"].shape[1]

        all_logits = []
        
        # Note: device_map handles multi-GPU automatically, no need to unwrap DataParallel
        with torch.no_grad():
            # Get logits for the full sequence
            outputs = self.model(**full_inputs)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            
            # Extract logits for solution tokens only
            # Logits at position i predict token at position i+1
            # So for solution starting after prompt, we need logits at positions
            # that predict each solution token
            for i in range(len(prompts)):
                # Get actual prompt length (excluding padding)
                prompt_len = (prompt_inputs["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()
                if self.tokenizer.pad_token_id is None:
                    prompt_len = prompt_lengths
                
                # Get the actual sequence length for this item (excluding padding)
                seq_len = (full_inputs["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()
                if self.tokenizer.pad_token_id is None:
                    seq_len = full_inputs["input_ids"].shape[1]
                
                # Logits for response tokens: positions [prompt_len-1, prompt_len, ..., seq_len-2]
                # These predict tokens at positions [prompt_len, prompt_len+1, ..., seq_len-1]
                # We want logits that predict each response token
                response_logits = logits[i, prompt_len-1:seq_len-1, :]  # (response_len, vocab_size)
                all_logits.append(response_logits.cpu().numpy())

        if is_single:
            return all_logits[0]  # (response_len, vocab_size)
        return all_logits

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
    from rl_onoff.backends.config import BackendConfig
    from rl_onoff.backends import create_backend
    config = BackendConfig(
        backend_type="huggingface",
        model_name="meta-llama/Llama-3.2-1B",  # Replace with your model
        device="cuda"  # Use "cuda" if GPU is available
    )
    backend = create_backend(config)
    
    # Generate text from a single prompt
    prompt = "The future of AI is"
    generated = backend.generate(
        prompt,
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
    
    # Example 4: Get logits for predicting solution tokens
    print("=" * 60)
    print("Example 4: Get logits for predicting solution tokens")
    print("=" * 60)
    
    prompt = "The answer is"
    response = " 42."
    logits = backend.get_logits(prompt, response)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Logits shape: {logits.shape}")
    print(f"Number of response tokens: {logits.shape[0]}")
    print(f"Vocabulary size: {logits.shape[1]}")
    print(f"Top 5 token logits for first response token: {np.argsort(logits[0])[-5:][::-1]}\n")
    
    # Example 5: Get probability distributions for response tokens
    print("=" * 60)
    print("Example 5: Get probability distributions for response tokens")
    print("=" * 60)
    
    prompt = "The answer is"
    response = " 42."
    probs = backend.get_probabilities(prompt, response, temperature=1.0)
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


