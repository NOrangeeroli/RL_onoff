"""HuggingFace Transformers backend implementation."""

from typing import List, Dict, Optional, Union, Any
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import PEFT for LoRA support
try:
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    PeftModel = None
    TaskType = None

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
        self.lora_config = hf_config.lora_config
        self.lora_adapter_path = hf_config.lora_adapter_path

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
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
            print(f"Using device_map: {self.device_map}")
        else:
            # Will set device manually after loading
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
        
        # Add LoRA adapter if configured
        if self.lora_config is not None or self.lora_adapter_path is not None:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "peft library is required for LoRA adapters. "
                    "Install it with: pip install peft"
                )
            
            if self.lora_adapter_path is not None:
                # Load pre-trained LoRA adapter
                print(f"Loading LoRA adapter from {self.lora_adapter_path}...")
                self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
                print("LoRA adapter loaded successfully!")
            elif self.lora_config is not None:
                # Initialize new LoRA adapter
                print("Initializing LoRA adapter...")
                if isinstance(self.lora_config, str):
                    # Load config from file
                    with open(self.lora_config, 'r') as f:
                        lora_config_dict = json.load(f)
                else:
                    lora_config_dict = self.lora_config
                
                # Create LoraConfig
                lora_config_obj = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    **lora_config_dict
                )
                
                # Apply LoRA to model
                self.model = get_peft_model(self.model, lora_config_obj)
                print("LoRA adapter initialized successfully!")
                # Print trainable parameters
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
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
        # If using device_map, find the device of the embedding layer
        if hasattr(self.model, 'hf_device_map') and self.model.get_input_embeddings() is not None:
            target_device = next(self.model.get_input_embeddings().parameters()).device
        else:
            target_device = self.device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

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
        # If using device_map, find the device of the embedding layer
        if hasattr(self.model, 'hf_device_map') and self.model.get_input_embeddings() is not None:
            target_device = next(self.model.get_input_embeddings().parameters()).device
        else:
            target_device = self.device
        full_inputs = {k: v.to(target_device) for k, v in full_inputs.items()}

        # Tokenize prompts separately to get prompt lengths
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        prompt_inputs = {k: v.to(target_device) for k, v in prompt_inputs.items()}
        prompt_lengths = prompt_inputs["input_ids"].shape[1]

        all_logits = []
        
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

    def get_lora_gradients(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
        reduction: str = "mean"
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """Compute token-wise gradients w.r.t. LoRA adapter parameters.
        
        Args:
            prompts: Single prompt or list of prompts
            responses: Single response or list of responses
            reduction: How to reduce gradients across batch dimension.
                       Options: "mean", "sum", "none" (returns per-sample gradients)
        
        Returns:
            If single prompt/response:
                Dictionary mapping parameter names to gradient arrays
            If multiple prompts/responses:
                List of dictionaries (one per sample) if reduction="none",
                or single dictionary with averaged gradients otherwise
                
        Each gradient array has shape matching the corresponding parameter tensor.
        For token-wise gradients, the gradients are computed for each token position
        in the response sequence.
        """
        if not self._is_loaded:
            self.load()
        
        # Check if LoRA adapter is present
        has_lora = False
        if PEFT_AVAILABLE:
            if isinstance(self.model, PeftModel):
                has_lora = True
            elif hasattr(self.model, 'peft_config'):
                has_lora = True
            else:
                # Check if any parameters have 'lora' in their name
                for name, _ in self.model.named_parameters():
                    if 'lora' in name.lower():
                        has_lora = True
                        break
        
        if not has_lora:
            raise ValueError(
                "No LoRA adapter found. Initialize LoRA adapter first using "
                "lora_config or lora_adapter_path in backend configuration."
            )
        
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
            responses = [responses]
        
        if len(prompts) != len(responses):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match "
                f"number of responses ({len(responses)})"
            )
        
        # Concatenate prompts and responses
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]
        
        # Tokenize full texts
        full_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Determine target device
        if hasattr(self.model, 'hf_device_map') and self.model.get_input_embeddings() is not None:
            target_device = next(self.model.get_input_embeddings().parameters()).device
        else:
            target_device = self.device
        full_inputs = {k: v.to(target_device) for k, v in full_inputs.items()}
        
        # Tokenize prompts separately to get prompt lengths
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        prompt_inputs = {k: v.to(target_device) for k, v in prompt_inputs.items()}
        
        # Set model to train mode to enable gradients
        was_training = self.model.training
        self.model.train()
        
        # Enable gradients for LoRA parameters only
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        all_gradients = []
        
        try:
            # Forward pass
            outputs = self.model(**full_inputs)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            
            # Get target token IDs (shifted by 1 for next-token prediction)
            input_ids = full_inputs["input_ids"]
            target_ids = input_ids[:, 1:].contiguous()  # Shift targets
            logits = logits[:, :-1, :].contiguous()  # Remove last logit position
            
            # Process each sample separately for token-wise gradients
            for i in range(len(prompts)):
                # Get actual prompt length (excluding padding)
                prompt_len = (prompt_inputs["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()
                if self.tokenizer.pad_token_id is None:
                    prompt_len = prompt_inputs["input_ids"].shape[1]
                
                # Get actual sequence length (excluding padding)
                seq_len = (input_ids[i] != self.tokenizer.pad_token_id).sum().item()
                if self.tokenizer.pad_token_id is None:
                    seq_len = input_ids.shape[1]
                
                # Only compute loss for response tokens (after prompt)
                response_start_idx = prompt_len - 1  # -1 because we shifted targets
                response_end_idx = seq_len - 1
                
                if response_start_idx >= response_end_idx:
                    # Empty response
                    all_gradients.append({})
                    continue
                
                # Extract logits and targets for response tokens only
                response_logits = logits[i, response_start_idx:response_end_idx, :]
                response_targets = target_ids[i, response_start_idx:response_end_idx]
                
                # Compute loss for each token position
                token_gradients = {}
                num_response_tokens = response_logits.shape[0]
                
                for token_idx in range(num_response_tokens):
                    # Zero gradients
                    self.model.zero_grad()
                    
                    # Compute cross-entropy loss for this token
                    token_logits = response_logits[token_idx:token_idx+1, :]  # (1, vocab_size)
                    token_target = response_targets[token_idx:token_idx+1]  # (1,)
                    
                    loss = torch.nn.functional.cross_entropy(
                        token_logits,
                        token_target,
                        reduction='mean'
                    )
                    
                    # Backward pass
                    loss.backward(retain_graph=True)
                    
                    # Collect gradients for LoRA parameters
                    token_grads = {}
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and 'lora' in name.lower():
                            token_grads[name] = param.grad.detach().clone().cpu().numpy()
                    
                    if token_idx == 0:
                        # Initialize structure
                        token_gradients = {name: [] for name in token_grads.keys()}
                    
                    # Append gradients for this token
                    for name, grad in token_grads.items():
                        token_gradients[name].append(grad)
                
                # Stack gradients: convert list of arrays to single array
                # Shape: (num_tokens, *param_shape) for each parameter
                stacked_gradients = {}
                for name, grad_list in token_gradients.items():
                    stacked_gradients[name] = np.stack(grad_list, axis=0)
                
                all_gradients.append(stacked_gradients)
        
        finally:
            # Restore model state
            self.model.train(was_training)
            # Zero gradients
            self.model.zero_grad()
        
        # Handle reduction
        if reduction == "none":
            if is_single:
                return all_gradients[0]
            return all_gradients
        elif reduction == "mean":
            # Average gradients across batch
            if len(all_gradients) == 0:
                return {}
            
            averaged_grads = {}
            for name in all_gradients[0].keys():
                stacked = np.stack([grads[name] for grads in all_gradients], axis=0)
                averaged_grads[name] = np.mean(stacked, axis=0)
            
            return averaged_grads
        elif reduction == "sum":
            # Sum gradients across batch
            if len(all_gradients) == 0:
                return {}
            
            summed_grads = {}
            for name in all_gradients[0].keys():
                stacked = np.stack([grads[name] for grads in all_gradients], axis=0)
                summed_grads[name] = np.sum(stacked, axis=0)
            
            return summed_grads
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


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
    config = BackendConfig.from_dict({
        "backend_type": "huggingface",
        "model_name": "Qwen/Qwen2.5-0.5B",  # Replace with your model
        "backend_specific": {
            "device": "cuda" if torch.cuda.is_available() else "cpu"  # Use "cuda" if GPU is available
        }
    })
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
    
    # Example 7: Initialize with LoRA adapter and compute gradients
    print("=" * 60)
    print("Example 7: Initialize with LoRA adapter and compute gradients")
    print("=" * 60)
    
    # Initialize backend with LoRA configuration
    from rl_onoff.backends.config import HuggingFaceBackendConfig
    lora_backend_config = BackendConfig.from_dict({
        "backend_type": "huggingface",
        "model_name": "Qwen/Qwen2.5-0.5B",  # Replace with your model
        "backend_specific": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "lora_config": {
                "r": 1,  # LoRA rank
                "lora_alpha": 1,  # LoRA alpha scaling factor
                # Apply LoRA to all linear layers in attention and MLP
                # For Qwen models, these are the typical module names:
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                    "gate_proj", "up_proj", "down_proj"      # MLP layers
                ],
                "lora_dropout": 0.1,  # LoRA dropout rate
            }
        }
    })
    lora_backend = create_backend(lora_backend_config)
    lora_backend.load()
    
    # Calculate and print LoRA parameter statistics
    print("\n" + "-" * 60)
    print("LoRA Parameter Statistics:")
    print("-" * 60)
    
    # Count trainable (LoRA) parameters
    trainable_params = sum(p.numel() for p in lora_backend.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_backend.model.parameters())
    non_trainable_params = total_params - trainable_params
    
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable (LoRA) parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.4f}%")
    
    # Count LoRA parameters by module type
    lora_params_by_module = {}
    for name, param in lora_backend.model.named_parameters():
        if param.requires_grad and 'lora' in name.lower():
            # Extract module type from parameter name (e.g., "base_model.model.layers.0.self_attn.q_proj.lora_A")
            parts = name.split('.')
            # Find the module name (q_proj, v_proj, etc.)
            module_name = None
            for part in reversed(parts):
                if part in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                    module_name = part
                    break
            
            if module_name:
                if module_name not in lora_params_by_module:
                    lora_params_by_module[module_name] = 0
                lora_params_by_module[module_name] += param.numel()
    
    if lora_params_by_module:
        print("\nLoRA parameters by module type:")
        for module_name, count in sorted(lora_params_by_module.items()):
            print(f"  {module_name}: {count:,} parameters")
    
    # Define prompts and responses for gradient computation
    prompts = [
        "What is 2+2?",
        "What is 3*3?"
    ]
    responses = [
        " 4",
        " 9"
    ]
    
    print(f"\nComputing gradients for {len(prompts)} prompt-response pairs...")
    gradients = lora_backend.get_lora_gradients(
        prompts=prompts,
        responses=responses,
        reduction="mean"  # Average gradients across batch
    )
    
    print(f"\nGradient computation complete!")
    print(f"Number of LoRA parameters with gradients: {len(gradients)}")
    
    # Print gradient information for each parameter
    for param_name, grad_array in gradients.items():
        print(f"\nParameter: {param_name}")
        print(f"  Gradient shape: {grad_array.shape}")
        print(f"  Number of token positions: {grad_array.shape[0]}")
        print(f"  Parameter shape: {grad_array.shape[1:]}")
        print(f"  Gradient norm (per token): {np.linalg.norm(grad_array, axis=tuple(range(1, len(grad_array.shape))))}")
        print(f"  Mean gradient magnitude: {np.abs(grad_array).mean():.6f}")
    
    # Example with single prompt/response
    print("\n" + "-" * 60)
    print("Single prompt/response example:")
    print("-" * 60)
    
    single_gradients = lora_backend.get_lora_gradients(
        prompts="The answer is",
        responses=" 42.",
        reduction="none"
    )
    
    print(f"Number of LoRA parameters: {len(single_gradients)}")
    print(f"Example parameter: {list(single_gradients.keys())[0] if single_gradients else 'None'}")
    if single_gradients:
        first_param = list(single_gradients.keys())[0]
        print(f"  Gradient shape: {single_gradients[first_param].shape}")
        print(f"  Number of response tokens: {single_gradients[first_param].shape[0]}\n")


