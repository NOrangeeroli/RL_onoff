"""HuggingFace Transformers backend implementation."""

from typing import List, Dict, Optional, Union, Any
import os
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
        self.num_process = hf_config.num_process  # Number of replicas from config
        self.model_replicas = None  # Will store multiple replicas for data parallelism
        self.replica_gpu_mappings = None  # Store which GPUs each replica uses

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

        # Check if we should set up replicas (hybrid parallelism)
        if self.num_process is not None and self.num_process > 1:
            # Set up replicas with hybrid parallelism
            self._setup_replicas(self.num_process)
            return  # _setup_replicas handles model loading
        
        # Prepare model loading kwargs for single model
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

    def _setup_replicas(self, num_process: int) -> None:
        """Set up model replicas using CUDA_VISIBLE_DEVICES approach.
        
        This is the standard way: each replica sees only its allocated GPUs,
        and device_map="auto" handles tensor parallelism within those GPUs.
        
        Args:
            num_process: Number of replicas (data parallelism)
        """
        if self.model_replicas is not None and len(self.model_replicas) == num_process:
            # Already set up correctly
            return
        
        # Load tokenizer first if not already loaded
        if self.tokenizer is None:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise ValueError("CUDA not available. Cannot set up multi-GPU replicas.")
        
        if num_process > num_gpus:
            raise ValueError(
                f"num_process ({num_process}) cannot exceed number of GPUs ({num_gpus})"
            )
        
        gpus_per_replica = num_gpus // num_process
        if gpus_per_replica < 1:
            raise ValueError(
                f"Not enough GPUs ({num_gpus}) for {num_process} replicas. "
                f"Need at least {num_process} GPUs."
            )
        
        print(f"Setting up {num_process} model replicas with {gpus_per_replica} GPUs each")
        print(f"Total GPUs: {num_gpus}, GPUs per replica: {gpus_per_replica}")
        
        # Store original CUDA_VISIBLE_DEVICES
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        
        self.model_replicas = []
        self.replica_gpu_mappings = []
        
        for replica_id in range(num_process):
            start_gpu = replica_id * gpus_per_replica
            end_gpu = start_gpu + gpus_per_replica
            replica_gpus = list(range(start_gpu, end_gpu))
            self.replica_gpu_mappings.append(replica_gpus)
            
            # Set CUDA_VISIBLE_DEVICES to only the GPUs this replica should use
            # This makes those GPUs appear as cuda:0, cuda:1, etc. to the model
            cuda_visible_str = ",".join(str(gpu) for gpu in replica_gpus)
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_str
            
            print(f"Loading model replica {replica_id} on GPUs {replica_gpus} (visible as cuda:0-{gpus_per_replica-1})")
            
            # Now load model with device_map="auto" - it will automatically
            # distribute across the visible GPUs (tensor parallelism)
            model_kwargs = {"device_map": "auto"}
            if self.torch_dtype is not None:
                model_kwargs["torch_dtype"] = self.torch_dtype
            
            replica_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            replica_model.eval()
            self.model_replicas.append(replica_model)
        
        # Restore original CUDA_VISIBLE_DEVICES
        if original_cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
        else:
            # Remove it if it wasn't set originally
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        
        # Use first replica as primary model for compatibility
        self.model = self.model_replicas[0]
        self.num_process = num_process
        self._is_loaded = True
        print(f"Successfully set up {num_process} replicas")
        print(f"Replica GPU mappings: {self.replica_gpu_mappings}")

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
        compute_gradients: bool = False,
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """Generate text from prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            stop_strings: List of strings that will stop generation when encountered
            return_logits: If True, return logits along with generated text
            return_probs: If True, return probabilities along with generated text
            compute_gradients: If True, enable gradient computation (for gradient-based analysis).
                              Note: Gradient computation may require different device configurations
                              and uses more memory. If using device_map="auto", gradients may not
                              be fully supported - consider using single GPU mode for gradient computation.
        """
        if not self._is_loaded:
            self.load()

        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]

        # Use replicas if available (set up during load() if num_process is in config)
        if self.model_replicas and len(self.model_replicas) > 1:
            return self._generate_with_replicas(
                prompts, max_length, temperature, top_k, top_p, do_sample,
                stop_strings, return_logits, return_probs, is_single, compute_gradients
            )
        else:
            # Single model path
            return self._generate_single(
                prompts, max_length, temperature, top_k, top_p, do_sample,
                stop_strings, return_logits, return_probs, is_single, compute_gradients
            )

    def _generate_single(
        self,
        prompts: List[str],
        max_length: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
        stop_strings: Optional[List[str]],
        return_logits: bool,
        return_probs: bool,
        is_single: bool,
        compute_gradients: bool,
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """Generate using single model."""
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
        # Conditionally enable/disable gradients
        if compute_gradients:
            context_manager = torch.enable_grad()
        else:
            context_manager = torch.no_grad()
        
        with context_manager:
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

    def _generate_with_replicas(
        self,
        prompts: List[str],
        max_length: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
        stop_strings: Optional[List[str]],
        return_logits: bool,
        return_probs: bool,
        is_single: bool,
        compute_gradients: bool,
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """Generate using multiple replicas (data parallelism)."""
        batch_size = len(prompts)
        num_replicas = len(self.model_replicas)
        replica_batch_size = (batch_size + num_replicas - 1) // num_replicas
        
        all_generated_ids = []
        all_scores = []
        
        # Conditionally enable/disable gradients
        if compute_gradients:
            context_manager = torch.enable_grad()
        else:
            context_manager = torch.no_grad()
        
        for replica_id, replica_model in enumerate(self.model_replicas):
            start_idx = replica_id * replica_batch_size
            end_idx = min(start_idx + replica_batch_size, batch_size)
            
            if start_idx >= batch_size:
                break
            
            replica_prompts = prompts[start_idx:end_idx]
            
            # Tokenize for this replica
            replica_inputs = self.tokenizer(
                replica_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            # Prepare generation kwargs
            gen_kwargs = {
                "max_length": max_length,
                "do_sample": do_sample,
            }
            
            if stop_strings is not None:
                gen_kwargs["stop_strings"] = stop_strings
                gen_kwargs["tokenizer"] = self.tokenizer
            
            if return_logits or return_probs:
                gen_kwargs["return_dict_in_generate"] = True
                gen_kwargs["output_scores"] = True
            
            if do_sample:
                gen_kwargs["temperature"] = temperature
                if top_k is not None:
                    gen_kwargs["top_k"] = top_k
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p
            
            # Generate on this replica
            with context_manager:
                if return_logits or return_probs:
                    outputs = replica_model.generate(**replica_inputs, **gen_kwargs)
                    all_generated_ids.append(outputs.sequences.cpu())
                    if outputs.scores:
                        all_scores.append([s.cpu() for s in outputs.scores])
                else:
                    outputs = replica_model.generate(**replica_inputs, **gen_kwargs)
                    all_generated_ids.append(outputs.cpu())
        
        # Concatenate results from all replicas
        generated_ids = torch.cat(all_generated_ids, dim=0)
        
        # Decode all at once
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # If not returning logits/probs, return texts
        if not return_logits and not return_probs:
            return texts[0] if is_single else texts
        
        # Prepare results with logits/probs
        # Need to properly handle scores from multiple replicas
        results = []
        text_idx = 0
        for replica_id in range(num_replicas):
            start_idx = replica_id * replica_batch_size
            end_idx = min(start_idx + replica_batch_size, batch_size)
            if start_idx >= batch_size:
                break
            
            replica_size = end_idx - start_idx
            for i in range(replica_size):
                result = {"text": texts[text_idx]}
                text_idx += 1
                
                if return_logits or return_probs:
                    # Get logits for this sample
                    if replica_id < len(all_scores) and all_scores[replica_id]:
                        replica_scores = all_scores[replica_id]
                        batch_logits = torch.stack(replica_scores, dim=1)  # (replica_batch_size, num_tokens, vocab_size)
                        logits_array = batch_logits[i].cpu().numpy()
                        
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
        compute_gradients: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get token logits for predicting response tokens given prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            responses: Single response or list of responses
            compute_gradients: If True, enable gradient computation (for gradient-based analysis).
                              Note: Gradient computation may require different device configurations
                              and uses more memory. If using device_map="auto", gradients may not
                              be fully supported - consider using single GPU mode for gradient computation.
        """
        if not self._is_loaded:
            self.load()

        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
            responses = [responses]
        
        if len(prompts) != len(responses):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of responses ({len(responses)})")

        # Use replicas if available (set up during load() if num_process is in config)
        if self.model_replicas and len(self.model_replicas) > 1:
            return self._get_logits_with_replicas(prompts, responses, is_single, compute_gradients)
        else:
            # Single model path
            return self._get_logits_single(prompts, responses, is_single, compute_gradients)

    def _get_logits_single(
        self,
        prompts: List[str],
        responses: List[str],
        is_single: bool,
        compute_gradients: bool,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get logits using single model."""
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
        
        # Conditionally enable/disable gradients
        if compute_gradients:
            context_manager = torch.enable_grad()
        else:
            context_manager = torch.no_grad()
        
        with context_manager:
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

    def _get_logits_with_replicas(
        self,
        prompts: List[str],
        responses: List[str],
        is_single: bool,
        compute_gradients: bool,
    ) -> List[np.ndarray]:
        """Get logits using multiple replicas (data parallelism)."""
        batch_size = len(prompts)
        num_replicas = len(self.model_replicas)
        replica_batch_size = (batch_size + num_replicas - 1) // num_replicas
        
        all_logits = []
        
        # Conditionally enable/disable gradients
        if compute_gradients:
            context_manager = torch.enable_grad()
        else:
            context_manager = torch.no_grad()
        
        for replica_id, replica_model in enumerate(self.model_replicas):
            start_idx = replica_id * replica_batch_size
            end_idx = min(start_idx + replica_batch_size, batch_size)
            
            if start_idx >= batch_size:
                break
            
            replica_prompts = prompts[start_idx:end_idx]
            replica_responses = responses[start_idx:end_idx]
            
            # Get logits for this replica's batch
            replica_logits = self._get_logits_for_batch(
                replica_model, replica_prompts, replica_responses, compute_gradients, context_manager
            )
            all_logits.extend(replica_logits)
        
        if is_single:
            return all_logits[0]
        return all_logits

    def _get_logits_for_batch(
        self,
        model,
        prompts: List[str],
        responses: List[str],
        compute_gradients: bool,
        context_manager,
    ) -> List[np.ndarray]:
        """Get logits for a batch using a specific model replica."""
        # Concatenate prompts and responses
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]
        
        # Tokenize full texts
        full_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Tokenize prompts separately
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        prompt_lengths = prompt_inputs["input_ids"].shape[1]
        
        # Get logits (device_map handles device placement automatically)
        with context_manager:
            outputs = model(**full_inputs)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            
            batch_logits = []
            for i in range(len(prompts)):
                prompt_len = (prompt_inputs["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()
                if self.tokenizer.pad_token_id is None:
                    prompt_len = prompt_lengths
                
                seq_len = (full_inputs["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()
                if self.tokenizer.pad_token_id is None:
                    seq_len = full_inputs["input_ids"].shape[1]
                
                response_logits = logits[i, prompt_len-1:seq_len-1, :]
                batch_logits.append(response_logits.cpu().numpy())
        
        return batch_logits

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


