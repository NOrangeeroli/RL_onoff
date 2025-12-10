"""HuggingFace Transformers backend implementation."""

from typing import List, Dict, Optional, Union, Any
import os
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import accelerate for hybrid parallelism
try:
    from accelerate import Accelerator
    from accelerate.utils import ParallelismConfig
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None
    ParallelismConfig = None

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
        self.num_process = hf_config.num_process  # Number of replicas from config (data parallelism)
        self.accelerator = None  # Accelerate accelerator instance for parallelism
        self.use_accelerate = False  # Whether to use Accelerate for parallelism

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

        # Check if we need parallelism (multi-GPU or multi-process)
        num_gpus = torch.cuda.device_count()
        needs_parallelism = (
            (self.num_process is not None and self.num_process > 1) or
            (num_gpus > 1 and (self.device_map == "auto" or self.device_map is None))
        )
        
        if needs_parallelism:
            if not ACCELERATE_AVAILABLE:
                raise ImportError(
                    "accelerate is required for multi-GPU/multi-process parallelism. "
                    "Install it with: pip install accelerate"
                )
            
            # Calculate parallelism parameters
            if self.num_process is not None and self.num_process > 1:
                # Multi-process case (data parallel or hybrid)
                if num_gpus == 0:
                    raise ValueError("CUDA not available. Cannot set up multi-process parallelism.")
                if self.num_process > num_gpus:
                    raise ValueError(
                        f"num_process ({self.num_process}) cannot exceed number of GPUs ({num_gpus})"
                    )
                gpus_per_replica = num_gpus // self.num_process
                if gpus_per_replica < 1:
                    raise ValueError(
                        f"Not enough GPUs ({num_gpus}) for {self.num_process} replicas. "
                        f"Need at least {self.num_process} GPUs."
                    )
                dp_shard_size = self.num_process
                tp_size = gpus_per_replica
            else:
                # Pure model parallel (single process, multiple GPUs)
                dp_shard_size = 1
                tp_size = num_gpus
            
            self._setup_with_accelerate(dp_shard_size, tp_size)
            return
        
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

    def _setup_with_accelerate(self, dp_shard_size: int, tp_size: int) -> None:
        """Set up model using HuggingFace Accelerate with ParallelismConfig.
        
        This uses Accelerate for all parallelism cases:
        - Pure data parallel: dp_shard_size > 1, tp_size = 1
        - Pure model parallel: dp_shard_size = 1, tp_size > 1
        - Hybrid: dp_shard_size > 1, tp_size > 1
        
        Args:
            dp_shard_size: Number of data parallel shards
            tp_size: Number of GPUs per shard (tensor parallelism)
        """
        if not ACCELERATE_AVAILABLE:
            raise ImportError(
                "accelerate is required for parallelism. "
                "Install it with: pip install accelerate"
            )
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise ValueError("CUDA not available. Cannot set up multi-GPU parallelism.")
        
        total_gpus_needed = dp_shard_size * tp_size
        if total_gpus_needed > num_gpus:
            raise ValueError(
                f"Not enough GPUs ({num_gpus}) for requested parallelism "
                f"(dp_shard_size={dp_shard_size}, tp_size={tp_size}, total={total_gpus_needed})"
            )
        
        # Determine parallelism type for logging
        if dp_shard_size > 1 and tp_size > 1:
            parallelism_type = "hybrid (data + tensor)"
        elif dp_shard_size > 1:
            parallelism_type = "data parallel"
        elif tp_size > 1:
            parallelism_type = "model parallel"
        else:
            parallelism_type = "single GPU"
        
        print(f"Setting up {parallelism_type} using Accelerate")
        print(f"  Data parallel shards: {dp_shard_size}")
        print(f"  Tensor parallel size: {tp_size}")
        print(f"  Total GPUs: {num_gpus}")
        
        # Manual distributed initialization if not already initialized
        # When using accelerate launch, distributed should be initialized, but ParallelismConfig
        # might need it initialized before creating the Accelerator
        if not dist.is_initialized():
            print("Distributed not initialized, setting up manually...")
            # Check if environment variables are set (e.g., by accelerate launch or torchrun)
            rank = int(os.environ.get('RANK', '0'))
            world_size = int(os.environ.get('WORLD_SIZE', str(dp_shard_size)))
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            
            # Set up missing environment variables
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = '29500'
            
            # Initialize process group
            try:
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank
                )
                print(f"Distributed initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize distributed environment: {e}\n"
                    f"To use Accelerate with ParallelismConfig, either:\n"
                    f"  1. Launch with: accelerate launch --num_processes={dp_shard_size} your_script.py\n"
                    f"  2. Launch with: torchrun --nproc_per_node={dp_shard_size} your_script.py\n"
                    f"  3. Or ensure distributed environment variables (RANK, WORLD_SIZE, LOCAL_RANK) are set correctly"
                )
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            print(f"Distributed already initialized: rank={rank}, world_size={world_size}")
            
            # Verify world_size matches dp_shard_size
            if world_size != dp_shard_size:
                print(f"Warning: WORLD_SIZE ({world_size}) does not match dp_shard_size ({dp_shard_size})")
        
        # Configure parallelism
        parallelism_config = ParallelismConfig(
            dp_shard_size=dp_shard_size,
            tp_size=tp_size,
        )
        
        # Initialize Accelerator with parallelism config
        self.accelerator = Accelerator(parallelism_config=parallelism_config)
        self.use_accelerate = True
        
        print(f"Accelerate initialized with dp_shard_size={dp_shard_size}, tp_size={tp_size}")
        
        # Load model
        print("Loading model with Accelerate...")
        model_kwargs = {}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        
        # Load model - Accelerate will handle distribution
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Prepare model with Accelerator
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        
        # Print device information
        print(f"Model prepared with Accelerate")
        if hasattr(self.accelerator, 'device'):
            print(f"  Accelerator device: {self.accelerator.device}")
        
        self._is_loaded = True
        print("Model loaded successfully with Accelerate!")

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

        # Use Accelerate if configured, otherwise use single model
        if self.use_accelerate and self.accelerator is not None:
            return self._generate_with_accelerate(
                prompts, max_length, temperature, top_k, top_p, do_sample,
                stop_strings, return_logits, return_probs, is_single, compute_gradients
            )
        else:
            # Single model path
            return self._generate_single(
                prompts, max_length, temperature, top_k, top_p, do_sample,
                stop_strings, return_logits, return_probs, is_single, compute_gradients
            )

    def _generate_with_accelerate(
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
        """Generate using Accelerate-prepared model (hybrid parallelism).
        
        This method splits prompts across processes for data parallelism,
        processes them in parallel, and gathers results from all processes.
        """
        # Unwrap model to access generate() method
        model = self.accelerator.unwrap_model(self.model)
        
        # Split prompts across processes for data parallelism
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        
        total_prompts = len(prompts)
        prompts_per_process = total_prompts // world_size
        remainder = total_prompts % world_size
        
        # Calculate this process's slice (distribute remainder across first few processes)
        start_idx = rank * prompts_per_process + min(rank, remainder)
        end_idx = start_idx + prompts_per_process + (1 if rank < remainder else 0)
        local_prompts = prompts[start_idx:end_idx]
        
        if len(local_prompts) == 0:
            # This process has no prompts to process
            local_results = []
        else:
            # Tokenize inputs for this process's prompts
            inputs = self.tokenizer(
                local_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            # Move inputs to accelerator device
            inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
            
            # Enable gradients on inputs if needed
            if compute_gradients:
                for k, v in inputs.items():
                    if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        v.requires_grad_(True)
            
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
            
            # Generate
            if compute_gradients:
                context_manager = torch.enable_grad()
            else:
                context_manager = torch.no_grad()
            
            with context_manager:
                if return_logits or return_probs:
                    outputs = model.generate(**inputs, **gen_kwargs)
                    generated_ids = outputs.sequences
                    scores = outputs.scores
                else:
                    outputs = model.generate(**inputs, **gen_kwargs)
                    generated_ids = outputs
                    scores = None
            
            # Decode outputs (remove input tokens)
            input_lengths = inputs["input_ids"].shape[1]
            generated_token_ids = generated_ids[:, input_lengths:]
            texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
            
            # Extract gradients if needed
            gradients_list = []
            if compute_gradients:
                # Note: To actually compute gradients, you need a loss function
                # This is a placeholder structure - gradients would be computed
                # via loss.backward() after computing a loss
                for k, v in inputs.items():
                    if v.grad is not None:
                        gradients_list.append({k: v.grad.cpu().numpy()})
            
            # Prepare results for this process
            if not return_logits and not return_probs:
                local_results = texts
            else:
                local_results = []
                for i, text in enumerate(texts):
                    result = {"text": text}
                    
                    if return_logits or return_probs:
                        if scores:
                            batch_logits = torch.stack(scores, dim=1)  # (batch_size, num_tokens, vocab_size)
                            logits_array = batch_logits[i].cpu().numpy()  # (num_tokens, vocab_size)
                            
                            if return_logits:
                                result["logits"] = logits_array
                            
                            if return_probs:
                                scaled_logits = logits_array / temperature
                                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
                                probs_array = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                                result["probs"] = probs_array
                    
                    if compute_gradients and gradients_list:
                        result["gradients"] = gradients_list[i] if i < len(gradients_list) else None
                    
                    local_results.append(result)
        
        # Gather results from all processes
        if world_size > 1:
            # Use gather_object for Python objects (lists, dicts, strings)
            # gather_object is a torch.distributed function (PyTorch 1.8+)
            if hasattr(dist, 'gather_object'):
                gathered_list = [None] * world_size
                dist.gather_object(local_results, gathered_list, dst=0)
                if rank == 0:
                    # Flatten the list of lists
                    gathered_results = []
                    for proc_results in gathered_list:
                        if proc_results is not None:
                            gathered_results.extend(proc_results)
                else:
                    gathered_results = None
            else:
                # Fallback for older PyTorch versions: serialize and use all_gather
                import pickle
                # Serialize local results
                serialized = pickle.dumps(local_results)
                serialized_tensor = torch.ByteTensor(list(serialized))
                # Gather serialized data
                gathered_tensors = [torch.zeros_like(serialized_tensor) for _ in range(world_size)]
                dist.all_gather(gathered_tensors, serialized_tensor)
                if rank == 0:
                    # Deserialize and flatten
                    gathered_results = []
                    for tensor in gathered_tensors:
                        data = pickle.loads(bytes(tensor.cpu().numpy().tolist()))
                        if data is not None:
                            gathered_results.extend(data)
                else:
                    gathered_results = None
            
            # Only return results on main process
            if self.accelerator.is_main_process:
                final_results = gathered_results
            else:
                # Non-main processes return None
                return None if not is_single else None
        else:
            # Single process - no gathering needed
            final_results = local_results
        
        # Return in the same format as input
        if is_single:
            return final_results[0] if final_results else None
        return final_results

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

        # Use Accelerate if configured, otherwise use single model
        if self.use_accelerate and self.accelerator is not None:
            return self._get_logits_with_accelerate(prompts, responses, is_single, compute_gradients)
        else:
            # Single model path
            return self._get_logits_single(prompts, responses, is_single, compute_gradients)

    def _get_logits_with_accelerate(
        self,
        prompts: List[str],
        responses: List[str],
        is_single: bool,
        compute_gradients: bool,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get logits using Accelerate-prepared model (hybrid parallelism).
        
        This method splits prompts/responses across processes for data parallelism,
        processes them in parallel, and gathers results from all processes.
        """
        # Unwrap model to access forward() method
        model = self.accelerator.unwrap_model(self.model)
        
        # Split prompts/responses across processes for data parallelism
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        
        total_pairs = len(prompts)
        pairs_per_process = total_pairs // world_size
        remainder = total_pairs % world_size
        
        # Calculate this process's slice
        start_idx = rank * pairs_per_process + min(rank, remainder)
        end_idx = start_idx + pairs_per_process + (1 if rank < remainder else 0)
        local_prompts = prompts[start_idx:end_idx]
        local_responses = responses[start_idx:end_idx]
        
        if len(local_prompts) == 0:
            # This process has no work to do
            local_logits = []
        else:
            # Concatenate prompts and responses for this process
            full_texts = [prompt + response for prompt, response in zip(local_prompts, local_responses)]
            
            # Tokenize full texts
            full_inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            # Tokenize prompts separately
            prompt_inputs = self.tokenizer(
                local_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            prompt_lengths = prompt_inputs["input_ids"].shape[1]
            
            # Move inputs to accelerator device
            full_inputs = {k: v.to(self.accelerator.device) for k, v in full_inputs.items()}
            prompt_inputs = {k: v.to(self.accelerator.device) for k, v in prompt_inputs.items()}
            
            # Enable gradients on inputs if needed
            if compute_gradients:
                for k, v in full_inputs.items():
                    if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        v.requires_grad_(True)
            
            local_logits = []
            
            # Conditionally enable/disable gradients
            if compute_gradients:
                context_manager = torch.enable_grad()
            else:
                context_manager = torch.no_grad()
            
            with context_manager:
                # Get logits for the full sequence
                outputs = model(**full_inputs)
                logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                
                # Extract logits for solution tokens only
                for i in range(len(local_prompts)):
                    prompt_len = (prompt_inputs["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()
                    if self.tokenizer.pad_token_id is None:
                        prompt_len = prompt_lengths
                    
                    seq_len = (full_inputs["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()
                    if self.tokenizer.pad_token_id is None:
                        seq_len = full_inputs["input_ids"].shape[1]
                    
                    response_logits = logits[i, prompt_len-1:seq_len-1, :]
                    local_logits.append(response_logits.cpu().numpy())
        
        # Gather results from all processes
        if world_size > 1:
            # Use gather_object for Python objects (lists of numpy arrays)
            # gather_object is a torch.distributed function (PyTorch 1.8+)
            if hasattr(dist, 'gather_object'):
                gathered_list = [None] * world_size
                dist.gather_object(local_logits, gathered_list, dst=0)
                if rank == 0:
                    # Flatten the list of lists
                    gathered_logits = []
                    for proc_logits in gathered_list:
                        if proc_logits is not None:
                            gathered_logits.extend(proc_logits)
                else:
                    gathered_logits = None
            else:
                # Fallback for older PyTorch versions: serialize and use all_gather
                import pickle
                # Serialize local logits
                serialized = pickle.dumps(local_logits)
                serialized_tensor = torch.ByteTensor(list(serialized))
                # Gather serialized data
                gathered_tensors = [torch.zeros_like(serialized_tensor) for _ in range(world_size)]
                dist.all_gather(gathered_tensors, serialized_tensor)
                if rank == 0:
                    # Deserialize and flatten
                    gathered_logits = []
                    for tensor in gathered_tensors:
                        data = pickle.loads(bytes(tensor.cpu().numpy().tolist()))
                        if data is not None:
                            gathered_logits.extend(data)
                else:
                    gathered_logits = None
            
            # Only return results on main process
            if self.accelerator.is_main_process:
                all_logits = gathered_logits
            else:
                # Non-main processes return None
                return None if not is_single else None
        else:
            # Single process - no gathering needed
            all_logits = local_logits
        
        if is_single:
            return all_logits[0] if all_logits else None
        return all_logits

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


