"""Sampling utilities for LLM generation."""

from typing import List, Dict, Optional, Union, Any

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Dummy tqdm if not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

from rl_onoff.backends.base import BaseBackend
from rl_onoff.sampling.config import SamplingConfig


class Sampler:
    """Sampler for generating text from models."""

    def __init__(self, backend: BaseBackend):
        """Initialize sampler with a backend.
        
        Args:
            backend: Model backend instance
        """
        self.backend = backend
        if not backend.is_loaded():
            backend.load()


    def sample(
        self,
        prompts: List[str],
        config: Optional[SamplingConfig] = None
    ) -> List[Union[str, List[str]]]:
        """Sample text from multiple prompts in batches.
        
        Args:
            prompts: List of prompts
            config: Sampling configuration (includes batch_size and num_samples)
            
        Returns:
            List of generated texts (or list of lists if num_samples > 1)
        """
        if config is None:
            config = SamplingConfig()

        # Prepare generation kwargs
        gen_kwargs = {
            "max_length": config.max_length,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "do_sample": config.do_sample,
        }
        if config.stop_strings is not None:
            gen_kwargs["stop_strings"] = config.stop_strings

        # Step 1: Duplicate prompts num_samples times
        # If num_samples=3 and prompts=[p1, p2], we get [p1, p1, p1, p2, p2, p2]
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * config.num_samples)
        
        # Step 2: Batchify the expanded prompts list
        batch_size = config.batch_size
        all_generated = []
        
        if batch_size is None:
            # Process all expanded prompts at once
            all_generated = self.backend.generate(expanded_prompts, **gen_kwargs)
        else:
            # Process in batches with progress bar
            num_batches = (len(expanded_prompts) + batch_size - 1) // batch_size
            batch_range = range(0, len(expanded_prompts), batch_size)
            
            if TQDM_AVAILABLE:
                batch_range = tqdm(batch_range, desc="Generating", total=num_batches, unit="batch")
            
            for i in batch_range:
                batch_prompts = expanded_prompts[i:i + batch_size]
                batch_results = self.backend.generate(batch_prompts, **gen_kwargs)
                all_generated.extend(batch_results)
        
        # Step 3: Group results by original prompt
        # Convert flat list back to grouped structure
        # all_generated = [s1_1, s1_2, s1_3, s2_1, s2_2, s2_3, ...]
        # Should become [[s1_1, s1_2, s1_3], [s2_1, s2_2, s2_3], ...]
        grouped_results = []
        for i in range(len(prompts)):
            start_idx = i * config.num_samples
            end_idx = start_idx + config.num_samples
            prompt_samples = all_generated[start_idx:end_idx]
            
            if config.num_samples == 1:
                # Return single string instead of list
                grouped_results.append(prompt_samples[0])
            else:
                # Return list of samples
                grouped_results.append(prompt_samples)
        
        return grouped_results


if __name__ == "__main__":
    """Simple use cases for Sampler."""
    
    try:
        from rl_onoff.backends import HuggingFaceBackend
        
        print("Running Sampler examples...")
        print("=" * 60)
        
        # Initialize backend (using a small model for demonstration)
        from rl_onoff.backends.config import BackendConfig
        from rl_onoff.backends import create_backend
        model_name = "meta-llama/Llama-3.2-1B"  # Replace with your preferred model
        config = BackendConfig(backend_type="huggingface", model_name=model_name)
        backend = create_backend(config)
        
        # Create sampler
        sampler = Sampler(backend)
        print(f"Sampler initialized with backend: {model_name}\n")
        
        # Example 1: Basic single prompt sampling
        print("=" * 60)
        print("Example 1: Basic single prompt sampling")
        print("=" * 60)
        prompt = "The future of artificial intelligence is"
        result = sampler.sample(prompt, config=SamplingConfig(max_length=20))
        print(f"Prompt: {prompt}")
        print(f"Generated: {result}\n")
        
        # Example 2: Multiple prompts
        print("=" * 60)
        print("Example 2: Multiple prompts")
        print("=" * 60)
        prompts = [
            "Python is a programming language that",
            "Machine learning is"
        ]
        results = sampler.sample(prompts, config=SamplingConfig(max_length=15))
        for prompt, result in zip(prompts, results):
            print(f"Prompt: {prompt}")
            print(f"Generated: {result}\n")
        
        # Example 3: Using SamplingConfig with custom parameters
        print("=" * 60)
        print("Example 3: Custom SamplingConfig (temperature, top_k, top_p)")
        print("=" * 60)
        config = SamplingConfig(
            max_length=20,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        result = sampler.sample(["The best way to learn programming is"], config=config)
        print(f"Prompt: The best way to learn programming is")
        print(f"Generated (temp=0.7, top_k=50, top_p=0.9): {result[0]}\n")
        
        # Example 4: Multiple samples per prompt
        print("=" * 60)
        print("Example 4: Multiple samples per prompt")
        print("=" * 60)
        config = SamplingConfig(max_length=15, num_samples=3)
        results = sampler.sample(["Once upon a time"], config=config)
        print(f"Prompt: Once upon a time")
        print(f"Generated {config.num_samples} samples:")
        for i, sample in enumerate(results[0], 1):
            print(f"  Sample {i}: {sample}")
        print()
        
        # Example 5: Deterministic sampling (greedy decoding)
        print("=" * 60)
        print("Example 5: Deterministic sampling (do_sample=False)")
        print("=" * 60)
        config = SamplingConfig(max_length=15, do_sample=False)
        result = sampler.sample(["The answer to life, the universe, and everything is"], config=config)
        print(f"Prompt: The answer to life, the universe, and everything is")
        print(f"Generated (deterministic): {result[0]}\n")
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure HuggingFace backend is available.")
    except Exception as e:
        print(f"\nAn error occurred during Sampler examples: {e}")
        print("Please ensure you have a compatible model and sufficient resources.")

