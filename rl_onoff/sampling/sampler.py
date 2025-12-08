"""Sampling utilities for LLM generation."""

from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass

from rl_onoff.backends.base import BaseBackend


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = True
    num_samples: int = 1
    seed: Optional[int] = None


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
        config: Optional[SamplingConfig] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[Union[str, List[str]]]:
        """Sample text from multiple prompts in batches.
        
        Args:
            prompts: List of prompts
            config: Sampling configuration
            batch_size: Batch size for processing (None for all at once)
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated texts (or list of lists if num_samples > 1)
        """
        if config is None:
            config = SamplingConfig()

        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "do_sample": config.do_sample,
            **kwargs
        }

        all_results = []

        if batch_size is None:
            # Process all prompts at once in a single batch
            if config.num_samples == 1:
                # Single sample per prompt: batch all prompts together
                results = self.backend.generate(prompts, **gen_kwargs)
                all_results = results
            else:
                # Multiple samples per prompt: batch all prompts for each sample iteration
                for sample_idx in range(config.num_samples):
                    batch_results = self.backend.generate(prompts, **gen_kwargs)
                    if sample_idx == 0:
                        # Initialize with lists for each prompt
                        all_results = [[result] for result in batch_results]
                    else:
                        # Append to existing lists
                        for i, result in enumerate(batch_results):
                            all_results[i].append(result)
        else:
            # Process in batches
            if config.num_samples == 1:
                # Single sample per prompt: process in batches
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i + batch_size]
                    batch_results = self.backend.generate(batch_prompts, **gen_kwargs)
                    all_results.extend(batch_results)
            else:
                # Multiple samples per prompt: for each sample iteration, process in batches
                for sample_idx in range(config.num_samples):
                    for i in range(0, len(prompts), batch_size):
                        batch_prompts = prompts[i:i + batch_size]
                        batch_results = self.backend.generate(batch_prompts, **gen_kwargs)
                        if sample_idx == 0:
                            # Initialize for first sample
                            all_results.extend([[result] for result in batch_results])
                        else:
                            # Append to existing lists
                            start_idx = i
                            for j, result in enumerate(batch_results):
                                all_results[start_idx + j].append(result)
        
        return all_results


if __name__ == "__main__":
    """Simple use cases for Sampler."""
    
    try:
        from rl_onoff.backends import HuggingFaceBackend
        
        print("Running Sampler examples...")
        print("=" * 60)
        
        # Initialize backend (using a small model for demonstration)
        model_name = "meta-llama/Llama-3.2-1B"  # Replace with your preferred model
        backend = HuggingFaceBackend(model_name=model_name)
        
        # Create sampler
        sampler = Sampler(backend)
        print(f"Sampler initialized with backend: {model_name}\n")
        
        # Example 1: Basic single prompt sampling
        print("=" * 60)
        print("Example 1: Basic single prompt sampling")
        print("=" * 60)
        prompt = "The future of artificial intelligence is"
        result = sampler.sample(prompt, config=SamplingConfig(max_new_tokens=20))
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
        results = sampler.sample(prompts, config=SamplingConfig(max_new_tokens=15))
        for prompt, result in zip(prompts, results):
            print(f"Prompt: {prompt}")
            print(f"Generated: {result}\n")
        
        # Example 3: Using SamplingConfig with custom parameters
        print("=" * 60)
        print("Example 3: Custom SamplingConfig (temperature, top_k, top_p)")
        print("=" * 60)
        config = SamplingConfig(
            max_new_tokens=20,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        result = sampler.sample("The best way to learn programming is", config=config)
        print(f"Prompt: The best way to learn programming is")
        print(f"Generated (temp=0.7, top_k=50, top_p=0.9): {result}\n")
        
        # Example 4: Multiple samples per prompt
        print("=" * 60)
        print("Example 4: Multiple samples per prompt")
        print("=" * 60)
        config = SamplingConfig(max_new_tokens=15, num_samples=3)
        results = sampler.sample("Once upon a time", config=config)
        print(f"Prompt: Once upon a time")
        print(f"Generated {config.num_samples} samples:")
        for i, sample in enumerate(results, 1):
            print(f"  Sample {i}: {sample}")
        print()
        
        # Example 5: Deterministic sampling (greedy decoding)
        print("=" * 60)
        print("Example 5: Deterministic sampling (do_sample=False)")
        print("=" * 60)
        config = SamplingConfig(max_new_tokens=15, do_sample=False)
        result = sampler.sample("The answer to life, the universe, and everything is", config=config)
        print(f"Prompt: The answer to life, the universe, and everything is")
        print(f"Generated (deterministic): {result}\n")
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure HuggingFace backend is available.")
    except Exception as e:
        print(f"\nAn error occurred during Sampler examples: {e}")
        print("Please ensure you have a compatible model and sufficient resources.")

