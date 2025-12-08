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
        prompts: Union[str, List[str]],
        config: Optional[SamplingConfig] = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """Sample text from the model.
        
        Args:
            prompts: Single prompt or list of prompts
            config: Sampling configuration
            **kwargs: Additional generation arguments
            
        Returns:
            If num_samples=1: single text or list of texts
            If num_samples>1: list of texts or list of lists of texts
        """
        if config is None:
            config = SamplingConfig()

        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]

        all_samples = []
        
        for prompt in prompts:
            prompt_samples = []
            for _ in range(config.num_samples):
                text = self.backend.generate(
                    prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    **kwargs
                )
                prompt_samples.append(text)
            
            if config.num_samples == 1:
                all_samples.append(prompt_samples[0])
            else:
                all_samples.append(prompt_samples)

        if is_single:
            return all_samples[0]
        return all_samples

    def sample_batch(
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

        if batch_size is None:
            # Process all at once
            return self.sample(prompts, config=config, **kwargs)

        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = self.sample(batch, config=config, **kwargs)
            results.extend(batch_results)
        
        return results


if __name__ == "__main__":
    """Simple use cases for Sampler."""
    
    try:
        from rl_onoff.backends import HuggingFaceBackend
        
        print("Running Sampler examples...")
        print("=" * 60)
        
        # Initialize backend (using a small model for demonstration)
        model_name = "gpt2"  # Replace with your preferred model
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
        
        # Example 5: Batch sampling
        print("=" * 60)
        print("Example 5: Batch sampling with batch_size")
        print("=" * 60)
        prompts = [
            "The capital of France is",
            "The capital of Japan is",
            "The capital of Brazil is",
            "The capital of Australia is"
        ]
        results = sampler.sample_batch(
            prompts,
            config=SamplingConfig(max_new_tokens=10),
            batch_size=2  # Process 2 prompts at a time
        )
        for prompt, result in zip(prompts, results):
            print(f"Prompt: {prompt}")
            print(f"Generated: {result}\n")
        
        # Example 6: Deterministic sampling (greedy decoding)
        print("=" * 60)
        print("Example 6: Deterministic sampling (do_sample=False)")
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

