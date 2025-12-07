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

