"""Sampling configuration management."""

from typing import Optional, List
from dataclasses import dataclass

from rl_onoff.utils.config import Config


@dataclass
class SamplingConfig(Config):
    """Configuration for sampling parameters.
    
    This extends the existing SamplingConfig dataclass with Config functionality
    for JSON serialization/deserialization.
    """
    
    max_length: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = True
    num_samples: int = 1
    batch_size: Optional[int] = None
    seed: Optional[int] = None
    stop_strings: Optional[List[str]] = None

