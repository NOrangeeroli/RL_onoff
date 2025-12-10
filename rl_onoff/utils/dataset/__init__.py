"""Dataset classes for different data formats."""

from typing import Optional
from pathlib import Path

from rl_onoff.utils.dataset.base import BaseDataset
from rl_onoff.utils.dataset.aime2025 import AIME2025Dataset
from rl_onoff.utils.dataset.amc23 import AMC23Dataset
from rl_onoff.utils.dataset.gsm8k_level1 import GSM8KLevel1Dataset
from rl_onoff.utils.dataset.math import MathDataset

# Registry mapping dataset names to their classes
DATASET_REGISTRY = {
    "aime2025": AIME2025Dataset,
    "amc23": AMC23Dataset,
    "gsm8k_level1": GSM8KLevel1Dataset,
    "math": MathDataset,
}


def create_dataset(
    name: str,
    split: str = "test",
    data_dir: Optional[Path] = None
) -> BaseDataset:
    """Create a dataset instance from a dataset name.
    
    Args:
        name: Dataset name (must be in DATASET_REGISTRY)
        split: Dataset split ("train" or "test")
        data_dir: Root directory containing data folder (default: project root)
    
    Returns:
        Dataset instance
    
    Raises:
        ValueError: If dataset name is not in the registry
    
    Examples:
        >>> # Create GSM8K dataset
        >>> dataset = create_dataset("gsm8k_level1", split="test")
        >>> 
        >>> # Create MATH dataset with custom data directory
        >>> dataset = create_dataset("math", split="train", data_dir=Path("/path/to/data"))
    """
    name_lower = name.lower()
    
    if name_lower not in DATASET_REGISTRY:
        available_datasets = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset name: '{name}'. "
            f"Available datasets: {available_datasets}"
        )
    
    dataset_class = DATASET_REGISTRY[name_lower]
    
    if data_dir is not None:
        return dataset_class(data_dir=data_dir, split=split)
    else:
        return dataset_class(split=split)


__all__ = [
    "BaseDataset",
    "AIME2025Dataset",
    "AMC23Dataset",
    "GSM8KLevel1Dataset",
    "MathDataset",
    "DATASET_REGISTRY",
    "create_dataset",
]

