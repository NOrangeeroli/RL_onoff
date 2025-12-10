"""Response format classes with system prompts and extractors."""

from rl_onoff.tasks.formats.base import BaseFormat
from rl_onoff.tasks.formats.boxed import BoxedFormat
from rl_onoff.tasks.formats.structured import StructuredFormat

# Registry mapping string names to format classes
FORMAT_REGISTRY = {
    "boxed": BoxedFormat,
    "structured": StructuredFormat,
}


def create_format(config: dict) -> BaseFormat:
    """Create a format instance from a config dict.
    
    Args:
        config: Dictionary with 'name' key
                Example: {"name": "boxed"}
                Note: Formats don't take any initialization parameters
        
    Returns:
        Format instance
        
    Raises:
        ValueError: If name is not recognized or config is invalid
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {type(config)}")
    
    name = config.get("name")
    if name is None:
        raise ValueError("config must have 'name' key")
    
    if name not in FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown format name: {name}. "
            f"Available: {list(FORMAT_REGISTRY.keys())}"
        )
    
    # Formats don't take any initialization parameters
    return FORMAT_REGISTRY[name]()


__all__ = [
    "BaseFormat",
    "BoxedFormat",
    "StructuredFormat",
    "FORMAT_REGISTRY",
    "create_format",
]

