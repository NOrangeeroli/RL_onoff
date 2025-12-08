"""Response format classes with system prompts and extractors."""

from rl_onoff.tasks.formats.base import BaseFormat
from rl_onoff.tasks.formats.boxed import BoxedFormat
from rl_onoff.tasks.formats.structured import StructuredFormat

# Registry mapping string names to format classes
FORMAT_REGISTRY = {
    "boxed": BoxedFormat,
    "structured": StructuredFormat,
}


def create_format(name: str, **kwargs) -> BaseFormat:
    """Create a format instance from a name.
    
    Args:
        name: Name of the format ("boxed", "structured")
        **kwargs: Additional arguments for format creation
        
    Returns:
        Format instance
        
    Raises:
        ValueError: If name is not recognized
    """
    if name not in FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown format name: {name}. "
            f"Available: {list(FORMAT_REGISTRY.keys())}"
        )
    
    return FORMAT_REGISTRY[name](**kwargs)


__all__ = [
    "BaseFormat",
    "BoxedFormat",
    "StructuredFormat",
    "FORMAT_REGISTRY",
    "create_format",
]

