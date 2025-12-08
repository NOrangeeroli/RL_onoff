"""Response format classes with system prompts and extractors."""

from rl_onoff.tasks.formats.base import BaseFormat
from rl_onoff.tasks.formats.boxed import BoxedFormat
from rl_onoff.tasks.formats.structured import StructuredFormat

__all__ = [
    "BaseFormat",
    "BoxedFormat",
    "StructuredFormat",
]

