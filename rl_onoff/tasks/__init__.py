"""Task framework for task-specific prompt templates and rewards."""

from rl_onoff.tasks.base import BaseTask
from rl_onoff.tasks.math import MathTask
from rl_onoff.tasks.formats import (
    BaseFormat,
    BoxedFormat,
    StructuredFormat,
)
from rl_onoff.tasks.chat_templates import (
    BaseChatTemplate,
    OpenAIChatTemplate,
    LlamaChatTemplate,
    ChatMLTemplate,
    SimpleChatTemplate,
)

__all__ = [
    "BaseTask",
    "MathTask",
    "BaseFormat",
    "BoxedFormat",
    "StructuredFormat",
    "BaseChatTemplate",
    "OpenAIChatTemplate",
    "LlamaChatTemplate",
    "ChatMLTemplate",
    "SimpleChatTemplate",
]

