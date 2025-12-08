"""Task framework for task-specific prompt templates and metrics."""

from rl_onoff.tasks.base import BaseTask
from rl_onoff.tasks.math import MathTask
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
    "BaseChatTemplate",
    "OpenAIChatTemplate",
    "LlamaChatTemplate",
    "ChatMLTemplate",
    "SimpleChatTemplate",
]

