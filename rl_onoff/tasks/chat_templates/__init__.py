"""Chat templates for formatting conversations."""

from rl_onoff.tasks.chat_templates.base import BaseChatTemplate
from rl_onoff.tasks.chat_templates.openai import OpenAIChatTemplate
from rl_onoff.tasks.chat_templates.llama import LlamaChatTemplate
from rl_onoff.tasks.chat_templates.chatml import ChatMLTemplate
from rl_onoff.tasks.chat_templates.simple import SimpleChatTemplate

__all__ = [
    "BaseChatTemplate",
    "OpenAIChatTemplate",
    "LlamaChatTemplate",
    "ChatMLTemplate",
    "SimpleChatTemplate",
]

