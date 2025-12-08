"""Chat templates for formatting conversations."""

from rl_onoff.tasks.chat_templates.base import BaseChatTemplate
from rl_onoff.tasks.chat_templates.openai import OpenAIChatTemplate
from rl_onoff.tasks.chat_templates.llama import LlamaChatTemplate
from rl_onoff.tasks.chat_templates.chatml import ChatMLTemplate
from rl_onoff.tasks.chat_templates.simple import SimpleChatTemplate
from rl_onoff.tasks.chat_templates.system_prompts import (
    get_system_prompt,
    SYSTEM_PROMPTS,
    MATH_SYSTEM_PROMPTS,
    CODING_SYSTEM_PROMPTS,
    QA_SYSTEM_PROMPTS,
    GENERAL_SYSTEM_PROMPTS,
)

__all__ = [
    "BaseChatTemplate",
    "OpenAIChatTemplate",
    "LlamaChatTemplate",
    "ChatMLTemplate",
    "SimpleChatTemplate",
    "get_system_prompt",
    "SYSTEM_PROMPTS",
    "MATH_SYSTEM_PROMPTS",
    "CODING_SYSTEM_PROMPTS",
    "QA_SYSTEM_PROMPTS",
    "GENERAL_SYSTEM_PROMPTS",
]

