"""Task framework for task-specific prompt templates and rewards."""

from typing import Union
from pathlib import Path

from rl_onoff.tasks.base import BaseTask
from rl_onoff.tasks.math import MathTask
from rl_onoff.tasks.config import TaskConfig
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


def create_task(config: Union[str, Path, TaskConfig, dict]) -> BaseTask:
    """Create a task instance from a config file, TaskConfig, or dict.
    
    Args:
        config: Config file path (str/Path), TaskConfig instance, or dict
        
    Returns:
        Task instance (BaseTask)
        
    Examples:
        >>> # From config file
        >>> task = create_task("rl_onoff/tasks/configs/math_default.json")
        >>> 
        >>> # From TaskConfig
        >>> config = TaskConfig(template_type="simple", reward_type="math_verify", format_type="boxed")
        >>> task = create_task(config)
        >>> 
        >>> # From dict
        >>> task = create_task({"template_type": "simple", "reward_type": "math_verify", "format_type": "boxed"})
    """
    return BaseTask(config)


__all__ = [
    "BaseTask",
    "MathTask",
    "TaskConfig",
    "create_task",
    "BaseFormat",
    "BoxedFormat",
    "StructuredFormat",
    "BaseChatTemplate",
    "OpenAIChatTemplate",
    "LlamaChatTemplate",
    "ChatMLTemplate",
    "SimpleChatTemplate",
]

