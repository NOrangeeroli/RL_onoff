"""Chat templates for formatting conversations."""

from rl_onoff.tasks.chat_templates.base import BaseChatTemplate
from rl_onoff.tasks.chat_templates.simple import SimpleChatTemplate
from rl_onoff.tasks.chat_templates.openai import OpenAIChatTemplate
from rl_onoff.tasks.chat_templates.llama import LlamaChatTemplate
from rl_onoff.tasks.chat_templates.chatml import ChatMLTemplate

# Registry mapping string names to chat template classes
CHAT_TEMPLATE_REGISTRY = {
    "simple": SimpleChatTemplate,
    "openai": OpenAIChatTemplate,
    "llama": LlamaChatTemplate,
    "chatml": ChatMLTemplate,
}


def create_chat_template(config: dict) -> BaseChatTemplate:
    """Create a chat template instance from a config dict.
    
    Args:
        config: Dictionary with 'name' key
                Example: {"name": "openai"}
                Note: Chat templates don't take any initialization parameters
        
    Returns:
        Chat template instance
        
    Raises:
        ValueError: If name is not recognized or config is invalid
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {type(config)}")
    
    name = config.get("name")
    if name is None:
        raise ValueError("config must have 'name' key")
    
    if name not in CHAT_TEMPLATE_REGISTRY:
        raise ValueError(
            f"Unknown chat template name: {name}. "
            f"Available: {list(CHAT_TEMPLATE_REGISTRY.keys())}"
        )
    
    # Chat templates don't take any initialization parameters
    return CHAT_TEMPLATE_REGISTRY[name]()


__all__ = [
    "BaseChatTemplate",
    "SimpleChatTemplate",
    "OpenAIChatTemplate",
    "LlamaChatTemplate",
    "ChatMLTemplate",
    "CHAT_TEMPLATE_REGISTRY",
    "create_chat_template",
]

