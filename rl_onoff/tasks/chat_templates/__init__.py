"""Chat templates for formatting conversations."""

from rl_onoff.tasks.chat_templates.base import BaseChatTemplate
from rl_onoff.tasks.chat_templates.openai import OpenAIChatTemplate
from rl_onoff.tasks.chat_templates.llama import LlamaChatTemplate
from rl_onoff.tasks.chat_templates.chatml import ChatMLTemplate
from rl_onoff.tasks.chat_templates.simple import SimpleChatTemplate

# Registry mapping string names to chat template classes
CHAT_TEMPLATE_REGISTRY = {
    "openai": OpenAIChatTemplate,
    "llama": LlamaChatTemplate,
    "chatml": ChatMLTemplate,
    "simple": SimpleChatTemplate,
}


def create_chat_template(name: str, **kwargs) -> BaseChatTemplate:
    """Create a chat template instance from a name.
    
    Args:
        name: Name of the chat template ("openai", "llama", "chatml", "simple")
        **kwargs: Additional arguments for chat template creation
        
    Returns:
        Chat template instance
        
    Raises:
        ValueError: If name is not recognized
    """
    if name not in CHAT_TEMPLATE_REGISTRY:
        raise ValueError(
            f"Unknown chat template name: {name}. "
            f"Available: {list(CHAT_TEMPLATE_REGISTRY.keys())}"
        )
    
    return CHAT_TEMPLATE_REGISTRY[name](**kwargs)


__all__ = [
    "BaseChatTemplate",
    "OpenAIChatTemplate",
    "LlamaChatTemplate",
    "ChatMLTemplate",
    "SimpleChatTemplate",
    "CHAT_TEMPLATE_REGISTRY",
    "create_chat_template",
]

