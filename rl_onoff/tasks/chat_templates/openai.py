"""OpenAI-style chat template."""

from typing import List, Dict, Optional

from rl_onoff.tasks.chat_templates.base import BaseChatTemplate


class OpenAIChatTemplate(BaseChatTemplate):
    """OpenAI-style chat template.
    
    Format: System message, then alternating user/assistant messages.
    Uses role labels like "system", "user", "assistant".
    """

    def __init__(self):
        """Initialize OpenAI chat template."""
        super().__init__("openai")

    def format(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        **kwargs
    ) -> str:
        """Format messages in OpenAI style.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            add_generation_prompt: If True, add "assistant:" prompt at the end
            **kwargs: Additional formatting options
            
        Returns:
            Formatted prompt string
        """
        formatted_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        if add_generation_prompt:
            formatted_parts.append("Assistant:")
        
        return "\n\n".join(formatted_parts)

