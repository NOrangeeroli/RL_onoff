"""Simple chat template without special formatting."""

from typing import List, Dict, Optional

from rl_onoff.tasks.chat_templates.base import BaseChatTemplate


class SimpleChatTemplate(BaseChatTemplate):
    """Simple chat template without special tokens.
    
    Just concatenates messages with simple separators.
    Useful for models that don't need special formatting.
    """

    def __init__(self, separator: str = "\n\n"):
        """Initialize simple chat template.
        
        Args:
            separator: String to separate messages (default: "\n\n")
        """
        super().__init__("simple")
        self.separator = separator

    def format(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        **kwargs
    ) -> str:
        """Format messages simply by concatenating.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            add_generation_prompt: If True, add empty line at the end
            **kwargs: Additional formatting options
            
        Returns:
            Concatenated messages separated by separator
        """
        parts = []
        
        for msg in messages:
            content = msg.get("content", "")
            if content:
                parts.append(content)
        
        result = self.separator.join(parts)
        
        if add_generation_prompt:
            result += self.separator
        
        return result

