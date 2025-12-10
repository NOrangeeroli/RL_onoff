"""ChatML-style template with XML-like tags."""

from typing import List, Dict, Optional

from rl_onoff.tasks.chat_templates.base import BaseChatTemplate


class ChatMLTemplate(BaseChatTemplate):
    """ChatML-style chat template.
    
    Uses XML-like tags: <|im_start|>role and <|im_end|>
    Format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>
    """

    def __init__(self):
        """Initialize ChatML template."""
        super().__init__("chatml")

    def format(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages in ChatML style.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            add_generation_prompt: If True, add assistant prompt at the end
            
        Returns:
            Formatted prompt string with ChatML tags
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "assistant_generation" and content:
                parts.append(f"<|im_start|>assistant\n{content}")
                add_generation_prompt = False
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        
        return "\n".join(parts)

