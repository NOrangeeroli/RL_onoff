"""Llama-style chat template with special tokens."""

from typing import List, Dict, Optional

from rl_onoff.tasks.chat_templates.base import BaseChatTemplate


class LlamaChatTemplate(BaseChatTemplate):
    """Llama-style chat template.
    
    Uses special tokens: <|begin_of_text|>, <|start_header_id|>, <|end_header_id|>, <|eot_id|>
    Format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>...
    """

    def __init__(self):
        """Initialize Llama chat template."""
        super().__init__("llama")

    def format(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages in Llama style.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            add_generation_prompt: If True, add assistant prompt at the end
            
        Returns:
            Formatted prompt string with Llama tokens
        """
        parts = []
        has_system = False
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
                has_system = True
            elif role == "user":
                if not has_system:
                    # Add begin_of_text if no system message
                    parts.append("<|begin_of_text|>")
                parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "assistant":
                parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "assistant_generation" and content:
                add_generation_prompt = False
                parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}")
        
        if add_generation_prompt:
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        return "".join(parts)

