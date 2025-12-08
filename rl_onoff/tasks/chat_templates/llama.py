"""Llama-style chat template with special tokens."""

from typing import List, Dict, Optional

from rl_onoff.tasks.chat_templates.base import BaseChatTemplate


class LlamaChatTemplate(BaseChatTemplate):
    """Llama-style chat template.
    
    Uses special tokens: <s>, </s>, [INST], [/INST]
    Format: <s>[INST] user_message [/INST] assistant_message </s>
    """

    def __init__(self):
        """Initialize Llama chat template."""
        super().__init__("llama")

    def format(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        **kwargs
    ) -> str:
        """Format messages in Llama style.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            add_generation_prompt: If True, add [INST] prompt at the end
            **kwargs: Additional formatting options
            
        Returns:
            Formatted prompt string with Llama tokens
        """
        parts = []
        
        # Start with <s> token
        parts.append("<s>")
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # System message at the beginning
                if i == 0:
                    parts.append(f"[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
                else:
                    parts.append(f"[INST] {content} [/INST]")
            elif role == "user":
                if i == 0 and not any(m.get("role") == "system" for m in messages):
                    parts.append(f"[INST] {content} [/INST]")
                else:
                    parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(f"{content} </s>")
        
        if add_generation_prompt:
            # If last message is not assistant, add prompt for assistant response
            if not messages or messages[-1].get("role") != "assistant":
                parts.append("<s> [INST]")
        
        return " ".join(parts)

