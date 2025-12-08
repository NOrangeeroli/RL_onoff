"""Base chat template interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union


class BaseChatTemplate(ABC):
    """Abstract base class for chat templates.
    
    Chat templates format conversations with roles (system, user, assistant)
    into a single prompt string that models can process.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize chat template.
        
        Args:
            name: Template name (defaults to class name)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def format(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        **kwargs
    ) -> str:
        """Format messages into a single prompt string.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles are typically 'system', 'user', or 'assistant'.
            add_generation_prompt: If True, add prompt tokens for assistant response
            **kwargs: Additional formatting options
            
        Returns:
            Formatted prompt string
        """
        pass

    def format_simple(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        assistant_message: Optional[str] = None,
        add_generation_prompt: bool = False,
        **kwargs
    ) -> str:
        """Format a simple conversation (system, user, optional assistant).
        
        Args:
            user_message: User's message/question
            system_message: Optional system message/instruction
            assistant_message: Optional assistant response (for few-shot or context)
            add_generation_prompt: If True, add prompt tokens for assistant response
            **kwargs: Additional formatting options
            
        Returns:
            Formatted prompt string
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        if assistant_message:
            messages.append({"role": "assistant", "content": assistant_message})
        
        return self.format(messages, add_generation_prompt=add_generation_prompt, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

