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
    ) -> str:
        """Format messages into a single prompt string.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles are typically 'system', 'user', or 'assistant'.
            add_generation_prompt: If True, add prompt tokens for assistant response
            
        Returns:
            Formatted prompt string
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

