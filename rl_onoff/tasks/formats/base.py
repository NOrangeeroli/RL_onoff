"""Base format interface for response formats."""

from abc import ABC, abstractmethod
from typing import Optional, Dict


class BaseFormat(ABC):
    """Abstract base class for response formats.
    
    Each format defines:
    - A system prompt that explains the expected response format
    - An extractor method to parse responses following that format
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize format.
        
        Args:
            name: Format name (defaults to class name)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt that explains the response format.
        
        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def extract(self, response: str) -> Dict[str, Optional[str]]:
        """Extract information from a response following this format.
        
        Args:
            response: Model response text
            
        Returns:
            Dictionary with extracted information. Must include:
            - "answer": The final answer part of the response (required)
            Optional fields:
            - "reasoning": The reasoning/process part of the response (optional)
            Additional fields can be added for extensibility.
            Values can be None if not found or not applicable.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

