"""Base format interface for response formats."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List


class BaseFormat(ABC):
    """Abstract base class for response formats.
    
    Each format defines:
    - A system prompt that explains the expected response format
    - An extractor method to parse responses following that format
    - A method to format a question into messages for chat templates
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
    
    @abstractmethod
    def apply_user_prompt_template(self, question: str) -> str:
        """Apply the user prompt template to a question.
        
        Args:
            question: The question/problem string
            
        Returns:
            The formatted question string
        """
        pass
        
    @abstractmethod
    def get_assistant_prompt(self) -> str:
        """Get the assistant prompt template.
        
        Returns:
            The assistant prompt template string
        """
        pass
    
        
    def format_message(self, question: str) -> List[Dict[str, str]]:
        """Format a question into a list of message dicts for chat templates.
        
        Args:
            question: The question/problem string
            
        Returns:
            List of message dicts with 'role' and 'content' keys:
            - First message: {'role': 'system', 'content': system_prompt}
            - Second message: {'role': 'user', 'content': question}
        """
        messages = []
        system_prompt = self.get_system_prompt()
        assistant_prompt = self.get_assistant_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.apply_user_prompt_template(question)})
        if assistant_prompt:
            messages.append({"role": "assistant_generation", "content": assistant_prompt})
        return messages

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

