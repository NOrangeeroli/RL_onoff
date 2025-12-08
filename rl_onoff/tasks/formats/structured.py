"""Structured format for responses with <think>...</think><answer>...</answer> tags."""

import re
from typing import Optional, Dict

from rl_onoff.tasks.formats.base import BaseFormat


class StructuredFormat(BaseFormat):
    """Structured format for responses with explicit reasoning and answer tags.
    
    System prompt instructs the model to use <think>...</think> and <answer>...</answer> tags.
    Extractor parses responses to extract content from these tags.
    """

    def __init__(self):
        """Initialize structured format."""
        super().__init__("structured")

    def get_system_prompt(self) -> str:
        """Get the system prompt for structured format.
        
        Returns:
            System prompt string
        """
        return """You are a helpful math tutor. Solve the problem step by step and format your response using the following structure:
<think>
[Your step-by-step reasoning and solution process goes here]
</think>
<answer>
[The final numerical or symbolic answer goes here]
</answer>
Make sure to include both tags with your reasoning and answer."""

    def extract(self, response: str) -> Dict[str, Optional[str]]:
        """Extract reasoning and answer from a response with structured format.
        
        Args:
            response: Model response text
            
        Returns:
            Dictionary with:
            - "answer": Content inside <answer>...</answer> tags (required)
            - "reasoning": Content inside <think>...</think> tags (optional)
        """
        response = response.strip()
        
        # Extract from <think>...</think><answer>...</answer>
        reasoning_match = re.search(
            r'<think>(.*?)</think>',
            response,
            re.DOTALL | re.IGNORECASE
        )
        answer_match = re.search(
            r'<answer>(.*?)</answer>',
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None
        answer = answer_match.group(1).strip() if answer_match else None
        
        return {"reasoning": reasoning, "answer": answer}

