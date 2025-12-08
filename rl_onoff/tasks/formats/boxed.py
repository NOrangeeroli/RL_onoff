"""Boxed format for math responses with \\boxed{answer}."""

import re
from typing import Optional, Dict

from rl_onoff.tasks.formats.base import BaseFormat


class BoxedFormat(BaseFormat):
    """Boxed format for math responses.
    
    System prompt instructs the model to provide the final answer in \\boxed{answer} format.
    Extractor parses responses to find the boxed answer and extract reasoning.
    """

    def __init__(self):
        """Initialize boxed format."""
        super().__init__("boxed")

    def get_system_prompt(self) -> str:
        """Get the system prompt for boxed format.
        
        Returns:
            System prompt string
        """
        return """You are a helpful math tutor. Solve the problem step by step and provide the final answer in a boxed format. At the end of your solution, include the final answer as \\boxed{{answer}}."""

    def extract(self, response: str) -> Dict[str, Optional[str]]:
        """Extract reasoning and answer from a response with boxed format.
        
        Args:
            response: Model response text
            
        Returns:
            Dictionary with:
            - "answer": The content inside \\boxed{...} (required)
            - "reasoning": Everything before the boxed answer (optional)
        """
        response = response.strip()
        
        # Extract from \boxed{answer} format
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, response)
        
        if match:
            answer = match.group(1).strip()
            # Everything before the boxed answer is reasoning
            reasoning = response[:match.start()].strip()
            return {"reasoning": reasoning, "answer": answer}
        else:
            # Fallback: try to extract answer using common patterns
            answer_patterns = [
                r'(?:the\s+)?answer\s+is\s*:?\s*([^\n\.]+)',
                r'answer\s*:?\s*([^\n\.]+)',
                r'final\s+answer\s*:?\s*([^\n\.]+)',
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip().rstrip('.,;!?')
                    reasoning = response[:match.start()].strip()
                    return {"reasoning": reasoning, "answer": answer}
            
            # If no answer found, return full response as reasoning
            return {"reasoning": response, "answer": None}

