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
        return """You are a helpful math tutor. Solve the problem step by step and provide the final answer in a boxed format. At the end of your solution, include the final answer as \\boxed{answer}."""

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
        
        # Extract from \boxed{answer} format - match the last occurrence
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        matches = list(re.finditer(boxed_pattern, response))
        
        if matches:
            # Use the last match
            match = matches[-1]
            answer = match.group(1).strip()
            # Everything before the last boxed answer is reasoning
            reasoning = response[:match.start()].strip()
            return {"reasoning": reasoning, "answer": answer}
        else:
            
               
            return {"reasoning": response, "answer": None}


if __name__ == "__main__":
    # Simple usage example for BoxedFormat
    print("BoxedFormat Usage Example")
    print("=" * 50)
    
    # Create format instance
    format = BoxedFormat()
    
    # 1. Get system prompt
    print("\n1. System Prompt:")
    system_prompt = format.get_system_prompt()
    print(system_prompt[:80] + "...")
    
    # 2. Format a question into messages
    print("\n2. Format Message:")
    question = "What is 2 + 2?"
    messages = format.format_message(question)
    print(f"Question: {question}")
    print(f"Messages: {len(messages)} message(s)")
    for msg in messages:
        print(f"  - {msg['role']}: {msg['content'][:50]}...")
    
    # 3. Extract from response
    print("\n3. Extract Answer:")
    response = """Let me solve this step by step.

2 + 2 = 4

Therefore, the answer is 4.

\\boxed{4}"""
    print(f"Response:\n{response}")
    result = format.extract(response)
    print(f"\nExtracted:")
    print(f"  Answer: {result.get('answer')}")
    print(f"  Reasoning: {result.get('reasoning', 'N/A')[:50]}...")
    
    # Example with multiple boxed answers (should use last one)
    print("\n" + "=" * 50)
    print("Example with multiple \\boxed{} (uses last occurrence):")
    response2 = "First attempt: \\boxed{3}. Actually, let me recalculate... \\boxed{4}"
    result2 = format.extract(response2)
    print(f"Response: {response2}")
    print(f"Extracted Answer: {result2.get('answer')}")  # Should be "4"

