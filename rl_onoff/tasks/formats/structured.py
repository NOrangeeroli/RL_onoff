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
        return """You are a helpful assistant. Before solving any problem, you must start by reflecting on your metacognitive strategy in the <think> section. In <think>, do not solve the problem yet. Instead, analyze what type of problem it is and what concept it tests, what key information and constraints are given, what general class of problems it belongs to and why, what reasoning method is most appropriate (e.g., algebraic, combinatorial, case analysis), and what common pitfalls to avoid. Finally, summarize a general, reusable approach or solution template that can be applied to this type of problem. After that, in the <answer> section, solve the problem step-by-step, clearly showing all work, and conclude with your final answer in \\boxed{}."""

    def apply_user_prompt_template(self, question: str) -> str:
        """Apply the user prompt template to a question.
        
        Args:
            question: The question/problem string
            
        Returns:
            The formatted question string
        """
        return f"""{question}\nYour output must strictly follow this format:\n<think> high-level metacognitive reasoning, problem type classification, and generalized solving strategy here </think>\n<answer> detailed step-by-step solution here, ending with \\boxed{{final result}} </answer>"""


    def get_assistant_prompt(self) -> str:
        """Get the assistant prompt for structured format.
        
        Returns:
            Assistant prompt string
        """
        return """<think>"""
    
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
        # Match the last occurrence of each tag
        reasoning_matches = list(re.finditer(
            r'<think>(.*?)</think>',
            response,
            re.DOTALL | re.IGNORECASE
        ))
        answer_matches = list(re.finditer(
            r'<answer>(.*?)</answer>',
            response,
            re.DOTALL | re.IGNORECASE
        ))
        
        reasoning = reasoning_matches[-1].group(1).strip() if reasoning_matches else None
        answer = answer_matches[-1].group(1).strip() if answer_matches else None
        
        return {"reasoning": reasoning, "answer": answer}


if __name__ == "__main__":
    # Simple usage example for StructuredFormat
    print("StructuredFormat Usage Example")
    print("=" * 50)
    
    # Create format instance
    format = StructuredFormat()
    
    # 1. Get system prompt
    print("\n1. System Prompt:")
    system_prompt = format.get_system_prompt()
    print(system_prompt[:80] + "...")
    
    # 2. Format a question into messages
    print("\n2. Format Message:")
    question = "What is 3 * 3?"
    messages = format.format_message(question)
    print(f"Question: {question}")
    print(f"Messages: {len(messages)} message(s)")
    for msg in messages:
        print(f"  - {msg['role']}: {msg['content'][:50]}...")
    
    # 3. Extract from response
    print("\n3. Extract Answer:")
    response = """<think>
Let me solve this step by step.
3 * 3 = 9
</think>
<answer>
9
</answer>"""
    print(f"Response:\n{response}")
    result = format.extract(response)
    print(f"\nExtracted:")
    print(f"  Answer: {result.get('answer')}")
    print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
    
    # Example with multiple answer tags (should use last one)
    print("\n" + "=" * 50)
    print("Example with multiple <answer> tags (uses last occurrence):")
    response2 = """<think>First attempt...</think>
<answer>8</answer>
<think>Actually, let me recalculate...</think>
<answer>9</answer>"""
    result2 = format.extract(response2)
    print(f"Response: {response2[:100]}...")
    print(f"Extracted Answer: {result2.get('answer')}")  # Should be "9"

