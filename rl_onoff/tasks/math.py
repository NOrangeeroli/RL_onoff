"""Math task with prompt template and math verification reward."""

from typing import Union
from pathlib import Path

from rl_onoff.tasks.base import BaseTask
from rl_onoff.tasks.config import TaskConfig


class MathTask(BaseTask):
    """Math problem-solving task.
    
    Configured via a config file that specifies:
    - Template type for formatting questions
    - Reward type (typically "math_verify")
    - Format type (typically "boxed" or "structured")
    """

    def __init__(self, config: Union[str, Path, TaskConfig, dict] = None):
        """Initialize math task from config.
        
        Args:
            config: Config file path, TaskConfig instance, dict, or None for default config
                   If None, loads from configs/math_default.json
        """
        if config is None:
            # Load default config from configs/math_default.json
            config_path = Path(__file__).parent / "configs" / "math_default.json"
            config = str(config_path)
        
        super().__init__(config)


if __name__ == "__main__":
    # Simple usage example for MathTask
    print("MathTask Usage Example")
    print("=" * 50)
    
    # Initialize MathTask with default config (from configs/math_default.json)
    task = MathTask()
    
    # Format a question into a prompt
    question = "What is 2 + 2?"
    prompt = task.format_query(question)
    print(f"\nQuestion: {question}")
    print(f"\nFormatted Prompt:\n{prompt}")
    
    # Simulate a model response (in practice, you'd use a backend to generate this)
    response = """Let me solve this step by step.
    
2 + 2 = 4

Therefore, the answer is 4.

\\boxed{4}"""
    
    print(f"\nModel Response:\n{response}")
    
    # Extract the answer from the response
    extracted = task.extract_answer(response)
    print(f"\nExtracted Answer: {extracted.get('answer')}")
    print(f"Extracted Reasoning: {extracted.get('reasoning', 'N/A')[:50]}...")
    
    # Evaluate the answer against a reference
    # Note: evaluate() extracts answers internally, so pass the full response
    reference = "4"
    score = task.evaluate(response, reference)
    print(f"\nReference Answer: {reference}")
    print(f"Score: {score} (1.0 = correct, 0.0 = incorrect)")
    
    # Example with multiple questions
    print("\n" + "=" * 50)
    print("Batch Example:")
    questions = ["What is 3 + 3?", "What is 5 * 2?"]
    references = ["6", "10"]
    
    for q, ref in zip(questions, references):
        prompt = task.format_query(q)
        print(f"\nQuestion: {q}")
        print(f"Reference: {ref}")
        # In practice, you would generate responses here
        # For demo, we'll just show the prompt
        print(f"Prompt length: {len(prompt)} characters")

