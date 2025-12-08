"""System prompts for different tasks that explain output formats."""

import re
from typing import Optional, Tuple, Callable, Dict

# Math task system prompts
MATH_SYSTEM_PROMPTS = {
    "boxed": """You are a helpful math tutor. Solve the problem step by step and provide the final answer in a boxed format. At the end of your solution, include the final answer as \\boxed{{answer}}.""",
    
    "structured": """You are a helpful math tutor. Solve the problem step by step and format your response using the following structure:
<think>
[Your step-by-step reasoning and solution process goes here]
</think>
<answer>
[The final numerical or symbolic answer goes here]
</answer>
Make sure to include both tags with your reasoning and answer.""",
}

# Coding task system prompts
CODING_SYSTEM_PROMPTS = {
    "default": """You are a helpful coding assistant. Write clean, well-commented code to solve the problem. Format your response as:
1. Brief explanation of the approach
2. Complete code solution
3. Brief explanation of how it works""",
    
    "python": """You are a Python programming assistant. Write Python code to solve the problem. Include:
1. Code solution with comments
2. Brief explanation if needed""",
    
    "test": """You are a coding assistant. Write code that solves the problem and includes test cases to verify correctness.""",
}

# Question answering system prompts
QA_SYSTEM_PROMPTS = {
    "default": """You are a helpful assistant. Answer the question clearly and concisely. Provide accurate information based on your knowledge.""",
    
    "detailed": """You are a helpful assistant. Provide a detailed, well-structured answer to the question. Include relevant context and explanations.""",
    
    "concise": """You are a helpful assistant. Provide a concise, direct answer to the question.""",
}

# General task system prompts
GENERAL_SYSTEM_PROMPTS = {
    "default": """You are a helpful assistant. Follow the instructions carefully and provide a clear, accurate response.""",
    
    "step_by_step": """You are a helpful assistant. Break down the task into steps and explain your reasoning clearly.""",
}

# All system prompts organized by task type
SYSTEM_PROMPTS = {
    "math": MATH_SYSTEM_PROMPTS,
    "coding": CODING_SYSTEM_PROMPTS,
    "qa": QA_SYSTEM_PROMPTS,
    "general": GENERAL_SYSTEM_PROMPTS,
}


def get_system_prompt(task_type: str, prompt_style: str = "default") -> str:
    """Get a system prompt for a given task type and style.
    
    Args:
        task_type: Type of task ("math", "coding", "qa", "general")
        prompt_style: Style of prompt ("default", "boxed", "concise", etc.)
        
    Returns:
        System prompt string
        
    Raises:
        KeyError: If task_type or prompt_style not found
    """
    if task_type not in SYSTEM_PROMPTS:
        raise KeyError(
            f"Unknown task type: {task_type}. "
            f"Available: {list(SYSTEM_PROMPTS.keys())}"
        )
    
    task_prompts = SYSTEM_PROMPTS[task_type]
    if prompt_style not in task_prompts:
        raise KeyError(
            f"Unknown prompt style '{prompt_style}' for task '{task_type}'. "
            f"Available: {list(task_prompts.keys())}"
        )
    
    return task_prompts[prompt_style]


def extract_response(task_type: str, prompt_style: str, response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract reasoning and answer from a response based on the prompt structure.
    
    Args:
        task_type: Type of task ("math", "coding", "qa", "general")
        prompt_style: Style of prompt ("default", "boxed", "concise", "structured", etc.)
        response: Model response text to extract from
        
    Returns:
        Tuple of (reasoning, answer) where:
        - reasoning: The reasoning/process part of the response
        - answer: The final answer part of the response
        Either can be None if not found or not applicable
        
    Raises:
        KeyError: If task_type or prompt_style not found, or no extractor available
    """
    if task_type == "math":
        if prompt_style not in MATH_EXTRACTORS:
            raise KeyError(
                f"No extractor available for math prompt style '{prompt_style}'. "
                f"Available: {list(MATH_EXTRACTORS.keys())}"
            )
        return MATH_EXTRACTORS[prompt_style](response)
    else:
        # For other task types, return the full response as reasoning and None as answer
        # Can be extended later with specific extractors
        return response, None

