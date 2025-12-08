# Response Formats

The `formats` module provides response format classes that define both the system prompt (which instructs the model on how to format responses) and the extractor logic (which parses responses following that format).

## Overview

Each format class implements:
- **System Prompt**: Explains to the model how to structure its response
- **Extractor**: Parses the model's response to extract reasoning and answer

## Available Formats

### BoxedFormat

For math responses that use `\boxed{answer}` format.

```python
from rl_onoff.tasks.formats import BoxedFormat

format = BoxedFormat()

# Get system prompt
system_prompt = format.get_system_prompt()
# Returns: "You are a helpful math tutor. Solve the problem step by step..."

# Extract from response
response = "Let me solve this step by step... The answer is 42. \\boxed{42}"
result = format.extract(response)
# result: {"reasoning": "Let me solve this step by step... The answer is 42.", "answer": "42"}
reasoning = result["reasoning"]
answer = result["answer"]
```

### StructuredFormat

For responses with explicit `<think>...</think>` and `<answer>...</answer>` tags.

```python
from rl_onoff.tasks.formats import StructuredFormat

format = StructuredFormat()

# Get system prompt
system_prompt = format.get_system_prompt()
# Returns: "You are a helpful math tutor. Solve the problem step by step..."

# Extract from response
response = """<think>
Let me solve this step by step...
</think>
<answer>
42
</answer>"""
result = format.extract(response)
# result: {"reasoning": "Let me solve this step by step...", "answer": "42"}
reasoning = result["reasoning"]
answer = result["answer"]
```

## Usage with Tasks

Tasks use formats to define their response structure:

```python
from rl_onoff.tasks import MathTask
from rl_onoff.tasks.formats import StructuredFormat

# Create task with structured format
task = MathTask(format=StructuredFormat())

# Get system prompt from task
system_prompt = task.get_system_prompt()

# Extract answer from response
result = task.answer_extractor(response)
reasoning = result["reasoning"]
answer = result["answer"]
```

## Creating Custom Formats

To create a custom format, inherit from `BaseFormat`:

```python
from rl_onoff.tasks.formats import BaseFormat
from typing import Optional, Dict

class MyCustomFormat(BaseFormat):
    def get_system_prompt(self) -> str:
        return "Your custom system prompt here..."
    
    def extract(self, response: str) -> Dict[str, Optional[str]]:
        # Your extraction logic here
        reasoning = ...
        answer = ...
        # You can add additional fields for extensibility
        return {
            "reasoning": reasoning,
            "answer": answer,
            # "additional_field": ...
        }
```

