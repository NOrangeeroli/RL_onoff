# Response Formats

The `formats` module provides response format classes that define both the system prompt (which instructs the model on how to format responses) and the extractor logic (which parses responses following that format).

## Overview

Each format class implements three main methods:
- **`get_system_prompt()`**: Returns the system prompt that explains to the model how to structure its response
- **`extract(response)`**: Parses the model's response to extract reasoning and answer
- **`format_message(question)`**: Formats a question into message dicts for chat templates (combines system prompt + question)

## Available Formats

### BoxedFormat

For math responses that use `\boxed{answer}` format.

```python
from rl_onoff.tasks.formats import BoxedFormat

format = BoxedFormat()

# 1. Get system prompt
system_prompt = format.get_system_prompt()
# Returns: "You are a helpful math tutor. Solve the problem step by step..."

# 2. Format a question into messages for chat templates
question = "What is 2 + 2?"
messages = format.format_message(question)
# Returns: [
#     {"role": "system", "content": "You are a helpful math tutor..."},
#     {"role": "user", "content": "What is 2 + 2?"}
# ]

# 3. Extract from response
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

# 1. Get system prompt
system_prompt = format.get_system_prompt()
# Returns: "You are a helpful math tutor. Solve the problem step by step..."

# 2. Format a question into messages for chat templates
question = "What is 2 + 2?"
messages = format.format_message(question)
# Returns: [
#     {"role": "system", "content": "You are a helpful math tutor..."},
#     {"role": "user", "content": "What is 2 + 2?"}
# ]

# 3. Extract from response
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

## Main Methods

### 1. `get_system_prompt() -> str`

Returns the system prompt that instructs the model on the expected response format.

```python
format = BoxedFormat()
system_prompt = format.get_system_prompt()
```

### 2. `format_message(question: str) -> List[Dict[str, str]]`

Formats a question into a list of message dicts ready for chat templates. Combines the system prompt with the user question.

```python
format = BoxedFormat()
messages = format.format_message("What is 2 + 2?")
# Returns: [
#     {"role": "system", "content": "You are a helpful math tutor..."},
#     {"role": "user", "content": "What is 2 + 2?"}
# ]

# Use with chat templates
from rl_onoff.tasks.chat_templates import create_chat_template
template = create_chat_template("openai")
prompt = template.format(messages, add_generation_prompt=True)
```

### 3. `extract(response: str) -> Dict[str, Optional[str]]`

Extracts information from a model response following the format. Returns a dictionary with:
- `"answer"`: The final answer (required)
- `"reasoning"`: The reasoning/process (optional)

```python
format = BoxedFormat()
response = "Let me solve this... \\boxed{42}"
result = format.extract(response)
# Returns: {"reasoning": "Let me solve this...", "answer": "42"}
```

## Usage with Tasks

Tasks use formats to define their response structure:

```python
from rl_onoff.tasks import create_task

# Create task (uses format from config)
task = create_task({
    "template_type": "simple",
    "reward_type": "math_verify",
    "format_type": "boxed"
})

# Format a query (uses format.format_message internally)
question = "What is 2 + 2?"
prompt = task.format_query(question)

# Extract answer from response (uses format.extract internally)
response = "The answer is \\boxed{4}"
result = task.extract_answer(response)
answer = result["answer"]  # "4"
```

## Using the Registry

Formats can be created from the registry:

```python
from rl_onoff.tasks.formats import create_format, FORMAT_REGISTRY

# Create format from name
format = create_format("boxed")

# List available formats
print(list(FORMAT_REGISTRY.keys()))
# ['boxed', 'structured']
```

## Creating Custom Formats

To create a custom format, inherit from `BaseFormat` and implement the required methods:

```python
from rl_onoff.tasks.formats import BaseFormat
from typing import Optional, Dict

class MyCustomFormat(BaseFormat):
    def get_system_prompt(self) -> str:
        """Return the system prompt explaining the format."""
        return "Your custom system prompt here..."
    
    def extract(self, response: str) -> Dict[str, Optional[str]]:
        """Extract answer and reasoning from response."""
        # Your extraction logic here
        reasoning = ...
        answer = ...
        # You can add additional fields for extensibility
        return {
            "reasoning": reasoning,
            "answer": answer,
            # "additional_field": ...
        }
    
    # format_message() is already implemented in BaseFormat
    # It automatically combines get_system_prompt() with the question
```

### Registering Custom Formats

To make your custom format available through the registry:

```python
from rl_onoff.tasks.formats import FORMAT_REGISTRY

# Register your format
FORMAT_REGISTRY["my_custom"] = MyCustomFormat

# Now you can use it
from rl_onoff.tasks.formats import create_format
format = create_format("my_custom")
```

