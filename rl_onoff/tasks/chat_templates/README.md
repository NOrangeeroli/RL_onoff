# Chat Templates

Chat templates format conversations into prompt strings that models can process. They handle different chat formats like OpenAI, Llama, ChatML, and Simple.

## Main Usage: format()

The primary method for formatting conversations is `format()`, which takes a list of messages and formats them according to the template style.

### Basic Usage

```python
from rl_onoff.tasks.chat_templates import create_chat_template

# Create a template from name (using config dict)
template = create_chat_template({"name": "openai"})

# Format messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages, add_generation_prompt=True)
```

### Using Registry

```python
from rl_onoff.tasks.chat_templates import CHAT_TEMPLATE_REGISTRY

# Access classes directly from registry
OpenAIChatTemplate = CHAT_TEMPLATE_REGISTRY["openai"]
template = OpenAIChatTemplate()

# Format messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages, add_generation_prompt=True)
```

## Available Templates

### 1. SimpleChatTemplate

Simple concatenation without special formatting tags or role labels. Just joins message contents with double newlines.

```python
from rl_onoff.tasks.chat_templates import create_chat_template

template = create_chat_template({"name": "simple"})
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages, add_generation_prompt=True)
# Returns: "You are a math tutor.\n\nWhat is 2+2?\n\n"
```

### 2. OpenAIChatTemplate

OpenAI-style with role labels (`System:`, `User:`, `Assistant:`).

```python
from rl_onoff.tasks.chat_templates import create_chat_template

template = create_chat_template({"name": "openai"})
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages, add_generation_prompt=True)
# Returns:
# System: You are a math tutor.
#
# User: What is 2+2?
#
# Assistant:
```

### 3. LlamaChatTemplate

Llama-style with special tokens: `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`

```python
from rl_onoff.tasks.chat_templates import create_chat_template

template = create_chat_template({"name": "llama"})
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages, add_generation_prompt=True)
# Returns:
# "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a math tutor.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
```

### 4. ChatMLTemplate

ChatML-style with XML-like tags: `<|im_start|>` and `<|im_end|>`

```python
from rl_onoff.tasks.chat_templates import create_chat_template

template = create_chat_template({"name": "chatml"})
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages, add_generation_prompt=True)
# Returns:
# <|im_start|>system
# You are a math tutor.<|im_end|>
# <|im_start|>user
# What is 2+2?<|im_end|>
# <|im_start|>assistant
```

## Message Format

All templates expect messages in the following format:

```python
messages = [
    {"role": "system", "content": "System message content"},
    {"role": "user", "content": "User message content"},
    {"role": "assistant", "content": "Assistant message content"},
]
```

The `role` field can be:
- `"system"`: System instructions or context
- `"user"`: User queries or inputs
- `"assistant"`: Assistant responses
- `"assistant_generation"`: Used internally when a response has already been generated

## Parameters

### format()

- `messages` (List[Dict[str, str]]): List of message dictionaries with `"role"` and `"content"` keys
- `add_generation_prompt` (bool, default=True): If `True`, adds formatting tokens/prompts at the end to indicate where the assistant should start generating

## Examples

### Multi-turn Conversation

```python
from rl_onoff.tasks.chat_templates import create_chat_template

template = create_chat_template({"name": "chatml"})
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."},
    {"role": "user", "content": "What is 3+3?"}
]
prompt = template.format(messages, add_generation_prompt=True)
```

### Without Generation Prompt

```python
from rl_onoff.tasks.chat_templates import create_chat_template

template = create_chat_template({"name": "openai"})
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
]
# Don't add generation prompt since we already have the assistant response
prompt = template.format(messages, add_generation_prompt=False)
```

## Creating Custom Templates

To create a custom chat template, inherit from `BaseChatTemplate` and implement the `format()` method:

```python
from rl_onoff.tasks.chat_templates.base import BaseChatTemplate
from typing import List, Dict

class MyChatTemplate(BaseChatTemplate):
    def format(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        # Your custom formatting logic
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"[{role.upper()}] {content}")
        
        if add_generation_prompt:
            parts.append("[ASSISTANT]")
        
        return "\n".join(parts)

# Register your template (optional)
from rl_onoff.tasks.chat_templates import CHAT_TEMPLATE_REGISTRY
CHAT_TEMPLATE_REGISTRY["my_template"] = MyChatTemplate
```

## Registry

All available templates are registered in `CHAT_TEMPLATE_REGISTRY`:

```python
from rl_onoff.tasks.chat_templates import CHAT_TEMPLATE_REGISTRY

print(CHAT_TEMPLATE_REGISTRY.keys())
# dict_keys(['simple', 'openai', 'llama', 'chatml'])
```

Use the `create_chat_template()` function to instantiate templates:

```python
from rl_onoff.tasks.chat_templates import create_chat_template

# All of these are equivalent:
template1 = create_chat_template({"name": "simple"})
template2 = create_chat_template({"name": "openai"})
template3 = create_chat_template({"name": "llama"})
template4 = create_chat_template({"name": "chatml"})
```
