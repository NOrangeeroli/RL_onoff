# Chat Templates

Chat templates format conversations with roles (system, user, assistant) into a single prompt string that models can process.

## Available Templates

### 1. SimpleChatTemplate
Simple concatenation without special formatting.

```python
from rl_onoff.tasks.chat_templates import SimpleChatTemplate

template = SimpleChatTemplate()
messages = [
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages)
# Returns: "What is 2+2?"
```

### 2. OpenAIChatTemplate
OpenAI-style with role labels.

```python
from rl_onoff.tasks.chat_templates import OpenAIChatTemplate

template = OpenAIChatTemplate()
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages)
# Returns:
# System: You are a helpful assistant.
#
# User: What is 2+2?
```

### 3. LlamaChatTemplate
Llama-style with special tokens: `<s>`, `</s>`, `[INST]`, `[/INST]`

```python
from rl_onoff.tasks.chat_templates import LlamaChatTemplate

template = LlamaChatTemplate()
messages = [
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages, add_generation_prompt=True)
# Returns: "<s> [INST] What is 2+2? [/INST] <s> [INST]"
```

### 4. ChatMLTemplate
ChatML-style with XML-like tags: `<|im_start|>` and `<|im_end|>`

```python
from rl_onoff.tasks.chat_templates import ChatMLTemplate

template = ChatMLTemplate()
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages, add_generation_prompt=True)
# Returns:
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# What is 2+2?<|im_end|>
# <|im_start|>assistant
```

## Usage with Simple Format

```python
from rl_onoff.tasks.chat_templates import OpenAIChatTemplate

template = OpenAIChatTemplate()

# Simple format method
prompt = template.format_simple(
    user_message="What is 2+2?",
    system_message="You are a math tutor.",
    add_generation_prompt=True
)
```

## Creating Custom Templates

```python
from rl_onoff.tasks.chat_templates import BaseChatTemplate

class MyChatTemplate(BaseChatTemplate):
    def format(self, messages, add_generation_prompt=False, **kwargs):
        # Your custom formatting logic
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"[{role.upper()}] {content}")
        
        if add_generation_prompt:
            parts.append("[ASSISTANT]")
        
        return "\n".join(parts)
```

