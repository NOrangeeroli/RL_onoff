# Chat Templates

Chat templates format conversations into prompt strings that models can process. They handle different chat formats like OpenAI, Llama, ChatML, and Simple.

## Main Usage: format()

The primary method for formatting conversations is `format()`, which takes a list of messages and formats them according to the template style.

### Basic Usage

```python
from rl_onoff.tasks.chat_templates import create_chat_template

# Create a template from name
template = create_chat_template("openai")

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
```

### Examples

#### OpenAI Format

```python
from rl_onoff.tasks.chat_templates import create_chat_template

template = create_chat_template("openai")
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages, add_generation_prompt=True)
```

#### Llama Format

```python
from rl_onoff.tasks.chat_templates import create_chat_template

template = create_chat_template("llama")
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages, add_generation_prompt=True)
```

#### Custom System Prompt

```python
from rl_onoff.tasks.chat_templates import create_chat_template

template = create_chat_template("chatml")
messages = [
    {"role": "system", "content": "You are an expert mathematician. Show all work."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = template.format(messages, add_generation_prompt=True)
```

## Available Templates

### 1. SimpleChatTemplate
Simple concatenation without special formatting.

```python
from rl_onoff.tasks.chat_templates import SimpleChatTemplate

template = SimpleChatTemplate()
prompt = template.format_query("What is 2+2?", task_type="math")
# Returns: system_prompt + "\n\n" + query + "\n\n"
```

### 2. OpenAIChatTemplate
OpenAI-style with role labels.

```python
from rl_onoff.tasks.chat_templates import OpenAIChatTemplate

template = OpenAIChatTemplate()
prompt = template.format_query("What is 2+2?", task_type="math")
# Returns:
# System: [math system prompt]
#
# User: What is 2+2?
#
# Assistant:
```

### 3. LlamaChatTemplate
Llama-style with special tokens: `<s>`, `</s>`, `[INST]`, `[/INST]`

```python
from rl_onoff.tasks.chat_templates import LlamaChatTemplate

template = LlamaChatTemplate()
prompt = template.format_query("What is 2+2?", task_type="math")
# Returns: "<s> [INST] <<SYS>>\n[system prompt]\n<</SYS>>\n\nWhat is 2+2? [/INST] <s> [INST]"
```

### 4. ChatMLTemplate
ChatML-style with XML-like tags: `<|im_start|>` and `<|im_end|>`

```python
from rl_onoff.tasks.chat_templates import ChatMLTemplate

template = ChatMLTemplate()
prompt = template.format_query("What is 2+2?", task_type="math")
# Returns:
# <|im_start|>system
# [math system prompt]<|im_end|>
# <|im_start|>user
# What is 2+2?<|im_end|>
# <|im_start|>assistant
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

