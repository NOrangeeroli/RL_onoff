# Chat Templates

Chat templates format queries with task-specific system prompts into a single prompt string that models can process.

## Main Usage: format_query()

The primary method for task-based formatting is `format_query()`, which combines a query with a system prompt that explains the output format.

### Basic Usage

```python
from rl_onoff.tasks.chat_templates import OpenAIChatTemplate

template = OpenAIChatTemplate()

# Format query with task-specific system prompt
query = "What is 2+2?"
prompt = template.format_query(query, task_type="math")
# Automatically uses math system prompt and formats appropriately
```

### Available Task Types

- **"math"** - Math problem solving (styles: "default", "boxed", "concise")
- **"coding"** - Code generation (styles: "default", "python", "test")
- **"qa"** - Question answering (styles: "default", "detailed", "concise")
- **"general"** - General tasks (styles: "default", "step_by_step")

### Examples

#### Math Task

```python
from rl_onoff.tasks.chat_templates import OpenAIChatTemplate

template = OpenAIChatTemplate()

# Default math prompt
prompt = template.format_query("Solve: 2x + 5 = 15", task_type="math")
# Uses math system prompt explaining step-by-step solution format

# Boxed answer format
prompt = template.format_query("What is 2+2?", task_type="math", prompt_style="boxed")
# Uses system prompt that asks for \\boxed{{answer}} format
```

#### Coding Task

```python
from rl_onoff.tasks.chat_templates import LlamaChatTemplate

template = LlamaChatTemplate()

# Python coding task
prompt = template.format_query(
    "Write a function to calculate factorial",
    task_type="coding",
    prompt_style="python"
)
# Uses coding system prompt with Python-specific instructions
```

#### Custom System Prompt

```python
from rl_onoff.tasks.chat_templates import ChatMLTemplate

template = ChatMLTemplate()

# Use custom system prompt
custom_prompt = "You are an expert mathematician. Show all work."
prompt = template.format_query(
    "What is 2+2?",
    system_prompt=custom_prompt
)
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

## System Prompts

System prompts are organized by task type and style. Access them directly:

```python
from rl_onoff.tasks.chat_templates import get_system_prompt

# Get math system prompt
math_prompt = get_system_prompt("math", "default")

# Get coding system prompt with Python style
coding_prompt = get_system_prompt("coding", "python")
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

