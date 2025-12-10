# Tasks Module

The tasks module provides a framework for defining task-specific prompt templates and rewards.

## Overview

Each task combines:
- **Chat Template**: Formats questions into prompts for the model
- **Format**: Defines response structure and extraction logic
- **Reward**: Evaluates model responses against reference answers

## Usage Example

### Creating a Task

```python
from rl_onoff.tasks import create_task
from rl_onoff.backends import create_backend
from rl_onoff.sampling import Sampler
from rl_onoff.sampling.config import SamplingConfig

# Create task from config file or dict
task = create_task({
    "template_type": "simple",
    "reward_type": "math_verify",
    "format_type": "boxed"
})

# Format a query from a question
question = "What is 2 + 2?"
prompt = task.format_query(question)

# Generate response
backend = create_backend({"backend_type": "huggingface", "model_name": "gpt2"})
backend.load()
sampler = Sampler(backend)
config = SamplingConfig(max_length=50)
response = sampler.sample([prompt], config=config)[0]

# Evaluate response
reference = "4"
score = task.evaluate(response, reference)
# Returns: 1.0 if mathematically equivalent, 0.0 otherwise
```

### Using a Config File

```python
from rl_onoff.tasks import create_task

# Create task from config file
task = create_task("rl_onoff/tasks/configs/math_default.yaml")

# Use the task
question = "What is 3 * 4?"
prompt = task.format_query(question)
```

### Using TaskConfig

```python
from rl_onoff.tasks import create_task, TaskConfig

# Create TaskConfig explicitly
config = TaskConfig(
    template_type="simple",
    reward_type="math_verify",
    format_type="boxed"
)
task = create_task(config)
```

