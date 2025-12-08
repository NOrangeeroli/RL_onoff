# Tasks Module

The tasks module provides a framework for defining task-specific prompt templates and metrics.

## Overview

Each task combines:
- **Prompt Template**: Formats questions into prompts for the model
- **Metric**: Evaluates model responses against reference answers

## Usage Example

### Math Task

```python
from rl_onoff.tasks import MathTask
from rl_onoff.backends import get_backend
from rl_onoff.sampling import Sampler, SamplingConfig

# Initialize task
math_task = MathTask()

# Format a prompt from a question
question = "What is 2 + 2?"
prompt = math_task.format_prompt(question)
# Returns: "Solve the following math problem step by step. Show your work and provide the final answer.\n\nProblem: What is 2 + 2?\n\nSolution:"

# Generate response
backend = get_backend("huggingface", model_name="gpt2")
backend.load()
sampler = Sampler(backend)
response = sampler.sample([prompt], config=SamplingConfig(max_new_tokens=50))[0]

# Evaluate response
reference = "4"
score = math_task.evaluate(response, reference)
# Returns: 1.0 if mathematically equivalent, 0.0 otherwise
```

### Custom Prompt Template

```python
# Use a custom prompt template
custom_template = "Answer this math question: $question\n\nYour answer:"
math_task = MathTask(prompt_template=custom_template)

prompt = math_task.format_prompt("What is 3 * 4?")
# Returns: "Answer this math question: What is 3 * 4?\n\nYour answer:"
```

## Creating New Tasks

To create a new task, inherit from `BaseTask`:

```python
from rl_onoff.tasks.base import BaseTask
from rl_onoff.tasks.rewards.builtin import ExactMatchMetric

class MyTask(BaseTask):
    def _create_metric(self):
        return ExactMatchMetric()
    
    def get_prompt_template(self):
        return "Question: $question\nAnswer:"
```

