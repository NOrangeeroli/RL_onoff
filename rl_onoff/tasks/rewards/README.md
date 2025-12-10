# Rewards Module

Rewards are used to evaluate model outputs and provide scores that can be used for training or evaluation.

## Available Rewards

1. **`math_verify`** - `MathVerifyReward`
   - Verifies mathematical equivalence between predictions and references
   - Uses the `math_verify` library for parsing and verification
   - Returns 1.0 if mathematically equivalent, 0.0 otherwise
   - **Dependencies**: `math-verify` (install with `pip install math-verify`)

## Usage

### Basic Usage

The easiest way to use rewards is through the registry:

```python
from rl_onoff.tasks.rewards import create_reward

# Create a reward instance
reward = create_reward({"name": "math_verify"})

# Compute score for single prediction
score = reward.compute("42", "42")
# Returns: 1.0

# Compute scores for multiple predictions
predictions = ["10", "15", "20"]
references = ["10", "15", "20"]
scores = reward.compute(predictions, references)
# Returns: [1.0, 1.0, 1.0]
```

### Direct Instantiation

You can also instantiate rewards directly:

```python
from rl_onoff.tasks.rewards.math_verify import MathVerifyReward

# Math verification
math_reward = MathVerifyReward()
score = math_reward.compute("${1,2,3,4}$", "${1,3} \\cup {2,4}$")
# Returns: 1.0 (mathematically equivalent)
```

### Using with Tasks

Rewards are typically used through tasks:

```python
from rl_onoff.tasks import create_task

# Create a task that uses MathVerifyReward
task = create_task({
    "template_type": "simple",
    "reward_type": "math_verify",
    "format_type": "boxed"
})

# Evaluate a response
prediction = "The answer is 42"
reference = "42"
score = task.evaluate(prediction, reference)
# Returns: 1.0 if mathematically equivalent
```

## Reward Interface

All rewards inherit from `BaseReward` and implement:

```python
def compute(
    self,
    predictions: Union[str, List[str]],
    references: Union[str, List[str], List[List[str]]],
) -> Union[float, List[float]]:
    """Compute reward value(s).
    
    Args:
        predictions: Predicted text(s)
        references: Reference text(s) or list of reference lists
        
    Returns:
        Reward value(s) as float or list of floats
    """
    pass
```

## Math Verification Examples

### Equivalent Expressions

```python
from rl_onoff.tasks.rewards import create_reward

reward = create_reward({"name": "math_verify"})

# Equivalent arithmetic
score = reward.compute("4", "2 + 2")
# Returns: 1.0

# Equivalent set expressions
score = reward.compute("${1,2,3,4}$", "${1,3} \\cup {2,4}$")
# Returns: 1.0

# Non-equivalent
score = reward.compute("5", "10")
# Returns: 0.0
```

## Registry

You can check available rewards:

```python
from rl_onoff.tasks.rewards import REWARD_REGISTRY

print(list(REWARD_REGISTRY.keys()))
# ['math_verify']
```

## Dependencies

- **MathVerifyReward**: `pip install math-verify`
