# Rewards Module

The rewards module provides a framework for evaluating model outputs against reference answers. Rewards compute scores that indicate how well a model's prediction matches the expected answer.

## Overview

Rewards are used in tasks to evaluate model responses. Each reward implements the `BaseReward` interface and provides a `compute()` method that takes predictions and references and returns a score.

## Available Rewards

### Built-in Rewards

The following rewards are available in the `REWARD_REGISTRY`:

1. **`math_verify`** - `MathVerifyReward`
   - Verifies mathematical equivalence using the `math_verify` library
   - Parses and verifies that predictions are mathematically equivalent to references
   - Returns 1.0 if equivalent, 0.0 otherwise
   - **Dependencies**: `math-verify` (install with `pip install math-verify`)

2. **`exact_match`** - `ExactMatchReward`
   - Checks if prediction exactly matches reference (with optional normalization)
   - Returns 1.0 for exact match, 0.0 otherwise
   - **Parameters**: `normalize` (bool, default: True) - normalize text before comparison

3. **`bleu`** - `BLEUReward`
   - Computes BLEU score for text similarity
   - Returns BLEU score between 0.0 and 1.0
   - **Parameters**: `n_gram` (int, default: 4) - n-gram order
   - **Dependencies**: `nltk` (install with `pip install nltk && python -m nltk.downloader punkt`)

4. **`rouge`** - `ROUGEReward`
   - Computes ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
   - Returns dictionary with ROUGE scores
   - **Parameters**: `rouge_types` (List[str], default: ['rouge1', 'rouge2', 'rougeL'])
   - **Dependencies**: `rouge-score` (install with `pip install rouge-score`)

5. **`perplexity`** - `PerplexityReward`
   - Computes perplexity of generated text using a model backend
   - Lower perplexity indicates better model fit
   - **Parameters**: `backend` (BaseBackend) - required backend instance
   - **Dependencies**: Model backend (HuggingFace, vLLM, or SGLang)

## Usage

### Using the Registry

The easiest way to use rewards is through the registry:

```python
from rl_onoff.tasks.rewards import create_reward

# Create a reward instance
reward = create_reward("math_verify")

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
from rl_onoff.tasks.rewards.builtin import MathVerifyReward, ExactMatchReward

# Math verification
math_reward = MathVerifyReward()
score = math_reward.compute("${1,2,3,4}$", "${1,3} \\cup {2,4}$")
# Returns: 1.0 (mathematically equivalent)

# Exact match
exact_reward = ExactMatchReward(normalize=True)
score = exact_reward.compute("Hello", "hello")
# Returns: 1.0 (with normalization)
```

### Using with Tasks

Rewards are typically used through tasks:

```python
from rl_onoff.tasks import MathTask

# MathTask uses MathVerifyReward by default
task = MathTask()

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
    **kwargs
) -> Union[float, List[float], Dict[str, Any]]:
    """Compute reward value(s).
    
    Args:
        predictions: Predicted text(s)
        references: Reference text(s) or list of reference lists
        **kwargs: Additional arguments
        
    Returns:
        Reward value(s) as float, list of floats, or dict
    """
    pass
```

### Input Formats

Rewards support flexible input formats:

- **Single prediction/reference**: `compute("pred", "ref")` → returns `float`
- **List of predictions/references**: `compute(["pred1", "pred2"], ["ref1", "ref2"])` → returns `List[float]`
- **Multiple references per prediction**: `compute("pred", [["ref1", "ref2"]])` → checks against all references

## Creating Custom Rewards

To create a custom reward, inherit from `BaseReward`:

```python
from rl_onoff.tasks.rewards.base import BaseReward
from typing import Union, List

class MyCustomReward(BaseReward):
    """Custom reward that computes a specific metric."""
    
    def __init__(self, threshold: float = 0.5):
        """Initialize custom reward.
        
        Args:
            threshold: Threshold for positive reward
        """
        super().__init__("my_custom_reward")
        self.threshold = threshold
    
    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]],
        **kwargs
    ) -> Union[float, List[float]]:
        """Compute custom reward."""
        is_single = isinstance(predictions, str)
        if is_single:
            predictions = [predictions]
        
        # Normalize references format
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references, list) and len(references) > 0:
            if isinstance(references[0], str):
                references = [[ref] for ref in references]
        
        scores = []
        for pred, refs in zip(predictions, references):
            # Your custom scoring logic here
            score = self._compute_score(pred, refs[0])
            scores.append(score)
        
        return scores[0] if is_single else scores
    
    def _compute_score(self, prediction: str, reference: str) -> float:
        """Compute score for a single prediction-reference pair."""
        # Your custom logic
        return 1.0 if prediction == reference else 0.0
```

### Registering Custom Rewards

To make your custom reward available through the registry:

```python
from rl_onoff.tasks.rewards import REWARD_REGISTRY

# Register your reward
REWARD_REGISTRY["my_custom"] = MyCustomReward

# Now you can use it
from rl_onoff.tasks.rewards import create_reward
reward = create_reward("my_custom", threshold=0.7)
```

## Examples

### Math Verification

```python
from rl_onoff.tasks.rewards import create_reward

reward = create_reward("math_verify")

# Equivalent expressions
score = reward.compute("4", "2 + 2")
# Returns: 1.0

# Set operations
score = reward.compute("${1,2,3,4}$", "${1,3} \\cup {2,4}$")
# Returns: 1.0

# Incorrect answer
score = reward.compute("5", "10")
# Returns: 0.0
```

### Exact Match

```python
from rl_onoff.tasks.rewards.builtin import ExactMatchReward

reward = ExactMatchReward(normalize=True)

# Case-insensitive match
score = reward.compute("Hello", "hello")
# Returns: 1.0

# No match
score = reward.compute("Hello", "World")
# Returns: 0.0
```

### BLEU Score

```python
from rl_onoff.tasks.rewards.builtin import BLEUReward

reward = BLEUReward(n_gram=4)

score = reward.compute(
    "the cat sat on the mat",
    "the cat is on the mat"
)
# Returns: BLEU score between 0.0 and 1.0
```

### ROUGE Scores

```python
from rl_onoff.tasks.rewards.builtin import ROUGEReward

reward = ROUGEReward(rouge_types=['rouge1', 'rouge2', 'rougeL'])

scores = reward.compute(
    "the cat sat on the mat",
    "the cat is on the mat"
)
# Returns: {'rouge1': 0.8, 'rouge2': 0.6, 'rougeL': 0.75}
```

### Perplexity

```python
from rl_onoff.tasks.rewards.builtin import PerplexityReward
from rl_onoff.backends import get_backend

# Initialize backend
backend = get_backend("huggingface", model_name="gpt2")
backend.load()

# Create reward with backend
reward = PerplexityReward(backend=backend)

# Compute perplexity
perplexity = reward.compute("the cat sat on the mat")
# Returns: perplexity value (lower is better)
```

## Reward Registry

The `REWARD_REGISTRY` maps string names to reward classes:

```python
from rl_onoff.tasks.rewards import REWARD_REGISTRY

# List available rewards
print(list(REWARD_REGISTRY.keys()))
# ['math_verify', 'exact_match', 'bleu', 'rouge', 'perplexity']

# Create reward from registry
reward = REWARD_REGISTRY["math_verify"]()
```

## Dependencies

Different rewards have different dependencies:

- **MathVerifyReward**: `pip install math-verify`
- **BLEUReward**: `pip install nltk && python -m nltk.downloader punkt`
- **ROUGEReward**: `pip install rouge-score`
- **PerplexityReward**: Requires a model backend (HuggingFace, vLLM, or SGLang)
- **ExactMatchReward**: No external dependencies

## Notes

- Rewards assume that predictions are already extracted answers (not full model responses)
- For tasks, use `task.extract_answer()` to extract answers from responses before evaluation
- Math verification uses `math_verify.parse()` and `math_verify.verify()` internally
- All rewards handle both single and batch inputs automatically
- Rewards can return different types: `float`, `List[float]`, or `Dict[str, float]` depending on the reward type

