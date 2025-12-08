# RL On/Off: LLM Evaluation and Model Switching Framework

A flexible Python framework for sampling from LLM models, calculating metrics, extracting token-wise distributions, computing divergences, and implementing conditional model switching based on divergence thresholds.

## Features

1. **Multi-Backend Support**: Works with HuggingFace Transformers, vLLM, and SGLang backends
2. **Sampling & Rewards**: Sample from models and calculate various rewards (perplexity, BLEU, ROUGE, exact match, and custom rewards)
3. **Distribution Extraction**: Extract token-wise probability distributions from models for given question-solution pairs
4. **Divergence Calculation**: Compute token-level KL and Jensen-Shannon divergences between model distributions
5. **Conditional Model Switching**: Generate responses with automatic switching between models when divergence exceeds thresholds

## Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# For optional backends, uncomment and install:
# pip install vllm
# pip install 'sglang[all]'
```

## Quick Start

### 1. Sample from a Model and Calculate Rewards

```bash
python scripts/sample_and_metrics.py \
    --backend-type huggingface \
    --model-name gpt2 \
    --dataset data.jsonl \
    --output results.json \
    --metrics all
```

### 2. Extract Token-Wise Distributions

```bash
python scripts/extract_distributions.py \
    --backend-type huggingface \
    --model-name gpt2 \
    --dataset data.jsonl \
    --output distributions.npz \
    --format npz
```

### 3. Calculate Token-Level Divergence

```bash
python scripts/calculate_divergence.py \
    --backend-type-a huggingface \
    --model-name-a gpt2 \
    --backend-type-b huggingface \
    --model-name-b gpt2-medium \
    --dataset data.jsonl \
    --output divergence_results.json \
    --divergence-type both
```

### 4. Generate with Conditional Model Switching

```bash
python scripts/switch_generate.py \
    --backend-type-a huggingface \
    --model-name-a gpt2 \
    --backend-type-b huggingface \
    --model-name-b gpt2-medium \
    --question "What is machine learning?" \
    --output switched_response.json \
    --threshold 0.5 \
    --divergence-type js
```

## Usage Examples

### Python API

#### Backend Initialization

```python
from rl_onoff.backends import get_backend

# HuggingFace backend
backend = get_backend("huggingface", model_name="gpt2")
backend.load()

# Generate text
response = backend.generate("Hello, world!", max_new_tokens=50)
```

#### Sampling and Rewards

```python
from rl_onoff.sampling import Sampler, SamplingConfig
from rl_onoff.tasks.rewards import RewardRegistry
from rl_onoff.tasks.rewards.builtin import BLEUReward, ROUGEReward

# Initialize sampler
sampler = Sampler(backend)
config = SamplingConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
)

# Generate samples
predictions = sampler.sample("What is AI?", config=config)

# Calculate rewards
registry = RewardRegistry()
registry.register(BLEUReward())
registry.register(ROUGEReward())

results = registry.compute_all(
    predictions=["AI is artificial intelligence."],
    references=[["AI stands for Artificial Intelligence."]]
)
```

#### Distribution Extraction

```python
from rl_onoff.distributions import DistributionExtractor

extractor = DistributionExtractor(backend)

# Extract distributions for a question-solution pair
distributions, token_ids = extractor.extract_distributions(
    question="What is 2+2?",
    solution="4",
    use_logits=False,
    return_token_ids=True,
)
# distributions shape: (seq_len, vocab_size)
```

#### Divergence Calculation

```python
from rl_onoff.divergence import DivergenceCalculator

calculator = DivergenceCalculator()

# Compute divergences for two models
result = calculator.compute_divergence_for_solutions(
    question="What is 2+2?",
    solution="4",
    backend_a=backend_a,
    backend_b=backend_b,
    divergence_type="both",
)

# result contains "kl" and/or "js" divergence arrays
kl_divergence = result["kl"]  # shape: (seq_len,)
js_divergence = result["js"]  # shape: (seq_len,)
```

#### Model Switching

```python
from rl_onoff.switching import ModelSwitcher

switcher = ModelSwitcher(
    backend_a=backend_a,
    backend_b=backend_b,
    divergence_type="js",
    threshold=0.5,
    switch_back_threshold=0.25,
)

# Generate with conditional switching
response, switch_points = switcher.generate_with_switching(
    question="Explain quantum computing",
    max_new_tokens=200,
    return_switch_points=True,
)

# switch_points contains information about when switches occurred
for switch in switch_points:
    print(f"Switched from {switch['from_model']} to {switch['to_model']} "
          f"at position {switch['position']} with divergence {switch['divergence']:.4f}")
```

#### Data Loading

```python
from rl_onoff.utils.data_loader import load_data

# Load from JSON, JSONL, CSV, or HuggingFace datasets
data = load_data(
    "data.jsonl",
    question_column="question",
    solution_column="solution",
)

for item in data:
    print(item["question"], item["solution"])
```

## Data Format

The framework supports multiple data formats:

### JSON/JSONL Format

```json
{
  "question": "What is machine learning?",
  "solution": "Machine learning is a subset of AI."
}
```

### CSV Format

```csv
question,solution
"What is ML?","Machine learning is..."
```

### HuggingFace Datasets

```python
from datasets import load_dataset

dataset = load_dataset("squad")
```

## Architecture

### Backend Abstraction Layer

All backends implement a common interface:
- `load()` - Load model and tokenizer
- `generate()` - Generate text
- `get_logits()` - Get token logits
- `get_probabilities()` - Get token probabilities
- `get_tokenizer()` - Get tokenizer instance

### Rewards Framework

Extensible reward system with built-in rewards:
- Perplexity
- BLEU
- ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- Exact Match

Custom rewards can be added by extending `BaseReward`.

### Distribution Extraction

Extracts token-wise probability distributions (or logits) at each position of a solution given a question.

### Divergence Calculation

Computes token-level divergences:
- **KL Divergence**: `KL(P||Q) = Σ P(i) * log(P(i) / Q(i))`
- **Jensen-Shannon Divergence**: `JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)` where `M = 0.5 * (P + Q)`

### Model Switching

Conditionally switches between two models during generation:
1. Start with model A
2. At each token, compute divergence between A and B distributions
3. If divergence > threshold, switch to model B
4. Monitor and switch back to model A when divergence drops below switch-back threshold
5. Return response with switch points logged

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- NumPy, SciPy
- pandas, datasets
- Optional: vLLM, SGLang for additional backends

## License

[Specify your license here]

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{rl_onoff,
  title = {RL On/Off: LLM Evaluation and Model Switching Framework},
  author = {Your Name},
  year = {2024},
}
```

