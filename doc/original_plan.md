---
name: LLM Evaluation and Switching Framework
overview: Create a Python framework for sampling from LLM models across multiple backends, calculating metrics, extracting token distributions, computing divergences, and implementing conditional model switching based on divergence thresholds.
todos:
  - id: setup-project
    content: Create project structure (directories, __init__.py files, setup.py, requirements.txt)
    status: completed
  - id: backend-base
    content: Implement base backend interface in rl_onoff/backends/base.py
    status: completed
  - id: backend-hf
    content: Implement HuggingFace backend in rl_onoff/backends/huggingface.py
    status: completed
  - id: backend-vllm
    content: Implement vLLM backend in rl_onoff/backends/vllm.py
    status: completed
  - id: backend-sglang
    content: Implement SGLang backend in rl_onoff/backends/sglang.py
    status: completed
  - id: sampling
    content: Implement sampling module with configurable strategies
    status: completed
  - id: metrics
    content: Create metrics framework with built-in and extensible metrics
    status: completed
  - id: distributions
    content: Implement token-wise distribution extraction (logits and probabilities)
    status: completed
  - id: divergence
    content: Implement JS and KL divergence calculations at token level
    status: completed
  - id: switching
    content: Implement conditional model switching with threshold-based logic and switch-back mechanism
    status: completed
  - id: data-utils
    content: Create data loading utilities supporting JSON, JSONL, CSV, and HuggingFace datasets
    status: completed
  - id: cli-scripts
    content: Create four CLI scripts for each requirement (sample+metrics, extract distributions, calculate divergence, switch generate)
    status: completed
  - id: documentation
    content: Create README.md with usage examples and API documentation
    status: completed
---

# LLM Evaluation and Model Switching Framework

## Overview

A flexible Python framework for:

1. Sampling from LLM models (HuggingFace, vLLM, SGLang) and calculating metrics
2. Extracting token-wise probability distributions from models
3. Computing token-level divergence metrics (JS, KL) between models
4. Conditional model switching during generation based on divergence thresholds

## Architecture

### Core Components

1. **Backend Abstraction Layer** (`rl_onoff/backends/`)

   - Unified interface for model backends
   - Implementations: HuggingFace, vLLM, SGLang
   - Methods: load_model, generate, get_logits, get_probabilities

2. **Sampling Module** (`rl_onoff/sampling/`)

   - Configurable sampling strategies (greedy, top-k, top-p, temperature)
   - Batch and single-item support
   - Response collection and formatting

3. **Metrics Module** (`rl_onoff/metrics/`)

   - Extensible metric framework
   - Built-in metrics: perplexity, BLEU, ROUGE, exact match
   - Custom metric registration system

4. **Distribution Module** (`rl_onoff/distributions/`)

   - Extract token-wise logits and probabilities
   - Support for both raw logits and normalized probabilities
   - Distribution caching for efficiency

5. **Divergence Module** (`rl_onoff/divergence/`)

   - Token-level JS divergence calculation
   - Token-level KL divergence calculation
   - Batch divergence computation

6. **Switching Module** (`rl_onoff/switching/`)

   - Conditional model switching during generation
   - Threshold-based switching logic
   - Switch-back mechanism to return to model A

7. **Data Utilities** (`rl_onoff/utils/`)

   - Multi-format data loading (JSON, JSONL, CSV, HuggingFace datasets)
   - Data validation and formatting
   - Question-solution pair handling

### Project Structure

```
rl_onoff/
├── rl_onoff/
│   ├── __init__.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py          # Base backend interface
│   │   ├── huggingface.py   # HuggingFace backend
│   │   ├── vllm.py          # vLLM backend
│   │   └── sglang.py        # SGLang backend
│   ├── sampling/
│   │   ├── __init__.py
│   │   └── sampler.py       # Sampling utilities
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── base.py          # Base metric interface
│   │   └── builtin.py       # Built-in metrics
│   ├── distributions/
│   │   ├── __init__.py
│   │   └── extractor.py     # Distribution extraction
│   ├── divergence/
│   │   ├── __init__.py
│   │   └── calculator.py    # JS and KL divergence
│   ├── switching/
│   │   ├── __init__.py
│   │   └── switcher.py      # Model switching logic
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py   # Data loading utilities
│       └── config.py        # Configuration management
├── scripts/
│   ├── sample_and_metrics.py      # CLI: Sample + metrics
│   ├── extract_distributions.py   # CLI: Extract distributions
│   ├── calculate_divergence.py    # CLI: Calculate divergences
│   └── switch_generate.py         # CLI: Conditional switching
├── tests/
│   ├── __init__.py
│   ├── test_backends.py
│   ├── test_sampling.py
│   ├── test_metrics.py
│   ├── test_distributions.py
│   ├── test_divergence.py
│   └── test_switching.py
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

## Implementation Details

### Backend Interface

All backends implement a common interface:

- `load(model_name, **kwargs)` - Load model
- `generate(prompts, **sampling_params)` - Generate text
- `get_logits(prompts, **kwargs)` - Get token logits
- `get_probabilities(prompts, **kwargs)` - Get token probabilities
- `get_tokenizer()` - Get tokenizer for encoding/decoding

### Distribution Extraction

For requirement 2, extract distributions at each token position:

- Input: question + solution tokens
- Output: probability distribution over vocabulary at each solution token position
- Format: `List[Dict[str, float]] `or `numpy.ndarray` with shape `(seq_len, vocab_size)`

### Divergence Calculation

For requirement 3:

- Compute JS divergence: `JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)` where `M = 0.5 * (P + Q)`
- Compute KL divergence: `KL(P||Q) = Σ P(i) * log(P(i) / Q(i))`
- Return per-token divergence values

### Conditional Switching

For requirement 4:

1. Start generating with model A
2. At each token, compute divergence between A and B distributions
3. If divergence > threshold, switch to model B for next token(s)
4. After switching, monitor and switch back to model A when divergence drops below threshold
5. Return final response with switch points logged

## Dependencies

- Python 3.10+
- `torch` - PyTorch for model operations
- `transformers` - HuggingFace transformers
- `vllm` - vLLM backend (optional)
- `sglang` - SGLang backend (optional)
- `numpy` - Numerical operations
- `scipy` - Statistical functions (divergence calculations)
- `datasets` - HuggingFace datasets support
- `pandas` - CSV handling
- `click` or `argparse` - CLI interface
- `tqdm` - Progress bars

## Key Files to Create

1. `rl_onoff/backends/base.py` - Abstract base class for backends
2. `rl_onoff/backends/huggingface.py` - HuggingFace implementation
3. `rl_onoff/backends/vllm.py` - vLLM implementation
4. `rl_onoff/backends/sglang.py` - SGLang implementation
5. `rl_onoff/sampling/sampler.py` - Sampling logic
6. `rl_onoff/metrics/base.py` - Metric interface
7. `rl_onoff/distributions/extractor.py` - Distribution extraction
8. `rl_onoff/divergence/calculator.py` - Divergence computation
9. `rl_onoff/switching/switcher.py` - Model switching logic
10. `scripts/` - Four CLI scripts for each requirement
11. `requirements.txt` - Dependency management
12. `setup.py` - Package installation
13. `README.md` - Documentation with usage examples