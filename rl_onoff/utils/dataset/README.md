# Dataset Module

This module provides dataset classes for loading and accessing different datasets in a unified format.

## Overview

Each dataset class:
1. Knows the file path structure for its dataset
2. Loads data using the `data_loader` utility
3. Extracts question, answer, and solution strings, normalizing different column names/formats
4. Returns tuples of (question, answer, solution) where solution may be None if not available

## Available Datasets

### AIME2025Dataset
- **File**: `data/aime2025/test.parquet`
- **Question**: `prompt.content`
- **Answer**: `reward_model.ground_truth.target`
- **Solution**: `reward_model.ground_truth.solution` (may be None)

### AMC23Dataset
- **File**: `data/amc23/test.parquet`
- **Question**: `prompt.content`
- **Answer**: `reward_model.ground_truth.target`
- **Solution**: `reward_model.ground_truth.solution` (may be None)

### GSM8KLevel1Dataset
- **Files**: `data/gsm8k_level1/train.parquet`, `data/gsm8k_level1/test.parquet`
- **Question**: `extra_info.question` (fallback: `prompt.content`)
- **Answer**: `extra_info.answer` (fallback: `reward_model.ground_truth`)
- **Solution**: `reward_model.ground_truth.solution` (may be None)
- **Supports**: `split="train"` or `split="test"`

### MathDataset
- **Files**: `data/math/train.parquet`, `data/math/test.parquet`
- **Question**: `prompt.content`
- **Answer**: `reward_model.ground_truth.target`
- **Solution**: `reward_model.ground_truth.solution` (may be None)
- **Supports**: `split="train"` or `split="test"`

## Usage

### Basic Usage

```python
from rl_onoff.utils.dataset import MathDataset, GSM8KLevel1Dataset

# Load MATH test set
math_test = MathDataset(split="test")
question, answer, solution = math_test[0]
print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"Solution: {solution}")  # May be None

# Load GSM8K Level 1 train set
gsm8k_train = GSM8KLevel1Dataset(split="train")
print(f"Dataset size: {len(gsm8k_train)}")
```

### Accessing All Data

```python
from rl_onoff.utils.dataset import AIME2025Dataset

dataset = AIME2025Dataset()

# Get all question-answer-solution triples
all_triples = dataset.get_all()  # List of (question, answer, solution) tuples

# Get just questions
questions = dataset.get_questions()  # List of question strings

# Get just answers
answers = dataset.get_answers()  # List of answer strings

# Get just solutions
solutions = dataset.get_solutions()  # List of solution strings (may contain None)
```

### Custom Data Directory

```python
from pathlib import Path
from rl_onoff.utils.dataset import MathDataset

# Specify custom data directory
custom_data_dir = Path("/path/to/your/data")
dataset = MathDataset(data_dir=custom_data_dir, split="test")
```

## BaseDataset Interface

All dataset classes inherit from `BaseDataset`, which provides:

- `load()`: Load the dataset from file
- `__len__()`: Get the number of examples
- `__getitem__(idx)`: Get a (question, answer, solution) tuple by index
- `get_all()`: Get all question-answer-solution triples
- `get_questions()`: Get all questions
- `get_answers()`: Get all answers
- `get_solutions()`: Get all solutions (may contain None)
- `extract_solution(entry)`: Extract solution from a data entry (returns None if not available)

## Creating New Dataset Classes

To create a new dataset class, inherit from `BaseDataset` and implement:

1. `get_file_path()`: Return the path to the data file
2. `extract_question(entry)`: Extract question string from a data entry
3. `extract_answer(entry)`: Extract answer string from a data entry

The `extract_solution(entry)` method is already implemented in the base class and tries to extract
`reward_model.ground_truth.solution`. You can override it if your dataset has a different structure.

Example:

```python
from rl_onoff.utils.dataset.base import BaseDataset
from pathlib import Path
from typing import Dict

class MyDataset(BaseDataset):
    def get_file_path(self) -> Path:
        return self.data_dir / "data" / "mydataset" / f"{self.split}.parquet"
    
    def extract_question(self, entry: Dict) -> str:
        return entry.get("question", "")
    
    def extract_answer(self, entry: Dict) -> str:
        return entry.get("answer", "")
```

