# Dataset Module

This module provides dataset classes for loading and accessing different datasets in a unified format.

## Overview

Each dataset class:
1. Knows the file path structure for its dataset
2. Loads data using the `data_loader` utility
3. Extracts question and answer strings, normalizing different column names/formats

## Available Datasets

### AIME2025Dataset
- **File**: `data/aime2025/test.parquet`
- **Question**: `prompt.content`
- **Answer**: `reward_model.ground_truth.target`

### AMC23Dataset
- **File**: `data/amc23/test.parquet`
- **Question**: `prompt.content`
- **Answer**: `reward_model.ground_truth.target`

### GSM8KLevel1Dataset
- **Files**: `data/gsm8k_level1/train.parquet`, `data/gsm8k_level1/test.parquet`
- **Question**: `extra_info.question` (fallback: `prompt.content`)
- **Answer**: `extra_info.answer` (fallback: `reward_model.ground_truth`)
- **Supports**: `split="train"` or `split="test"`

### MathDataset
- **Files**: `data/math/train.parquet`, `data/math/test.parquet`
- **Question**: `prompt.content`
- **Answer**: `reward_model.ground_truth.target`
- **Supports**: `split="train"` or `split="test"`

## Usage

### Basic Usage

```python
from rl_onoff.utils.dataset import MathDataset, GSM8KLevel1Dataset

# Load MATH test set
math_test = MathDataset(split="test")
question, answer = math_test[0]
print(f"Question: {question}")
print(f"Answer: {answer}")

# Load GSM8K Level 1 train set
gsm8k_train = GSM8KLevel1Dataset(split="train")
print(f"Dataset size: {len(gsm8k_train)}")
```

### Accessing All Data

```python
from rl_onoff.utils.dataset import AIME2025Dataset

dataset = AIME2025Dataset()

# Get all question-answer pairs
all_pairs = dataset.get_all()  # List of (question, answer) tuples

# Get just questions
questions = dataset.get_questions()  # List of question strings

# Get just answers
answers = dataset.get_answers()  # List of answer strings
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
- `__getitem__(idx)`: Get a (question, answer) tuple by index
- `get_all()`: Get all question-answer pairs
- `get_questions()`: Get all questions
- `get_answers()`: Get all answers

## Creating New Dataset Classes

To create a new dataset class, inherit from `BaseDataset` and implement:

1. `get_file_path()`: Return the path to the data file
2. `extract_question(entry)`: Extract question string from a data entry
3. `extract_answer(entry)`: Extract answer string from a data entry

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

