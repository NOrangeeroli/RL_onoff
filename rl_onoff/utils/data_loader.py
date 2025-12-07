"""Data loading utilities for multiple formats."""

from typing import List, Dict, Optional, Union, Any
import json
import csv
import pandas as pd
from pathlib import Path

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    load_dataset = None


class DataLoader:
    """Loader for multiple data formats."""

    def __init__(self):
        """Initialize data loader."""
        pass

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Union[Dict, List]:
        """Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded data (dict or list)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
        """Load data from JSONL file (one JSON object per line).
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of dictionaries
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    @staticmethod
    def load_csv(
        file_path: Union[str, Path],
        question_column: Optional[str] = None,
        solution_column: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            question_column: Name of column containing questions
            solution_column: Name of column containing solutions
            **kwargs: Additional arguments passed to pandas.read_csv
            
        Returns:
            List of dictionaries with 'question' and 'solution' keys
        """
        df = pd.read_csv(file_path, **kwargs)
        
        # If column names specified, use them; otherwise try to infer
        if question_column and solution_column:
            question_col = question_column
            solution_col = solution_column
        else:
            # Try to find common column names
            question_col = None
            solution_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if question_col is None and any(
                    x in col_lower for x in ['question', 'prompt', 'input', 'query']
                ):
                    question_col = col
                if solution_col is None and any(
                    x in col_lower for x in ['solution', 'answer', 'output', 'response', 'target']
                ):
                    solution_col = col
            
            if question_col is None or solution_col is None:
                # Use first two columns as fallback
                cols = list(df.columns)
                question_col = cols[0] if len(cols) > 0 else None
                solution_col = cols[1] if len(cols) > 1 else None
        
        data = []
        for _, row in df.iterrows():
            item = {
                'question': str(row[question_col]) if question_col else '',
                'solution': str(row[solution_col]) if solution_col else '',
            }
            # Add all other columns as additional metadata
            for col in df.columns:
                if col not in [question_col, solution_col]:
                    item[col] = row[col]
            data.append(item)
        
        return data

    @staticmethod
    def load_huggingface_dataset(
        dataset_name: str,
        split: Optional[str] = None,
        question_column: Optional[str] = None,
        solution_column: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """Load data from HuggingFace datasets.
        
        Args:
            dataset_name: Name or path of HuggingFace dataset
            split: Dataset split to load (e.g., 'train', 'test')
            question_column: Name of column containing questions
            solution_column: Name of column containing solutions
            **kwargs: Additional arguments passed to load_dataset
            
        Returns:
            List of dictionaries with 'question' and 'solution' keys
        """
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets not available. Install with: pip install datasets"
            )
        
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        
        # Convert to list of dicts
        data_dict = dataset.to_dict()
        
        # Infer column names if not specified
        if question_column is None or solution_column is None:
            cols = list(data_dict.keys())
            question_col = question_column
            solution_col = solution_column
            
            if question_col is None:
                for col in cols:
                    if any(x in col.lower() for x in ['question', 'prompt', 'input', 'query']):
                        question_col = col
                        break
                if question_col is None and len(cols) > 0:
                    question_col = cols[0]
            
            if solution_col is None:
                for col in cols:
                    if any(x in col.lower() for x in ['solution', 'answer', 'output', 'response', 'target']):
                        solution_col = col
                        break
                if solution_col is None and len(cols) > 1:
                    solution_col = cols[1]
        else:
            question_col = question_column
            solution_col = solution_column
        
        # Convert to list of dictionaries
        data = []
        num_items = len(data_dict[question_col]) if question_col in data_dict else 0
        
        for i in range(num_items):
            item = {
                'question': str(data_dict[question_col][i]) if question_col in data_dict else '',
                'solution': str(data_dict[solution_col][i]) if solution_col in data_dict else '',
            }
            # Add other columns as metadata
            for col, values in data_dict.items():
                if col not in [question_col, solution_col]:
                    item[col] = values[i]
            data.append(item)
        
        return data

    @staticmethod
    def load_data(
        file_path: Union[str, Path],
        question_column: Optional[str] = None,
        solution_column: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """Auto-detect file format and load data.
        
        Args:
            file_path: Path to data file
            question_column: Name of column containing questions (for CSV/datasets)
            solution_column: Name of column containing solutions (for CSV/datasets)
            **kwargs: Additional format-specific arguments
            
        Returns:
            List of dictionaries with 'question' and 'solution' keys
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            data = DataLoader.load_json(file_path)
            # If it's a list, use directly; if dict, try to extract list
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Try common keys
                for key in ['data', 'items', 'examples', 'samples']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # If no list found, wrap in list
                return [data]
            else:
                return []
        
        elif suffix == '.jsonl':
            return DataLoader.load_jsonl(file_path)
        
        elif suffix == '.csv':
            return DataLoader.load_csv(
                file_path,
                question_column=question_column,
                solution_column=solution_column,
                **kwargs
            )
        
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .json, .jsonl, .csv"
            )


def load_data(
    file_path: Union[str, Path],
    question_column: Optional[str] = None,
    solution_column: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """Convenience function to load data.
    
    Args:
        file_path: Path to data file
        question_column: Name of column containing questions
        solution_column: Name of column containing solutions
        **kwargs: Additional format-specific arguments
        
    Returns:
        List of dictionaries with 'question' and 'solution' keys
    """
    return DataLoader.load_data(
        file_path=file_path,
        question_column=question_column,
        solution_column=solution_column,
        **kwargs
    )

