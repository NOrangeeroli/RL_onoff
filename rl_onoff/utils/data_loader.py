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
        **kwargs
    ) -> List[Dict]:
        """Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments passed to pandas.read_csv
            
        Returns:
            List of dictionaries with original column names preserved
        """
        df = pd.read_csv(file_path, **kwargs)
        
        # Convert DataFrame to list of dictionaries, preserving all columns as-is
        data = df.to_dict('records')
        
        return data

    @staticmethod
    def load_parquet(
        file_path: Union[str, Path],
        **kwargs
    ) -> List[Dict]:
        """Load data from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional arguments passed to pandas.read_parquet
            
        Returns:
            List of dictionaries with original column names preserved
        """
        df = pd.read_parquet(file_path, **kwargs)
        
        # Convert DataFrame to list of dictionaries, preserving all columns as-is
        data = df.to_dict('records')
        
        return data

    @staticmethod
    def load_huggingface_dataset(
        dataset_name: str,
        split: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """Load data from HuggingFace datasets.
        
        Args:
            dataset_name: Name or path of HuggingFace dataset
            split: Dataset split to load (e.g., 'train', 'test')
            **kwargs: Additional arguments passed to load_dataset
            
        Returns:
            List of dictionaries with original column names preserved
        """
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets not available. Install with: pip install datasets"
            )
        
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        
        # Convert dataset to list of dictionaries, preserving all columns as-is
        data = dataset.to_list()
        
        return data

    @staticmethod
    def load_data(
        file_path: Union[str, Path],
        **kwargs
    ) -> List[Dict]:
        """Auto-detect file format and load data.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional format-specific arguments
            
        Returns:
            List of dictionaries with original column names preserved
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
            return DataLoader.load_csv(file_path, **kwargs)
        
        elif suffix == '.parquet':
            return DataLoader.load_parquet(file_path, **kwargs)
        
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .json, .jsonl, .csv, .parquet"
            )


def load_data(
    file_path: Union[str, Path],
    **kwargs
) -> List[Dict]:
    """Convenience function to load data.
    
    Args:
        file_path: Path to data file
        **kwargs: Additional format-specific arguments
        
    Returns:
        List of dictionaries with original column names preserved
    """
    return DataLoader.load_data(file_path=file_path, **kwargs)

