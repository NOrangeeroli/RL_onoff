"""Data loading utilities for multiple formats."""

from typing import List, Dict, Optional, Union, Any
import json
import csv
import pandas as pd
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    load_dataset = None


def _convert_numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy arrays and scalars to Python native types.
    
    Args:
        obj: Object that may contain numpy arrays
        
    Returns:
        Object with numpy arrays converted to Python lists
    """
    if NUMPY_AVAILABLE and np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
    
    if isinstance(obj, dict):
        return {key: _convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_python(item) for item in obj]
    
    return obj


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
        
        # Convert numpy arrays to Python native types
        data = [_convert_numpy_to_python(item) for item in data]
        
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
        
        # Convert numpy arrays to Python native types
        data = [_convert_numpy_to_python(item) for item in data]
        
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


if __name__ == "__main__":
    """Simple use cases for DataLoader."""
    
    import tempfile
    import os
    
    print("=" * 60)
    print("DataLoader Use Cases")
    print("=" * 60)
    
    # Example 1: Load JSON file
    print("\nExample 1: Load JSON file")
    print("-" * 60)
    
    # Create a temporary JSON file
    json_data = [
        {"question": "What is 2+2?", "solution": "4"},
        {"question": "What is 3+3?", "solution": "6"}
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_data, f)
        json_file = f.name
    
    try:
        data = DataLoader.load_json(json_file)
        print(f"Loaded {len(data)} items from JSON")
        print(f"First item: {data[0]}")
    finally:
        os.unlink(json_file)
    
    # Example 2: Load JSONL file
    print("\nExample 2: Load JSONL file")
    print("-" * 60)
    
    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in json_data:
            f.write(json.dumps(item) + '\n')
        jsonl_file = f.name
    
    try:
        data = DataLoader.load_jsonl(jsonl_file)
        print(f"Loaded {len(data)} items from JSONL")
        print(f"First item: {data[0]}")
    finally:
        os.unlink(jsonl_file)
    
    # Example 3: Load CSV file
    print("\nExample 3: Load CSV file")
    print("-" * 60)
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "solution"])
        writer.writeheader()
        writer.writerows(json_data)
        csv_file = f.name
    
    try:
        data = DataLoader.load_csv(csv_file)
        print(f"Loaded {len(data)} items from CSV")
        print(f"First item: {data[0]}")
        print(f"Column names preserved: {list(data[0].keys())}")
    finally:
        os.unlink(csv_file)
    
    # Example 4: Load Parquet file (if pandas is available)
    print("\nExample 4: Load Parquet file")
    print("-" * 60)
    
    try:
        # Create a temporary Parquet file
        df = pd.DataFrame(json_data)
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            parquet_file = f.name
        
        df.to_parquet(parquet_file, index=False)
        
        try:
            data = DataLoader.load_parquet(parquet_file)
            print(f"Loaded {len(data)} items from Parquet")
            print(f"First item: {data[0]}")
            print(f"Column names preserved: {list(data[0].keys())}")
        finally:
            os.unlink(parquet_file)
    except Exception as e:
        print(f"Parquet example skipped: {e}")
    
    # Example 5: Auto-detect file format
    print("\nExample 5: Auto-detect file format")
    print("-" * 60)
    
    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_data, f)
        auto_file = f.name
    
    try:
        # Use convenience function with auto-detection
        data = load_data(auto_file)
        print(f"Auto-detected format and loaded {len(data)} items")
        print(f"First item: {data[0]}")
    finally:
        os.unlink(auto_file)
    
    # Example 6: Load HuggingFace dataset (if available)
    print("\nExample 6: Load HuggingFace dataset")
    print("-" * 60)
    
    if DATASETS_AVAILABLE:
        try:
            # Load a small dataset for demonstration
            data = DataLoader.load_huggingface_dataset(
                dataset_name="HuggingFaceH4/aime_2024",
                split="train[:5]"  # Only first 5 examples
            )
            print(f"Loaded {len(data)} items from HuggingFace dataset")
            if len(data) > 0:
                print(f"First item keys: {list(data[0].keys())}")
                print(f"Sample keys: {list(data[0].keys())[:3]}")
        except Exception as e:
            print(f"HuggingFace dataset example skipped: {e}")
    else:
        print("HuggingFace datasets not available. Install with: pip install datasets")
    
    # Example 7: Handle different data structures
    print("\nExample 7: Handle different JSON structures")
    print("-" * 60)
    
    # JSON with nested structure
    nested_json = {
        "data": [
            {"question": "What is 2+2?", "solution": "4"},
            {"question": "What is 3+3?", "solution": "6"}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(nested_json, f)
        nested_file = f.name
    
    try:
        data = load_data(nested_file)
        print(f"Loaded {len(data)} items from nested JSON structure")
        print(f"Auto-extracted from 'data' key")
    finally:
        os.unlink(nested_file)
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
