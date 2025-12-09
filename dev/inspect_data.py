#!/usr/bin/env python3
"""Inspect all datasets in the data folder.

This script recursively finds all data files in the data/ folder,
loads each one, and prints the first entry.
"""

import json
import sys
from pathlib import Path
from typing import List

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl_onoff.utils.data_loader import load_data


def find_data_files(data_dir: Path) -> List[Path]:
    """Find all supported data files in the directory tree.
    
    Args:
        data_dir: Root directory to search
        
    Returns:
        List of file paths
    """
    supported_extensions = {'.json', '.jsonl', '.csv', '.parquet'}
    data_files = []
    
    for ext in supported_extensions:
        data_files.extend(data_dir.rglob(f'*{ext}'))
    
    # Sort for consistent output
    return sorted(data_files)


def format_entry(entry: dict, max_length: int = 200) -> str:
    """Format a data entry for printing.
    
    Args:
        entry: Dictionary entry to format
        max_length: Maximum length for string values
        
    Returns:
        Formatted string representation
    """
    formatted = {}
    for key, value in entry.items():
        if isinstance(value, str):
            if len(value) > max_length:
                formatted[key] = value[:max_length] + "..."
            else:
                formatted[key] = value
        else:
            formatted[key] = value
    
    return json.dumps(formatted, indent=2, ensure_ascii=False)


def main():
    """Main function to inspect all datasets."""
    # Use parent directory since script is in dev/
    data_dir = Path(__file__).parent.parent / "data"
    
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found!")
        return
    
    print("=" * 80)
    print("Dataset Inspection")
    print("=" * 80)
    print(f"Searching for data files in: {data_dir.absolute()}\n")
    
    # Find all data files
    data_files = find_data_files(data_dir)
    
    if not data_files:
        print("No data files found!")
        return
    
    print(f"Found {len(data_files)} data file(s)\n")
    
    # Process each file
    for i, file_path in enumerate(data_files, 1):
        print("=" * 80)
        print(f"Dataset {i}/{len(data_files)}: {file_path.relative_to(data_dir.parent)}")
        print("=" * 80)
        print(f"Full path: {file_path.absolute()}")
        
        try:
            # For parquet files, use pyarrow for efficient reading
            if file_path.suffix.lower() == '.parquet':
                try:
                    import pyarrow.parquet as pq
                    # Read metadata first (very fast, no data loaded)
                    parquet_file = pq.ParquetFile(str(file_path))
                    total_rows = parquet_file.metadata.num_rows
                    
                    # Read just the first row (very efficient)
                    table = parquet_file.read_row_groups([0], columns=None)
                    if len(table) == 0:
                        print("⚠️  Warning: Dataset is empty!\n")
                        continue
                    
                    # Get first row as dict
                    first_row = table.slice(0, 1)
                    import pandas as pd
                    df = first_row.to_pandas()
                    first_entry = df.iloc[0].to_dict()
                    
                    # Convert numpy types to Python types
                    first_entry = {
                        k: (v.item() if hasattr(v, 'item') and not isinstance(v, str) else v) 
                        for k, v in first_entry.items()
                    }
                    
                    print(f"\nTotal entries: {total_rows:,}")
                    print(f"Columns/Keys ({len(first_entry)}): {list(first_entry.keys())}")
                    
                    # Print first entry
                    print("\n" + "-" * 80)
                    print("First Entry:")
                    print("-" * 80)
                    print(format_entry(first_entry))
                    print()
                except ImportError:
                    # Fallback to pandas if pyarrow not available
                    import pandas as pd
                    df = pd.read_parquet(str(file_path), nrows=1)
                    if len(df) == 0:
                        print("⚠️  Warning: Dataset is empty!\n")
                        continue
                    
                    # Get total count (might be slow for large files)
                    try:
                        # Try to get row count from file size estimate
                        import os
                        file_size = os.path.getsize(file_path)
                        # Rough estimate: assume ~1KB per row (very rough)
                        total_rows = f"~{file_size // 1024:,} (estimated)"
                    except:
                        total_rows = "unknown"
                    
                    first_entry = df.iloc[0].to_dict()
                    first_entry = {
                        k: (v.item() if hasattr(v, 'item') and not isinstance(v, str) else v) 
                        for k, v in first_entry.items()
                    }
                    
                    print(f"\nTotal entries: {total_rows}")
                    print(f"Columns/Keys ({len(first_entry)}): {list(first_entry.keys())}")
                    
                    print("\n" + "-" * 80)
                    print("First Entry:")
                    print("-" * 80)
                    print(format_entry(first_entry))
                    print()
            else:
                # For other formats, load normally
                data = load_data(file_path)
                
                if not data:
                    print("⚠️  Warning: Dataset is empty!\n")
                    continue
                
                print(f"\nTotal entries: {len(data):,}")
                print(f"Columns/Keys ({len(data[0])}): {list(data[0].keys())}")
                
                # Print first entry
                print("\n" + "-" * 80)
                print("First Entry:")
                print("-" * 80)
                print(format_entry(data[0]))
                print()
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print(f"   Exception type: {type(e).__name__}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print("=" * 80)
    print("Inspection complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

