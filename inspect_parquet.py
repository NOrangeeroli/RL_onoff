#!/usr/bin/env python3
"""Simple script to inspect parquet file columns."""

import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    print("pyarrow not available, trying pandas...")
    import pandas as pd
    pyarrow_available = False
else:
    pyarrow_available = True

def inspect_file(file_path):
    """Inspect a single parquet file."""
    print(f"\n📁 File: {file_path}")
    print("-" * 80)
    
    try:
        if pyarrow_available:
            # Fast method: read schema only
            parquet_file = pq.ParquetFile(str(file_path))
            schema = parquet_file.schema_arrow
            columns = [field.name for field in schema]
            num_rows = parquet_file.metadata.num_rows
            print(f"  Columns ({len(columns)}): {columns}")
            print(f"  Number of rows: {num_rows:,}")
        else:
            # Fallback: use pandas (slower but works)
            df = pd.read_parquet(str(file_path), nrows=0)
            print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Find all parquet files
parquet_files = sorted(Path('data').rglob('*.parquet'))

print("=" * 80)
print("PARQUET FILES COLUMN ANALYSIS")
print("=" * 80)

for file_path in parquet_files:
    inspect_file(file_path)

print("\n" + "=" * 80)

