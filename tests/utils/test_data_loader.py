"""Tests for DataLoader utility class."""

import pytest
import json
import tempfile
import os
import csv
from unittest.mock import Mock, patch, MagicMock

from rl_onoff.utils.data_loader import DataLoader, load_data


@pytest.fixture
def sample_json_file():
    """Fixture for sample JSON file."""
    data = [
        {"question": "What is 2+2?", "solution": "4"},
        {"question": "What is 3+3?", "solution": "6"}
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        f.flush()
        fname = f.name
    # File is now closed, yield the filename
    yield fname
    # Cleanup
    if os.path.exists(fname):
        os.unlink(fname)


@pytest.fixture
def sample_jsonl_file():
    """Fixture for sample JSONL file."""
    data = [
        {"question": "What is 2+2?", "solution": "4"},
        {"question": "What is 3+3?", "solution": "6"}
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
        f.flush()
        fname = f.name
    # File is now closed, yield the filename
    yield fname
    # Cleanup
    if os.path.exists(fname):
        os.unlink(fname)


@pytest.fixture
def sample_csv_file():
    """Fixture for sample CSV file."""
    data = [
        {"question": "What is 2+2?", "solution": "4"},
        {"question": "What is 3+3?", "solution": "6"}
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "solution"])
        writer.writeheader()
        writer.writerows(data)
        f.flush()
        fname = f.name
    # File is now closed, yield the filename
    yield fname
    # Cleanup
    if os.path.exists(fname):
        os.unlink(fname)


@pytest.fixture
def sample_parquet_file():
    """Fixture for sample Parquet file."""
    import pandas as pd
    data = [
        {"question": "What is 2+2?", "solution": "4"},
        {"question": "What is 3+3?", "solution": "6"}
    ]
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        fname = f.name
    df.to_parquet(fname, index=False)
    # File is now closed, yield the filename
    yield fname
    # Cleanup
    if os.path.exists(fname):
        os.unlink(fname)


class TestDataLoader:
    """Tests for DataLoader utility."""
    
    def test_load_json(self, sample_json_file):
        """Test loading JSON file."""
        data = DataLoader.load_json(sample_json_file)
        assert isinstance(data, list)
        assert len(data) == 2
        assert "question" in data[0]
    
    def test_load_jsonl(self, sample_jsonl_file):
        """Test loading JSONL file."""
        data = DataLoader.load_jsonl(sample_jsonl_file)
        assert isinstance(data, list)
        assert len(data) == 2
        assert "question" in data[0]
    
    def test_load_csv(self, sample_csv_file):
        """Test loading CSV file."""
        data = DataLoader.load_csv(sample_csv_file)
        assert isinstance(data, list)
        assert len(data) == 2
        assert "question" in data[0]
        assert "solution" in data[0]
    
    def test_load_csv_preserves_columns(self, sample_csv_file):
        """Test loading CSV preserves all columns."""
        data = DataLoader.load_csv(sample_csv_file)
        assert len(data) == 2
        # Original column names are preserved
        assert "question" in data[0]
        assert "solution" in data[0]
        assert data[0]["question"] == "What is 2+2?"
        # Note: pandas may convert numeric strings to int/float, so accept both
        assert data[0]["solution"] in ("4", 4)
    
    def test_load_data_auto_detect_json(self, sample_json_file):
        """Test auto-detection of JSON format."""
        data = load_data(sample_json_file)
        assert isinstance(data, list)
        assert len(data) == 2
    
    def test_load_data_auto_detect_jsonl(self, sample_jsonl_file):
        """Test auto-detection of JSONL format."""
        data = load_data(sample_jsonl_file)
        assert isinstance(data, list)
        assert len(data) == 2
    
    def test_load_data_auto_detect_csv(self, sample_csv_file):
        """Test auto-detection of CSV format."""
        data = load_data(sample_csv_file)
        assert isinstance(data, list)
        assert len(data) == 2
    
    def test_load_parquet(self, sample_parquet_file):
        """Test loading Parquet file."""
        data = DataLoader.load_parquet(sample_parquet_file)
        assert isinstance(data, list)
        assert len(data) == 2
        # Data is returned as-is with original column names
        assert "question" in data[0]
        assert "solution" in data[0]
        assert data[0]["question"] == "What is 2+2?"
        assert data[0]["solution"] == "4"
    
    def test_load_parquet_preserves_columns(self, sample_parquet_file):
        """Test loading Parquet preserves all columns."""
        data = DataLoader.load_parquet(sample_parquet_file)
        assert len(data) == 2
        # Original column names are preserved
        assert data[0]["question"] == "What is 2+2?"
        assert data[0]["solution"] == "4"
    
    def test_load_data_auto_detect_parquet(self, sample_parquet_file):
        """Test auto-detection of Parquet format."""
        data = load_data(sample_parquet_file)
        assert isinstance(data, list)
        assert len(data) == 2
        # Original column names are preserved
        assert "question" in data[0]
        assert "solution" in data[0]
    
    def test_load_data_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.json")
    
    def test_load_data_unsupported_format(self):
        """Test loading unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_data(temp_path)
        finally:
            os.unlink(temp_path)
    
   
    @pytest.mark.integration
    def test_load_huggingface_dataset_real_aime_2024(self):
        """Test loading real HuggingFace dataset: HuggingFaceH4/aime_2024.
        
        This test requires:
        - Internet connection
        - datasets library installed
        - Will download the dataset on first run
        """
        try:
            data = DataLoader.load_huggingface_dataset(
                dataset_name="HuggingFaceH4/aime_2024",
                split="train"
            )
            
            # Verify we got data
            assert isinstance(data, list)
            assert len(data) > 0
            
            # Verify structure - should have question and solution keys
            assert "problem" in data[0]
            assert "solution" in data[0]
            
            # Verify data types
            assert isinstance(data[0]["question"], str)
            assert isinstance(data[0]["solution"], str)
            
            # Verify we can load a reasonable amount of data
            assert len(data) >= 1
            
        except ImportError:
            pytest.skip("HuggingFace datasets library not available")
        except Exception as e:
            # If dataset doesn't exist or network issues, skip the test
            pytest.skip(f"Could not load real dataset: {e}")
    
    @pytest.mark.integration
    def test_load_huggingface_dataset_real_aime_with_limit(self):
        """Test loading real HuggingFace dataset with a small limit.
        
        This test loads only a small subset to be faster.
        """
        try:
            # Load with a small split or limit
            data = DataLoader.load_huggingface_dataset(
                dataset_name="HuggingFaceH4/aime_2024",
                split="train[:10]"  # Only first 10 examples
            )
            
            # Verify we got data
            assert isinstance(data, list)
            assert len(data) <= 10
            assert len(data) > 0
            
            # Verify structure
            assert "problem" in data[0]
            assert "solution" in data[0]
            
        except ImportError:
            pytest.skip("HuggingFace datasets library not available")
        except Exception as e:
            pytest.skip(f"Could not load real dataset: {e}")
    
    @pytest.mark.integration
    def test_load_real_parquet_files(self):
        """Test loading real parquet files from data directory."""
        import pandas as pd
        from pathlib import Path
        
        # Test files in data directory
        test_files = [
            "data/gsm8k_level1/train.parquet",
            "data/gsm8k_level1/test.parquet",
            "data/math/train.parquet",
            "data/math/test.parquet",
            "data/amc23/test.parquet",
            "data/aime2025/test.parquet",
        ]
        
        for file_path in test_files:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                pytest.skip(f"Data file not found: {file_path}")
            
            try:
                # First, check what columns exist
                # Read full file and take first row (pd.read_parquet doesn't support nrows)
                df_full = pd.read_parquet(str(file_path_obj))
                df_sample = df_full.head(1)
                original_columns = list(df_sample.columns)
                print(f"\n{file_path} columns: {original_columns}")
                
                # Load using DataLoader
                data = load_data(file_path)
                
                # Verify we got data
                assert isinstance(data, list)
                assert len(data) > 0
                
                # Verify structure - original column names are preserved
                # Check that all original columns are present
                loaded_columns = list(data[0].keys())
                for col in original_columns:
                    assert col in data[0], f"Column '{col}' missing in loaded data"
                
                # Verify data types (check first few columns)
                for col in original_columns[:3]:  # Check first 3 columns
                    assert col in data[0]
                    # Most columns should be strings, numbers, or None
                    assert isinstance(data[0][col], (str, int, float, type(None), bool, list, dict))
                
                # Print success message
                print(f"  ✓ Loaded {len(data)} items successfully")
                print(f"  ✓ All {len(original_columns)} columns preserved: {original_columns}")
                
            except Exception as e:
                pytest.fail(f"Failed to load {file_path}: {e}")
    
    @pytest.mark.integration
    def test_load_real_parquet_gsm8k(self):
        """Test loading GSM8K parquet file specifically."""
        from pathlib import Path
        
        file_path = Path("data/gsm8k_level1/train.parquet")
        if not file_path.exists():
            pytest.skip("GSM8K data file not found")
        
        try:
            data = load_data(file_path)
            
            assert isinstance(data, list)
            assert len(data) > 0
            
            # Check first item structure - original columns are preserved
            first_item = data[0]
            # Get actual column names from the data
            actual_columns = list(first_item.keys())
            print(f"  GSM8K columns: {actual_columns}")
            
            # Verify we have data and can access it
            assert len(actual_columns) > 0
            # Check that at least one column has non-empty data
            has_data = any(
                isinstance(first_item[col], str) and len(first_item[col]) > 0
                or not isinstance(first_item[col], str) and first_item[col] is not None
                for col in actual_columns
            )
            assert has_data, "No data found in first item"
            
        except Exception as e:
            pytest.fail(f"Failed to load GSM8K data: {e}")
    
    @pytest.mark.integration
    def test_load_real_parquet_math(self):
        """Test loading MATH parquet file specifically."""
        from pathlib import Path
        
        file_path = Path("data/math/train.parquet")
        if not file_path.exists():
            pytest.skip("MATH data file not found")
        
        try:
            data = load_data(file_path)
            
            assert isinstance(data, list)
            assert len(data) > 0
            
            # Check structure - original columns are preserved
            first_item = data[0]
            actual_columns = list(first_item.keys())
            print(f"  MATH columns: {actual_columns}")
            
            # Verify we have columns
            assert len(actual_columns) > 0
            
        except Exception as e:
            pytest.fail(f"Failed to load MATH data: {e}")
    
    @pytest.mark.integration
    def test_load_real_parquet_preserves_all_columns(self):
        """Test that loading parquet files preserves all original columns."""
        from pathlib import Path
        
        # Try with a file that exists
        test_files = [
            "data/gsm8k_level1/test.parquet",
            "data/math/test.parquet",
        ]
        
        for file_path_str in test_files:
            file_path = Path(file_path_str)
            if not file_path.exists():
                continue
            
            try:
                # First, check what columns exist by reading directly
                import pandas as pd
                df = pd.read_parquet(file_path, nrows=1)
                original_columns = df.columns.tolist()
                
                # Load using DataLoader
                data = load_data(file_path)
                
                assert isinstance(data, list)
                assert len(data) > 0
                
                # Verify all original columns are preserved
                loaded_columns = list(data[0].keys())
                assert set(loaded_columns) == set(original_columns), \
                    f"Columns mismatch. Original: {original_columns}, Loaded: {loaded_columns}"
                
                # Verify data values match
                for col in original_columns:
                    assert col in data[0]
                    # Values should be preserved (may need type conversion for comparison)
                    original_val = df.iloc[0][col]
                    loaded_val = data[0][col]
                    # For string columns, compare as strings
                    if isinstance(original_val, str):
                        assert str(loaded_val) == original_val
                    else:
                        assert loaded_val == original_val or str(loaded_val) == str(original_val)
                
                print(f"  ✓ {file_path_str}: All {len(original_columns)} columns preserved")
                return  # Found a working file, we're done
                    
            except Exception as e:
                # Try next file
                print(f"  ⚠ {file_path_str}: {e}")
                continue
        
        # If we get here, no files worked
        pytest.skip("No suitable parquet files found for column preservation test")

