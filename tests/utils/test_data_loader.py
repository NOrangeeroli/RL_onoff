"""Tests for DataLoader utility class."""

import pytest
import json
import tempfile
import os
import csv

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
    
    def test_load_csv_with_columns(self, sample_csv_file):
        """Test loading CSV with specified columns."""
        data = DataLoader.load_csv(
            sample_csv_file,
            question_column="question",
            solution_column="solution"
        )
        assert len(data) == 2
        assert "question" in data[0]
        assert "solution" in data[0]
    
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

