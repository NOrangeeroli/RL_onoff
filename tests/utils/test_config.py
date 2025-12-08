"""Tests for Config utility class."""

import pytest
import json
import tempfile
import os
from dataclasses import dataclass

from rl_onoff.utils.config import Config


class TestConfig:
    """Tests for Config utility."""
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        @dataclass
        class TestConfig(Config):
            value1: int = 1
            value2: str = "test"
        
        config = TestConfig(value1=42, value2="hello")
        config_dict = config.to_dict()
        
        assert config_dict["value1"] == 42
        assert config_dict["value2"] == "hello"
    
    def test_config_to_json(self):
        """Test saving config to JSON."""
        @dataclass
        class TestConfig(Config):
            value: int = 1
        
        config = TestConfig(value=42)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.to_json(temp_path)
            
            # Read it back
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded["value"] == 42
        finally:
            os.unlink(temp_path)
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        @dataclass
        class TestConfig(Config):
            value: int = 1
        
        data = {"value": 42}
        config = TestConfig.from_dict(data)
        
        assert config.value == 42
    
    def test_config_from_json(self):
        """Test loading config from JSON."""
        @dataclass
        class TestConfig(Config):
            value: int = 1
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"value": 42}, f)
            temp_path = f.name
        
        try:
            config = TestConfig.from_json(temp_path)
            assert config.value == 42
        finally:
            os.unlink(temp_path)

