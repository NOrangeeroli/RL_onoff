"""Tests for BaseFormat."""

import pytest
from typing import Dict, Optional
from rl_onoff.tasks.formats.base import BaseFormat


class ConcreteFormat(BaseFormat):
    """Concrete implementation for testing."""
    
    def get_system_prompt(self) -> str:
        return "Test system prompt"
    
    def extract(self, response: str) -> Dict[str, Optional[str]]:
        return {"answer": "test_answer", "reasoning": "test_reasoning"}


class TestBaseFormat:
    """Tests for BaseFormat abstract class."""
    
    def test_init_default_name(self):
        """Test initialization with default name."""
        format_obj = ConcreteFormat()
        assert format_obj.name == "ConcreteFormat"
    
    def test_init_custom_name(self):
        """Test initialization with custom name."""
        format_obj = ConcreteFormat(name="custom")
        assert format_obj.name == "custom"
    
    def test_get_system_prompt(self):
        """Test get_system_prompt method."""
        format_obj = ConcreteFormat()
        prompt = format_obj.get_system_prompt()
        assert prompt == "Test system prompt"
    
    def test_extract(self):
        """Test extract method."""
        format_obj = ConcreteFormat()
        result = format_obj.extract("test response")
        assert isinstance(result, dict)
        assert "answer" in result
        assert result["answer"] == "test_answer"
        assert "reasoning" in result
    
    def test_repr(self):
        """Test __repr__ method."""
        format_obj = ConcreteFormat()
        repr_str = repr(format_obj)
        assert "ConcreteFormat" in repr_str
        assert "name=" in repr_str

