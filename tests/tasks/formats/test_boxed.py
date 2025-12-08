"""Tests for BoxedFormat."""

import pytest
from rl_onoff.tasks.formats.boxed import BoxedFormat


class TestBoxedFormat:
    """Tests for BoxedFormat."""
    
    def test_init(self):
        """Test initialization."""
        format_obj = BoxedFormat()
        assert format_obj.name == "boxed"
    
    def test_get_system_prompt(self):
        """Test get_system_prompt returns correct prompt."""
        format_obj = BoxedFormat()
        prompt = format_obj.get_system_prompt()
        assert "boxed" in prompt.lower()
        assert "\\boxed" in prompt or "boxed" in prompt
    
    def test_extract_with_boxed_answer(self):
        """Test extract with proper boxed format."""
        format_obj = BoxedFormat()
        response = "Let me solve this step by step.\nFirst, I calculate...\n\\boxed{42}"
        result = format_obj.extract(response)
        
        assert "answer" in result
        assert result["answer"] == "42"
        assert "reasoning" in result
        assert "Let me solve" in result["reasoning"]
    
    def test_extract_with_answer_pattern(self):
        """Test extract with answer pattern fallback."""
        format_obj = BoxedFormat()
        response = "Let me solve this.\nThe answer is 42."
        result = format_obj.extract(response)
        
        assert "answer" in result
        assert result["answer"] == "42"
        assert "reasoning" in result
    
    def test_extract_no_answer(self):
        """Test extract when no answer is found."""
        format_obj = BoxedFormat()
        response = "Let me solve this step by step."
        result = format_obj.extract(response)
        
        assert "answer" in result
        assert result["answer"] is None
        assert "reasoning" in result
        assert result["reasoning"] == response
    
    def test_extract_empty_response(self):
        """Test extract with empty response."""
        format_obj = BoxedFormat()
        result = format_obj.extract("")
        
        assert "answer" in result
        assert result["answer"] is None
        assert "reasoning" in result
    
    def test_extract_multiple_boxed(self):
        """Test extract when multiple boxed answers exist (should get first)."""
        format_obj = BoxedFormat()
        response = "First: \\boxed{10}\nSecond: \\boxed{20}"
        result = format_obj.extract(response)
        
        assert "answer" in result
        assert result["answer"] == "10"  # Should get first one
        assert "reasoning" in result

