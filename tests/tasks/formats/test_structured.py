"""Tests for StructuredFormat."""

import pytest
from rl_onoff.tasks.formats.structured import StructuredFormat


class TestStructuredFormat:
    """Tests for StructuredFormat."""
    
    def test_init(self):
        """Test initialization."""
        format_obj = StructuredFormat()
        assert format_obj.name == "structured"
    
    def test_get_system_prompt(self):
        """Test get_system_prompt returns correct prompt."""
        format_obj = StructuredFormat()
        prompt = format_obj.get_system_prompt()
        assert "<think>" in prompt or "redacted_reasoning" in prompt.lower()
        assert "<answer>" in prompt.lower()
    
    def test_extract_with_proper_tags(self):
        """Test extract with proper structured format."""
        format_obj = StructuredFormat()
        response = """<think>
Let me solve this step by step.
First, I calculate 2 + 2 = 4.
</think>
<answer>
4
</answer>"""
        result = format_obj.extract(response)
        
        assert "answer" in result
        assert result["answer"] == "4"
        assert "reasoning" in result
        assert "Let me solve" in result["reasoning"]
    
    def test_extract_with_redacted_reasoning_tag(self):
        """Test extract with redacted_reasoning tag."""
        format_obj = StructuredFormat()
        response = """<think>
Step by step solution here.
</think>
<answer>
42
</answer>"""
        result = format_obj.extract(response)
        
        assert "answer" in result
        assert result["answer"] == "42"
        assert "reasoning" in result
        assert "Step by step" in result["reasoning"]
    
    def test_extract_missing_tags(self):
        """Test extract when tags are missing."""
        format_obj = StructuredFormat()
        response = "Just some text without tags."
        result = format_obj.extract(response)
        
        assert "answer" in result
        assert result["answer"] is None
        assert "reasoning" in result
        assert result["reasoning"] is None
    
    def test_extract_only_answer_tag(self):
        """Test extract when only answer tag is present."""
        format_obj = StructuredFormat()
        response = "<answer>42</answer>"
        result = format_obj.extract(response)
        
        assert "answer" in result
        assert result["answer"] == "42"
        assert "reasoning" in result
        assert result["reasoning"] is None
    
    def test_extract_only_reasoning_tag(self):
        """Test extract when only reasoning tag is present."""
        format_obj = StructuredFormat()
        response = "<think>Some reasoning</think>"
        result = format_obj.extract(response)
        
        assert "answer" in result
        assert result["answer"] is None
        assert "reasoning" in result
        assert result["reasoning"] == "Some reasoning"
    
    def test_extract_case_insensitive(self):
        """Test extract is case insensitive for tags."""
        format_obj = StructuredFormat()
        response = """<THINK>
Reasoning here.
</THINK>
<ANSWER>
42
</ANSWER>"""
        result = format_obj.extract(response)
        
        assert "answer" in result
        assert result["answer"] == "42"
        assert "reasoning" in result
    
    def test_extract_multiline_content(self):
        """Test extract with multiline content in tags."""
        format_obj = StructuredFormat()
        response = """<think>
Line 1
Line 2
Line 3
</think>
<answer>
The answer
is 42
</answer>"""
        result = format_obj.extract(response)
        
        assert "answer" in result
        assert "42" in result["answer"]
        assert "reasoning" in result
        assert "Line 1" in result["reasoning"]
        assert "Line 3" in result["reasoning"]

