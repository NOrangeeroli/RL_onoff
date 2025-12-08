"""Tests for MathTask."""

import pytest
from rl_onoff.tasks.math import MathTask
from rl_onoff.tasks.formats.boxed import BoxedFormat
from rl_onoff.tasks.formats.structured import StructuredFormat
from rl_onoff.tasks.rewards.builtin import MathVerifyReward


class TestMathTask:
    """Tests for MathTask."""
    
    def test_init_default(self):
        """Test initialization with defaults."""
        task = MathTask()
        assert task.name == "math"
        assert isinstance(task.format, BoxedFormat)
        assert isinstance(task.reward, MathVerifyReward)
    
    def test_init_custom_name(self):
        """Test initialization with custom name."""
        task = MathTask(name="custom_math")
        assert task.name == "custom_math"
    
    def test_init_custom_template(self):
        """Test initialization with custom prompt template."""
        custom_template = "Solve: $question"
        task = MathTask(prompt_template=custom_template)
        assert task.get_prompt_template() == custom_template
    
    def test_init_custom_format(self):
        """Test initialization with custom format."""
        custom_format = StructuredFormat()
        task = MathTask(format=custom_format)
        assert task.format is custom_format
        assert isinstance(task.format, StructuredFormat)
    
    def test_get_prompt_template_default(self):
        """Test get_prompt_template returns default template."""
        task = MathTask()
        template = task.get_prompt_template()
        assert "Problem:" in template
        assert "$question" in template
        assert "Solution:" in template
    
    def test_get_prompt_template_custom(self):
        """Test get_prompt_template returns custom template."""
        custom_template = "Solve: $question"
        task = MathTask(prompt_template=custom_template)
        assert task.get_prompt_template() == custom_template
    
    def test_get_system_prompt_default(self):
        """Test get_system_prompt with default BoxedFormat."""
        task = MathTask()
        prompt = task.get_system_prompt()
        assert "boxed" in prompt.lower() or "\\boxed" in prompt
    
    def test_get_system_prompt_structured(self):
        """Test get_system_prompt with StructuredFormat."""
        task = MathTask(format=StructuredFormat())
        prompt = task.get_system_prompt()
        assert "<think>" in prompt or "redacted_reasoning" in prompt.lower()
        assert "<answer>" in prompt.lower()
    
    def test_answer_extractor_boxed(self):
        """Test answer_extractor with BoxedFormat."""
        task = MathTask(format=BoxedFormat())
        response = "Let me solve this. \\boxed{42}"
        result = task.answer_extractor(response)
        assert result["answer"] == "42"
        assert "reasoning" in result
    
    def test_answer_extractor_structured(self):
        """Test answer_extractor with StructuredFormat."""
        task = MathTask(format=StructuredFormat())
        response = """<think>
Step by step solution.
</think>
<answer>
42
</answer>"""
        result = task.answer_extractor(response)
        assert result["answer"] == "42"
        assert "reasoning" in result
    
    def test_format_prompt(self):
        """Test format_prompt formats question correctly."""
        task = MathTask()
        result = task.format_prompt("What is 2+2?")
        assert "What is 2+2?" in result
        assert "Problem:" in result or "Solve" in result
    
    def test_format_prompts_single(self):
        """Test format_prompts with single question."""
        task = MathTask()
        result = task.format_prompts("What is 2+2?")
        assert isinstance(result, str)
        assert "What is 2+2?" in result
    
    def test_format_prompts_multiple(self):
        """Test format_prompts with multiple questions."""
        task = MathTask()
        questions = ["What is 2+2?", "What is 3+3?"]
        result = task.format_prompts(questions)
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_evaluate(self):
        """Test evaluate uses MathVerifyReward."""
        task = MathTask()
        # MathVerifyReward may require math_verify library
        # So we just test that the method exists and can be called
        try:
            result = task.evaluate("2+2", "4")
            assert isinstance(result, (float, int, dict))
        except ImportError:
            pytest.skip("math_verify not available")
    
    def test_repr(self):
        """Test __repr__ method."""
        task = MathTask()
        repr_str = repr(task)
        assert "MathTask" in repr_str
        assert "name=" in repr_str

