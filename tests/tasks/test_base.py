"""Tests for BaseTask."""

import pytest
from typing import Optional, Dict
from rl_onoff.tasks.base import BaseTask
from rl_onoff.tasks.rewards.base import BaseReward
from rl_onoff.tasks.formats.base import BaseFormat


class ConcreteFormat(BaseFormat):
    """Concrete format for testing."""
    
    def get_system_prompt(self) -> str:
        return "Test system prompt"
    
    def extract(self, response: str) -> Dict[str, Optional[str]]:
        return {"answer": "test_answer", "reasoning": "test_reasoning"}


class ConcreteReward(BaseReward):
    """Concrete reward for testing."""
    
    def compute(self, predictions, references, **kwargs):
        return 1.0


class ConcreteTask(BaseTask):
    """Concrete task for testing."""
    
    def _create_reward(self) -> BaseReward:
        return ConcreteReward()
    
    def _create_format(self) -> BaseFormat:
        return ConcreteFormat()
    
    def get_prompt_template(self) -> str:
        return "Question: $question\nAnswer:"


class TestBaseTask:
    """Tests for BaseTask abstract class."""
    
    def test_init_default_name(self):
        """Test initialization with default name."""
        task = ConcreteTask()
        assert task.name == "ConcreteTask"
    
    def test_init_custom_name(self):
        """Test initialization with custom name."""
        task = ConcreteTask(name="custom")
        assert task.name == "custom"
    
    def test_init_with_format(self):
        """Test initialization with custom format."""
        custom_format = ConcreteFormat(name="custom_format")
        task = ConcreteTask(format=custom_format)
        assert task.format is custom_format
        assert task.format.name == "custom_format"
    
    def test_get_system_prompt(self):
        """Test get_system_prompt delegates to format."""
        task = ConcreteTask()
        prompt = task.get_system_prompt()
        assert prompt == "Test system prompt"
    
    def test_answer_extractor(self):
        """Test answer_extractor delegates to format."""
        task = ConcreteTask()
        result = task.answer_extractor("test response")
        assert isinstance(result, dict)
        assert result["answer"] == "test_answer"
        assert result["reasoning"] == "test_reasoning"
    
    def test_format_prompt_single(self):
        """Test format_prompt with single question."""
        task = ConcreteTask()
        result = task.format_prompt("What is 2+2?")
        assert "Question: What is 2+2?" in result
        assert "Answer:" in result
    
    def test_format_prompt_with_kwargs(self):
        """Test format_prompt with additional kwargs."""
        task = ConcreteTask()
        result = task.format_prompt("What is 2+2?", extra="test")
        assert "Question: What is 2+2?" in result
    
    def test_format_prompts_single(self):
        """Test format_prompts with single question."""
        task = ConcreteTask()
        result = task.format_prompts("What is 2+2?")
        assert isinstance(result, str)
        assert "Question: What is 2+2?" in result
    
    def test_format_prompts_multiple(self):
        """Test format_prompts with multiple questions."""
        task = ConcreteTask()
        questions = ["What is 2+2?", "What is 3+3?"]
        result = task.format_prompts(questions)
        assert isinstance(result, list)
        assert len(result) == 2
        assert "Question: What is 2+2?" in result[0]
        assert "Question: What is 3+3?" in result[1]
    
    def test_format_query(self):
        """Test format_query alias for format_prompt."""
        task = ConcreteTask()
        result1 = task.format_query("What is 2+2?")
        result2 = task.format_prompt("What is 2+2?")
        assert result1 == result2
    
    def test_evaluate(self):
        """Test evaluate delegates to reward."""
        task = ConcreteTask()
        result = task.evaluate("prediction", "reference")
        assert result == 1.0
    
    def test_evaluate_list(self):
        """Test evaluate with list inputs."""
        task = ConcreteTask()
        result = task.evaluate(["pred1", "pred2"], ["ref1", "ref2"])
        assert result == 1.0  # ConcreteReward returns 1.0 for any input
    
    def test_repr(self):
        """Test __repr__ method."""
        task = ConcreteTask()
        repr_str = repr(task)
        assert "ConcreteTask" in repr_str
        assert "name=" in repr_str

