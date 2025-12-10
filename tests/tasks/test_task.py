"""Tests for Task."""

import pytest
from typing import Optional, Dict
from rl_onoff.tasks.task import Task
from rl_onoff.tasks.config import TaskConfig


class TestTask:
    """Tests for Task class."""
    
    def test_init_from_dict(self):
        """Test initialization from dict config."""
        config = {
            "template_type": "simple",
            "reward_type": "math_verify",
            "format_type": "boxed"
        }
        task = Task(config)
        assert task.name == "Task"
        assert task.config.template_type == "simple"
        assert task.config.reward_type == "math_verify"
        assert task.config.format_type == "boxed"
    
    def test_init_from_taskconfig(self):
        """Test initialization from TaskConfig."""
        config = TaskConfig(
            template_type="simple",
            reward_type="math_verify",
            format_type="boxed"
        )
        task = Task(config)
        assert task.config.template_type == "simple"
    
    def test_init_from_config_file(self):
        """Test initialization from config file."""
        config_path = "rl_onoff/tasks/configs/math_default.yaml"
        task = Task(config_path)
        assert task.config.template_type == "simple"
        assert task.config.reward_type == "math_verify"
    
    def test_extract_answer(self):
        """Test extract_answer delegates to format."""
        config = {
            "template_type": "simple",
            "reward_type": "math_verify",
            "format_type": "boxed"
        }
        task = Task(config)
        response = "The answer is \\boxed{42}"
        result = task.extract_answer(response)
        assert isinstance(result, dict)
        assert "answer" in result
        assert result["answer"] == "42"
    
    def test_format_query(self):
        """Test format_query formats question correctly."""
        config = {
            "template_type": "simple",
            "reward_type": "math_verify",
            "format_type": "boxed"
        }
        task = Task(config)
        question = "What is 2+2?"
        result = task.format_query(question)
        assert isinstance(result, str)
        assert question in result or "2+2" in result
    
    def test_evaluate_single(self):
        """Test evaluate with single prediction."""
        config = {
            "template_type": "simple",
            "reward_type": "math_verify",
            "format_type": "boxed"
        }
        task = Task(config)
        # Use a response that extracts to "4"
        response = "The answer is \\boxed{4}"
        reference = "4"
        try:
            score = task.evaluate(response, reference)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
        except ImportError:
            pytest.skip("math_verify not available")
    
    def test_evaluate_list(self):
        """Test evaluate with list inputs."""
        config = {
            "template_type": "simple",
            "reward_type": "math_verify",
            "format_type": "boxed"
        }
        task = Task(config)
        responses = [
            "The answer is \\boxed{4}",
            "The answer is \\boxed{6}"
        ]
        references = ["4", "6"]
        try:
            scores = task.evaluate(responses, references)
            assert isinstance(scores, list)
            assert len(scores) == 2
            assert all(0.0 <= s <= 1.0 for s in scores)
        except ImportError:
            pytest.skip("math_verify not available")
    
    def test_repr(self):
        """Test __repr__ method."""
        config = {
            "template_type": "simple",
            "reward_type": "math_verify",
            "format_type": "boxed"
        }
        task = Task(config)
        repr_str = repr(task)
        assert "Task" in repr_str
        assert "name=" in repr_str
