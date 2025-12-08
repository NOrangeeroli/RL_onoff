"""Tests for BaseChatTemplate."""

import pytest
from rl_onoff.tasks.chat_templates.base import BaseChatTemplate


class ConcreteChatTemplate(BaseChatTemplate):
    """Concrete implementation for testing."""
    
    def format(self, messages, add_generation_prompt=False, **kwargs):
        result = []
        for msg in messages:
            result.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
        if add_generation_prompt:
            result.append("assistant:")
        return "\n".join(result)


class TestBaseChatTemplate:
    """Tests for BaseChatTemplate abstract class."""
    
    def test_init_default_name(self):
        """Test initialization with default name."""
        template = ConcreteChatTemplate()
        assert template.name == "ConcreteChatTemplate"
    
    def test_init_custom_name(self):
        """Test initialization with custom name."""
        template = ConcreteChatTemplate(name="custom")
        assert template.name == "custom"
    
    def test_format_simple_with_system(self):
        """Test format_simple with system message."""
        template = ConcreteChatTemplate()
        result = template.format_simple(
            user_message="Hello",
            system_message="You are helpful"
        )
        result_lower = result.lower()
        assert "system:" in result_lower
        assert "you are helpful" in result_lower
        assert "user:" in result_lower
        assert "hello" in result_lower
    
    def test_format_simple_without_system(self):
        """Test format_simple without system message."""
        template = ConcreteChatTemplate()
        result = template.format_simple(user_message="Hello")
        result_lower = result.lower()
        assert "user:" in result_lower
        assert "hello" in result_lower
        assert "system:" not in result_lower
    
    def test_format_simple_with_assistant(self):
        """Test format_simple with assistant message."""
        template = ConcreteChatTemplate()
        result = template.format_simple(
            user_message="Hello",
            assistant_message="Hi there"
        )
        result_lower = result.lower()
        assert "user:" in result_lower
        assert "hello" in result_lower
        assert "assistant:" in result_lower
        assert "hi there" in result_lower
    
    def test_format_simple_with_generation_prompt(self):
        """Test format_simple with add_generation_prompt."""
        template = ConcreteChatTemplate()
        result = template.format_simple(
            user_message="Hello",
            add_generation_prompt=True
        )
        assert "assistant:" in result.lower()
    
    def test_format_simple_all_roles(self):
        """Test format_simple with all roles."""
        template = ConcreteChatTemplate()
        result = template.format_simple(
            system_message="You are helpful",
            user_message="What is 2+2?",
            assistant_message="4",
            add_generation_prompt=True
        )
        assert "system:" in result.lower()
        assert "user:" in result.lower()
        assert "assistant: 4" in result.lower()
        assert "assistant:" in result  # generation prompt
    
    def test_repr(self):
        """Test __repr__ method."""
        template = ConcreteChatTemplate()
        repr_str = repr(template)
        assert "ConcreteChatTemplate" in repr_str
        assert "name=" in repr_str

