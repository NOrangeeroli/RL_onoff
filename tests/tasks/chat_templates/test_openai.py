"""Tests for OpenAIChatTemplate."""

import pytest
from rl_onoff.tasks.chat_templates.openai import OpenAIChatTemplate


class TestOpenAIChatTemplate:
    """Tests for OpenAIChatTemplate."""
    
    def test_init(self):
        """Test initialization."""
        template = OpenAIChatTemplate()
        assert template.name == "openai"
    
    def test_format_single_user_message(self):
        """Test formatting a single user message."""
        template = OpenAIChatTemplate()
        messages = [{"role": "user", "content": "Hello"}]
        result = template.format(messages)
        assert "User: Hello" in result
        assert "System:" not in result
        assert "Assistant:" not in result
    
    def test_format_system_and_user(self):
        """Test formatting system and user messages."""
        template = OpenAIChatTemplate()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        result = template.format(messages)
        assert "System: You are helpful" in result
        assert "User: Hello" in result
    
    def test_format_with_assistant(self):
        """Test formatting with assistant message."""
        template = OpenAIChatTemplate()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
        result = template.format(messages)
        assert "User: What is 2+2?" in result
        assert "Assistant: 4" in result
    
    def test_format_multiple_turns(self):
        """Test formatting multiple conversation turns."""
        template = OpenAIChatTemplate()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
        result = template.format(messages)
        assert "System: You are helpful" in result
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result
        assert "User: What is 2+2?" in result
        assert "Assistant: 4" in result
    
    def test_format_with_generation_prompt(self):
        """Test formatting with add_generation_prompt."""
        template = OpenAIChatTemplate()
        messages = [{"role": "user", "content": "Hello"}]
        result = template.format(messages, add_generation_prompt=True)
        assert "User: Hello" in result
        assert "Assistant:" in result
        assert result.endswith("Assistant:")
    
    def test_format_empty_messages(self):
        """Test formatting empty messages list."""
        template = OpenAIChatTemplate()
        result = template.format([])
        assert result == ""
    
    def test_format_empty_messages_with_generation_prompt(self):
        """Test formatting empty messages with generation prompt."""
        template = OpenAIChatTemplate()
        result = template.format([], add_generation_prompt=True)
        assert result == "Assistant:"
    
    def test_format_missing_role(self):
        """Test formatting message with missing role (defaults to user)."""
        template = OpenAIChatTemplate()
        messages = [{"content": "Hello"}]
        result = template.format(messages)
        assert "User: Hello" in result
    
    def test_format_missing_content(self):
        """Test formatting message with missing content."""
        template = OpenAIChatTemplate()
        messages = [{"role": "user"}]
        result = template.format(messages)
        assert "User: " in result
    
    def test_format_simple(self):
        """Test format_simple convenience method."""
        template = OpenAIChatTemplate()
        result = template.format_simple(
            user_message="Hello",
            system_message="You are helpful"
        )
        assert "System: You are helpful" in result
        assert "User: Hello" in result
    
    def test_format_simple_with_generation_prompt(self):
        """Test format_simple with generation prompt."""
        template = OpenAIChatTemplate()
        result = template.format_simple(
            user_message="Hello",
            add_generation_prompt=True
        )
        assert "User: Hello" in result
        assert "Assistant:" in result

