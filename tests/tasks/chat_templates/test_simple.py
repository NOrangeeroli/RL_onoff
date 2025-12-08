"""Tests for SimpleChatTemplate."""

import pytest
from rl_onoff.tasks.chat_templates.simple import SimpleChatTemplate


class TestSimpleChatTemplate:
    """Tests for SimpleChatTemplate."""
    
    def test_init_default_separator(self):
        """Test initialization with default separator."""
        template = SimpleChatTemplate()
        assert template.name == "simple"
        assert template.separator == "\n\n"
    
    def test_init_custom_separator(self):
        """Test initialization with custom separator."""
        template = SimpleChatTemplate(separator="---")
        assert template.separator == "---"
    
    def test_format_single_user_message(self):
        """Test formatting a single user message."""
        template = SimpleChatTemplate()
        messages = [{"role": "user", "content": "Hello"}]
        result = template.format(messages)
        assert result == "Hello"
    
    def test_format_system_and_user(self):
        """Test formatting system and user messages."""
        template = SimpleChatTemplate()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        result = template.format(messages)
        assert "You are helpful" in result
        assert "Hello" in result
        assert result == "You are helpful\n\nHello"
    
    def test_format_with_assistant(self):
        """Test formatting with assistant message."""
        template = SimpleChatTemplate()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
        result = template.format(messages)
        assert "What is 2+2?" in result
        assert "4" in result
        assert result == "What is 2+2?\n\n4"
    
    def test_format_multiple_turns(self):
        """Test formatting multiple conversation turns."""
        template = SimpleChatTemplate()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
        result = template.format(messages)
        assert "You are helpful" in result
        assert "Hello" in result
        assert "Hi there" in result
        assert "What is 2+2?" in result
        assert "4" in result
        # Check separator appears between messages
        assert result.count("\n\n") == 4
    
    def test_format_with_generation_prompt(self):
        """Test formatting with add_generation_prompt."""
        template = SimpleChatTemplate()
        messages = [{"role": "user", "content": "Hello"}]
        result = template.format(messages, add_generation_prompt=True)
        assert result == "Hello\n\n"
        assert result.endswith("\n\n")
    
    def test_format_empty_messages(self):
        """Test formatting empty messages list."""
        template = SimpleChatTemplate()
        result = template.format([])
        assert result == ""
    
    def test_format_empty_messages_with_generation_prompt(self):
        """Test formatting empty messages with generation prompt."""
        template = SimpleChatTemplate()
        result = template.format([], add_generation_prompt=True)
        assert result == "\n\n"
    
    def test_format_empty_content(self):
        """Test formatting message with empty content (should be skipped)."""
        template = SimpleChatTemplate()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "World"}
        ]
        result = template.format(messages)
        assert "Hello" in result
        assert "World" in result
        # Empty content should not appear
        assert result.count("\n\n") == 1  # Only one separator between Hello and World
    
    def test_format_missing_content(self):
        """Test formatting message with missing content (should be skipped)."""
        template = SimpleChatTemplate()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant"},  # Missing content
            {"role": "user", "content": "World"}
        ]
        result = template.format(messages)
        assert "Hello" in result
        assert "World" in result
        # Missing content should not appear
        assert result.count("\n\n") == 1
    
    def test_format_custom_separator(self):
        """Test formatting with custom separator."""
        template = SimpleChatTemplate(separator="---")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        result = template.format(messages)
        assert result == "Hello---Hi"
    
    def test_format_custom_separator_with_generation_prompt(self):
        """Test formatting with custom separator and generation prompt."""
        template = SimpleChatTemplate(separator="---")
        messages = [{"role": "user", "content": "Hello"}]
        result = template.format(messages, add_generation_prompt=True)
        assert result == "Hello---"
    
    def test_format_simple(self):
        """Test format_simple convenience method."""
        template = SimpleChatTemplate()
        result = template.format_simple(
            user_message="Hello",
            system_message="You are helpful"
        )
        assert "You are helpful" in result
        assert "Hello" in result
        assert result == "You are helpful\n\nHello"
    
    def test_format_simple_with_assistant(self):
        """Test format_simple with assistant message."""
        template = SimpleChatTemplate()
        result = template.format_simple(
            user_message="Hello",
            assistant_message="Hi there"
        )
        assert "Hello" in result
        assert "Hi there" in result
        assert result == "Hello\n\nHi there"
    
    def test_format_simple_with_generation_prompt(self):
        """Test format_simple with generation prompt."""
        template = SimpleChatTemplate()
        result = template.format_simple(
            user_message="Hello",
            add_generation_prompt=True
        )
        assert result == "Hello\n\n"

