"""Tests for ChatMLTemplate."""

import pytest
from rl_onoff.tasks.chat_templates.chatml import ChatMLTemplate


class TestChatMLTemplate:
    """Tests for ChatMLTemplate."""
    
    def test_init(self):
        """Test initialization."""
        template = ChatMLTemplate()
        assert template.name == "chatml"
    
    def test_format_single_user_message(self):
        """Test formatting a single user message."""
        template = ChatMLTemplate()
        messages = [{"role": "user", "content": "Hello"}]
        result = template.format(messages)
        assert "<|im_start|>user" in result
        assert "Hello" in result
        assert "<|im_end|>" in result
    
    def test_format_system_and_user(self):
        """Test formatting system and user messages."""
        template = ChatMLTemplate()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        result = template.format(messages)
        assert "<|im_start|>system" in result
        assert "You are helpful" in result
        assert "<|im_end|>" in result
        assert "<|im_start|>user" in result
        assert "Hello" in result
    
    def test_format_with_assistant(self):
        """Test formatting with assistant message."""
        template = ChatMLTemplate()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
        result = template.format(messages)
        assert "<|im_start|>user" in result
        assert "What is 2+2?" in result
        assert "<|im_start|>assistant" in result
        assert "4" in result
        assert "<|im_end|>" in result
    
    def test_format_multiple_turns(self):
        """Test formatting multiple conversation turns."""
        template = ChatMLTemplate()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
        result = template.format(messages)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "Hello" in result
        assert "<|im_start|>assistant" in result
        assert "Hi there" in result
        assert "What is 2+2?" in result
        assert "4" in result
        # Check that each message has im_end
        assert result.count("<|im_end|>") == 5
    
    def test_format_with_generation_prompt(self):
        """Test formatting with add_generation_prompt."""
        template = ChatMLTemplate()
        messages = [{"role": "user", "content": "Hello"}]
        result = template.format(messages, add_generation_prompt=True)
        assert "<|im_start|>user" in result
        assert "Hello" in result
        assert "<|im_start|>assistant" in result
        assert result.endswith("<|im_start|>assistant\n")
    
    def test_format_empty_messages(self):
        """Test formatting empty messages list."""
        template = ChatMLTemplate()
        result = template.format([])
        assert result == ""
    
    def test_format_empty_messages_with_generation_prompt(self):
        """Test formatting empty messages with generation prompt."""
        template = ChatMLTemplate()
        result = template.format([], add_generation_prompt=True)
        assert result == "<|im_start|>assistant\n"
    
    def test_format_missing_role(self):
        """Test formatting message with missing role (defaults to user)."""
        template = ChatMLTemplate()
        messages = [{"content": "Hello"}]
        result = template.format(messages)
        assert "<|im_start|>user" in result
        assert "Hello" in result
    
    def test_format_missing_content(self):
        """Test formatting message with missing content."""
        template = ChatMLTemplate()
        messages = [{"role": "user"}]
        result = template.format(messages)
        assert "<|im_start|>user" in result
        assert "<|im_end|>" in result
    
    def test_format_newline_separation(self):
        """Test that messages are separated by newlines."""
        template = ChatMLTemplate()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        result = template.format(messages)
        # Should have newline between messages
        assert "\n" in result
        parts = result.split("\n")
        assert len(parts) >= 2
    
    def test_format_simple(self):
        """Test format_simple convenience method."""
        template = ChatMLTemplate()
        result = template.format_simple(
            user_message="Hello",
            system_message="You are helpful"
        )
        assert "<|im_start|>system" in result
        assert "You are helpful" in result
        assert "<|im_start|>user" in result
        assert "Hello" in result

