"""Tests for LlamaChatTemplate."""

import pytest
from rl_onoff.tasks.chat_templates.llama import LlamaChatTemplate


class TestLlamaChatTemplate:
    """Tests for LlamaChatTemplate."""
    
    def test_init(self):
        """Test initialization."""
        template = LlamaChatTemplate()
        assert template.name == "llama"
    
    def test_format_single_user_message(self):
        """Test formatting a single user message."""
        template = LlamaChatTemplate()
        messages = [{"role": "user", "content": "Hello"}]
        result = template.format(messages)
        assert result.startswith("<s>")
        assert "[INST] Hello [/INST]" in result
    
    def test_format_system_and_user(self):
        """Test formatting system and user messages."""
        template = LlamaChatTemplate()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        result = template.format(messages)
        assert result.startswith("<s>")
        assert "<<SYS>>" in result
        assert "You are helpful" in result
        assert "<</SYS>>" in result
        assert "[INST] Hello [/INST]" in result
    
    def test_format_with_assistant(self):
        """Test formatting with assistant message."""
        template = LlamaChatTemplate()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
        result = template.format(messages)
        assert "[INST] What is 2+2? [/INST]" in result
        assert "4 </s>" in result
    
    def test_format_multiple_turns(self):
        """Test formatting multiple conversation turns."""
        template = LlamaChatTemplate()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
        result = template.format(messages)
        assert result.startswith("<s>")
        assert "<<SYS>>" in result
        assert "[INST] Hello [/INST]" in result
        assert "Hi there </s>" in result
        assert "[INST] What is 2+2? [/INST]" in result
        assert "4 </s>" in result
    
    def test_format_with_generation_prompt(self):
        """Test formatting with add_generation_prompt."""
        template = LlamaChatTemplate()
        messages = [{"role": "user", "content": "Hello"}]
        result = template.format(messages, add_generation_prompt=True)
        assert "[INST] Hello [/INST]" in result
        assert "<s> [INST]" in result
    
    def test_format_with_generation_prompt_after_assistant(self):
        """Test generation prompt not added after assistant message."""
        template = LlamaChatTemplate()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        result = template.format(messages, add_generation_prompt=True)
        assert "Hi </s>" in result
        # Should not add generation prompt after assistant
        assert result.count("<s> [INST]") == 0
    
    def test_format_system_not_first(self):
        """Test formatting system message not at the beginning."""
        template = LlamaChatTemplate()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "You are helpful"}
        ]
        result = template.format(messages)
        assert "[INST] Hello [/INST]" in result
        assert "[INST] You are helpful [/INST]" in result
        assert "<<SYS>>" not in result  # Only first system message uses SYS tags
    
    def test_format_empty_messages(self):
        """Test formatting empty messages list."""
        template = LlamaChatTemplate()
        result = template.format([])
        assert result == "<s>"
    
    def test_format_empty_messages_with_generation_prompt(self):
        """Test formatting empty messages with generation prompt."""
        template = LlamaChatTemplate()
        result = template.format([], add_generation_prompt=True)
        assert "<s>" in result
        assert "<s> [INST]" in result
    
    def test_format_missing_role(self):
        """Test formatting message with missing role (defaults to user)."""
        template = LlamaChatTemplate()
        messages = [{"content": "Hello"}]
        result = template.format(messages)
        assert "[INST] Hello [/INST]" in result
    
    def test_format_missing_content(self):
        """Test formatting message with missing content."""
        template = LlamaChatTemplate()
        messages = [{"role": "user"}]
        result = template.format(messages)
        assert "[INST]  [/INST]" in result
    
    def test_format_simple(self):
        """Test format_simple convenience method."""
        template = LlamaChatTemplate()
        result = template.format_simple(
            user_message="Hello",
            system_message="You are helpful"
        )
        assert result.startswith("<s>")
        assert "<<SYS>>" in result
        assert "You are helpful" in result
        assert "[INST] Hello [/INST]" in result

