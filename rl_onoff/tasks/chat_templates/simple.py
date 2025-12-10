"""Simple chat template with basic concatenation."""

from typing import List, Dict

from rl_onoff.tasks.chat_templates.base import BaseChatTemplate


class SimpleChatTemplate(BaseChatTemplate):
    """Simple chat template.
    
    Concatenates messages without special formatting tags or role labels.
    Just joins message contents with newlines.
    """

    def __init__(self):
        """Initialize simple chat template."""
        super().__init__("simple")

    def format(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages by simple concatenation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            add_generation_prompt: If True, adds a newline at the end for generation
            
        Returns:
            Formatted prompt string with messages concatenated
        """
        parts = []
        
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            
            # For simple template, just use the content
            # Skip empty messages
            if content:
                parts.append(content)
            
            # If we encounter an assistant message with content, don't add generation prompt
            if role == "assistant_generation" and content:
                add_generation_prompt = False
        
        result = "\n\n".join(parts)
        
        # Add newline at end for generation if requested
        if add_generation_prompt:
            result += "\n\n"
        
        return result


if __name__ == "__main__":
    # Simple usage example
    print("SimpleChatTemplate Usage Example")
    print("=" * 50)
    
    template = SimpleChatTemplate()
    
    # Example 1: System and user message
    print("\nExample 1: System + User message")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    prompt = template.format(messages, add_generation_prompt=True)
    print("Messages:", messages)
    print("Formatted prompt:")
    print(prompt)
    print()
    
    # Example 2: Only user message
    print("\nExample 2: User message only")
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    prompt = template.format(messages, add_generation_prompt=True)
    print("Messages:", messages)
    print("Formatted prompt:")
    print(repr(prompt))
    print()
    
    # Example 3: Multiple messages
    print("\nExample 3: Multiple messages")
    messages = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},
        {"role": "user", "content": "What is 4+4?"}
    ]
    prompt = template.format(messages, add_generation_prompt=True)
    print("Messages:", messages)
    print("Formatted prompt:")
    print(prompt)
    print()
    
    print("=" * 50)

