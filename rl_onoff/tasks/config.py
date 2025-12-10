"""Task configuration management."""

from typing import Optional, Dict, Any, Union, Literal
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from rl_onoff.utils.config import Config

# Import registries to get valid values
from rl_onoff.tasks.chat_templates import CHAT_TEMPLATE_REGISTRY
from rl_onoff.tasks.formats import FORMAT_REGISTRY
from rl_onoff.tasks.rewards import REWARD_REGISTRY

# Type aliases for valid values
TemplateType = Literal["openai", "llama", "chatml", "simple"]
RewardType = Literal["math_verify"]
FormatType = Literal["boxed", "structured"]


@dataclass
class TaskConfig(Config):
    """Configuration for a task.
    
    Specifies:
    - template_type: Type of chat template to use (must be from CHAT_TEMPLATE_REGISTRY)
    - reward_type: Type of reward to use (must be from REWARD_REGISTRY)
    - format_type: Type of response format to use (must be from FORMAT_REGISTRY)
    - Additional parameters for each type
    
    Raises:
        ValueError: If template_type, reward_type, or format_type is not in the respective registry
    """
    
    template_type: TemplateType = "simple"
    reward_type: RewardType = "math_verify"
    format_type: FormatType = "boxed"
    
    def __post_init__(self):
        """Initialize default values for optional fields and validate types."""
        
        # Validate template_type
        if self.template_type not in CHAT_TEMPLATE_REGISTRY:
            raise ValueError(
                f"Invalid template_type: '{self.template_type}'. "
                f"Must be one of: {list(CHAT_TEMPLATE_REGISTRY.keys())}"
            )
        
        # Validate reward_type
        if self.reward_type not in REWARD_REGISTRY:
            raise ValueError(
                f"Invalid reward_type: '{self.reward_type}'. "
                f"Must be one of: {list(REWARD_REGISTRY.keys())}"
            )
        
        # Validate format_type
        if self.format_type not in FORMAT_REGISTRY:
            raise ValueError(
                f"Invalid format_type: '{self.format_type}'. "
                f"Must be one of: {list(FORMAT_REGISTRY.keys())}"
            )

