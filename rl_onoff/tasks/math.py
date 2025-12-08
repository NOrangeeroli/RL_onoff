"""Math task with prompt template and math verification reward."""

from typing import Union
from pathlib import Path

from rl_onoff.tasks.base import BaseTask
from rl_onoff.tasks.config import TaskConfig


class MathTask(BaseTask):
    """Math problem-solving task.
    
    Configured via a config file that specifies:
    - Template type for formatting questions
    - Reward type (typically "math_verify")
    - Format type (typically "boxed" or "structured")
    """

    def __init__(self, config: Union[str, Path, TaskConfig, dict] = None):
        """Initialize math task from config.
        
        Args:
            config: Config file path, TaskConfig instance, dict, or None for default config
        """
        if config is None:
            # Default math task config
            config = TaskConfig(
                template_type="simple",
                reward_type="math_verify",
                format_type="boxed"
            )
        
        super().__init__(config)
    
    def get_prompt_template(self) -> str:
        """Get the prompt template for math problems.
        
        Returns:
            Template string with $question placeholder
        """
        return """Solve the following math problem step by step. Show your work and provide the final answer.

Problem: $question

Solution:"""

