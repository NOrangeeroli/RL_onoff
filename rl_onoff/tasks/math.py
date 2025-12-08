"""Math task with prompt template and math verification reward."""

from typing import Optional

from rl_onoff.tasks.base import BaseTask
from rl_onoff.tasks.rewards.builtin import MathVerifyReward
from rl_onoff.tasks.formats.base import BaseFormat
from rl_onoff.tasks.formats.boxed import BoxedFormat
from rl_onoff.tasks.formats.structured import StructuredFormat


class MathTask(BaseTask):
    """Math problem-solving task.
    
    Includes:
    - Prompt template for math problems
    - Response format (system prompt + extractor)
    - MathVerifyReward for evaluating solutions
    """

    def __init__(
        self, 
        prompt_template: Optional[str] = None, 
        name: Optional[str] = None,
        format: Optional[BaseFormat] = None
    ):
        """Initialize math task.
        
        Args:
            prompt_template: Custom prompt template (uses default if None)
            name: Task name (defaults to "math")
            format: Response format instance (defaults to BoxedFormat)
        """
        self._custom_template = prompt_template
        super().__init__(name=name or "math", format=format)

    def _create_reward(self) -> MathVerifyReward:
        """Create the math verification reward."""
        return MathVerifyReward()

    def _create_format(self) -> BaseFormat:
        """Create the default format for math task (boxed format)."""
        return BoxedFormat()

    def get_prompt_template(self) -> str:
        """Get the prompt template for math problems.
        
        Returns:
            Template string with $question placeholder
        """
        if self._custom_template is not None:
            return self._custom_template
        
        # Default math prompt template
        return """Solve the following math problem step by step. Show your work and provide the final answer.

Problem: $question

Solution:"""

