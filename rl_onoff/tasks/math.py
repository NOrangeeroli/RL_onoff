"""Math task with prompt template and math verification metric."""

from typing import Optional

from rl_onoff.tasks.base import BaseTask
from rl_onoff.tasks.rewards.builtin import MathVerifyMetric


class MathTask(BaseTask):
    """Math problem-solving task.
    
    Includes:
    - Prompt template for math problems
    - MathVerifyMetric for evaluating solutions
    """

    def __init__(self, prompt_template: Optional[str] = None, name: Optional[str] = None):
        """Initialize math task.
        
        Args:
            prompt_template: Custom prompt template (uses default if None)
            name: Task name (defaults to "math")
        """
        self._custom_template = prompt_template
        super().__init__(name=name or "math")

    def _create_metric(self) -> MathVerifyMetric:
        """Create the math verification metric."""
        return MathVerifyMetric()

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

