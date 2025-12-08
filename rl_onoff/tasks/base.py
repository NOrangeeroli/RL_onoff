"""Base task interface for task-specific prompt templates and rewards."""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional, Tuple
from string import Template

from rl_onoff.tasks.rewards.base import BaseReward
from rl_onoff.tasks.formats.base import BaseFormat


class BaseTask(ABC):
    """Abstract base class for all tasks.
    
    Each task defines:
    - A prompt template for formatting questions
    - A format for response structure (system prompt + extractor)
    - A reward for evaluating responses
    """

    def __init__(self, name: Optional[str] = None, format: Optional[BaseFormat] = None):
        """Initialize task.
        
        Args:
            name: Task name (defaults to class name)
            format: Response format instance (uses default if None)
        """
        self.name = name or self.__class__.__name__
        self.format = format or self._create_format()
        self.reward = self._create_reward()

    @abstractmethod
    def _create_reward(self) -> BaseReward:
        """Create the reward instance for this task.
        
        Returns:
            Reward instance
        """
        pass

    @abstractmethod
    def _create_format(self) -> BaseFormat:
        """Create the format instance for this task.
        
        Returns:
            Format instance
        """
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get the prompt template string for this task.
        
        Returns:
            Template string (can use $variable syntax for string.Template)
        """
        pass

    def get_system_prompt(self) -> str:
        """Get the system prompt for this task from its format.
        
        Returns:
            System prompt string
        """
        return self.format.get_system_prompt()

    def answer_extractor(self, response: str) -> Dict[str, Optional[str]]:
        """Extract information from a model response using the task's format.
        
        Args:
            response: Model response text
            
        Returns:
            Dictionary with extracted information. Must include:
            - "answer": The final answer part of the response (required)
            Optional fields:
            - "reasoning": The reasoning/process part of the response (optional)
            Additional fields may be present depending on the format.
            Values can be None if not found or not applicable.
        """
        return self.format.extract(response)

    def format_query(self, question: str, **kwargs) -> str:
        """Format a query from a question using the task's template.
        
        This is an alias for format_prompt for consistency.
        
        Args:
            question: The question/problem to format
            **kwargs: Additional variables for template substitution
            
        Returns:
            Formatted query string
        """
        return self.format_prompt(question, **kwargs)

    def format_prompt(self, question: str, **kwargs) -> str:
        """Format a prompt from a question using the task's template.
        
        Args:
            question: The question/problem to format
            **kwargs: Additional variables for template substitution
            
        Returns:
            Formatted prompt string
        """
        template_str = self.get_prompt_template()
        template = Template(template_str)
        
        # Default variables
        template_vars = {
            'question': question,
            **kwargs
        }
        
        return template.safe_substitute(template_vars)

    def format_prompts(self, questions: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """Format multiple prompts from questions.
        
        Args:
            questions: Single question or list of questions
            **kwargs: Additional variables for template substitution
            
        Returns:
            Formatted prompt(s)
        """
        is_single = isinstance(questions, str)
        if is_single:
            questions = [questions]
        
        prompts = [self.format_prompt(q, **kwargs) for q in questions]
        
        return prompts[0] if is_single else prompts

    def evaluate(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]],
        **kwargs
    ) -> Union[float, List[float], Dict[str, Any]]:
        """Evaluate predictions using the task's reward.
        
        Args:
            predictions: Predicted response(s)
            references: Reference answer(s) or list of reference lists
            **kwargs: Additional arguments passed to reward
            
        Returns:
            Reward score(s)
        """
        return self.reward.compute(predictions, references, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

