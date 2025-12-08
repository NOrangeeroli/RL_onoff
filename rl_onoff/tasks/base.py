"""Base task interface for task-specific prompt templates and metrics."""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
from string import Template

from rl_onoff.tasks.rewards.base import BaseMetric


class BaseTask(ABC):
    """Abstract base class for all tasks.
    
    Each task defines:
    - A prompt template for formatting questions
    - A metric for evaluating responses
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize task.
        
        Args:
            name: Task name (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.metric = self._create_metric()

    @abstractmethod
    def _create_metric(self) -> BaseMetric:
        """Create the metric instance for this task.
        
        Returns:
            Metric instance
        """
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get the prompt template string for this task.
        
        Returns:
            Template string (can use $variable syntax for string.Template)
        """
        pass

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
        """Evaluate predictions using the task's metric.
        
        Args:
            predictions: Predicted response(s)
            references: Reference answer(s) or list of reference lists
            **kwargs: Additional arguments passed to metric
            
        Returns:
            Metric score(s)
        """
        return self.metric.compute(predictions, references, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

