"""Base task interface for task-specific prompt templates and rewards."""

from typing import Union, List, Dict, Any, Optional
from string import Template
from pathlib import Path

from rl_onoff.tasks.rewards.base import BaseReward
from rl_onoff.tasks.formats.base import BaseFormat
from rl_onoff.tasks.config import TaskConfig
from rl_onoff.tasks.chat_templates import create_chat_template
from rl_onoff.tasks.formats import create_format
from rl_onoff.tasks.rewards import create_reward


class BaseTask:
    """Base class for all tasks.
    
    Each task is configured via a config file that specifies:
    - Template type for formatting questions
    - Reward type for evaluating responses
    - Format type for response structure (system prompt + extractor)
    """

    def __init__(self, config: Union[str, Path, TaskConfig, Dict[str, Any]]):
        """Initialize task from config.
        
        Args:
            config: Config file path (str/Path), TaskConfig instance, or dict
        """
        # Load config
        if isinstance(config, (str, Path)):
            self.config = TaskConfig.from_json(config)
        elif isinstance(config, TaskConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = TaskConfig.from_dict(config)
        else:
            raise TypeError(f"config must be str, Path, TaskConfig, or dict, got {type(config)}")
        
        self.name = self.__class__.__name__
        
        # Create template, format and reward from config
        self.template = create_chat_template(
            self.config.template_type,
            **(self.config.template_kwargs or {})
        )
        self.format = create_format(
            self.config.format_type,
            **(self.config.format_kwargs or {})
        )
        self.reward = create_reward(
            self.config.reward_type,
            **(self.config.reward_kwargs or {})
        )
    

    def extract_answer(self, response: str) -> Dict[str, Optional[str]]:
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
        """Format a query from a question using the task's chat template.
        
        Combines the question with the system prompt from the format using
        the configured chat template. This is the preferred method for
        formatting queries for model generation.
        
        Args:
            question: The question/problem to format
            **kwargs: Additional arguments passed to template.format
                      (e.g., add_generation_prompt=True)
            
        Returns:
            Formatted query string ready for model input
        """
        messages = self.format.format_message(question)
        add_generation_prompt = kwargs.pop('add_generation_prompt', True)
        return self.template.format(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs
        )

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
        # Check if single prediction
        is_single = isinstance(predictions, str)
        if is_single:
            predictions = [predictions]
        
        # Extract answers from responses
        extracted = [self.extract_answer(prediction) for prediction in predictions]
        # Extract the "answer" field from each extracted dict (handle None values)
        answers = [ext.get("answer") or "" for ext in extracted]
        
        # Compute scores
        scores = self.reward.compute(answers, references, **kwargs)
        
        # Return single value for single prediction, list for multiple
        return scores[0] if is_single and isinstance(scores, list) else scores

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

