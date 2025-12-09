"""LLM Evaluation and Model Switching Framework."""

__version__ = "0.1.0"

from rl_onoff.backends import create_backend
from rl_onoff.sampling import Sampler
from rl_onoff.tasks.rewards import RewardRegistry
from rl_onoff.distributions import DistributionExtractor
from rl_onoff.divergence import DivergenceCalculator
from rl_onoff.switching import ModelSwitcher
from rl_onoff.tasks import BaseTask, MathTask

__all__ = [
    "create_backend",
    "Sampler",
    "RewardRegistry",
    "DistributionExtractor",
    "DivergenceCalculator",
    "ModelSwitcher",
    "BaseTask",
    "MathTask",
]

