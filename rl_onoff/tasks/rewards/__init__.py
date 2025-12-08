"""Rewards framework for evaluating model outputs."""

from rl_onoff.tasks.rewards.base import BaseMetric, MetricRegistry
from rl_onoff.tasks.rewards.builtin import (
    PerplexityMetric,
    BLEUMetric,
    ROUGEMetric,
    ExactMatchMetric,
    MathVerifyMetric,
)

__all__ = [
    "BaseMetric",
    "MetricRegistry",
    "PerplexityMetric",
    "BLEUMetric",
    "ROUGEMetric",
    "ExactMatchMetric",
    "MathVerifyMetric",
]

