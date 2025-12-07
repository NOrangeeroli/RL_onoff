"""Base metric interface for extensible metrics framework."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import numpy as np


class BaseMetric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, name: Optional[str] = None):
        """Initialize metric.
        
        Args:
            name: Metric name (defaults to class name)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]],
        **kwargs
    ) -> Union[float, Dict[str, float]]:
        """Compute metric value(s).
        
        Args:
            predictions: Predicted text(s)
            references: Reference text(s) or list of reference lists
            **kwargs: Additional arguments
            
        Returns:
            Metric value(s) as float or dict of metric values
        """
        pass

    def __call__(self, *args, **kwargs):
        """Allow metric to be called directly."""
        return self.compute(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class MetricRegistry:
    """Registry for managing metrics."""

    def __init__(self):
        """Initialize registry."""
        self._metrics: Dict[str, BaseMetric] = {}

    def register(self, metric: BaseMetric, name: Optional[str] = None):
        """Register a metric.
        
        Args:
            metric: Metric instance to register
            name: Optional name override
        """
        metric_name = name or metric.name
        self._metrics[metric_name] = metric

    def get(self, name: str) -> BaseMetric:
        """Get a metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Metric instance
            
        Raises:
            KeyError: If metric not found
        """
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found. Available: {list(self._metrics.keys())}")
        return self._metrics[name]

    def compute_all(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]],
        metric_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute all registered metrics or a subset.
        
        Args:
            predictions: Predicted text(s)
            references: Reference text(s)
            metric_names: Optional list of metric names to compute (all if None)
            **kwargs: Additional arguments passed to metrics
            
        Returns:
            Dictionary mapping metric names to their values
        """
        metrics_to_compute = metric_names or list(self._metrics.keys())
        results = {}
        
        for name in metrics_to_compute:
            metric = self.get(name)
            try:
                results[name] = metric.compute(predictions, references, **kwargs)
            except Exception as e:
                results[name] = {"error": str(e)}
        
        return results

    def list_metrics(self) -> List[str]:
        """List all registered metric names.
        
        Returns:
            List of metric names
        """
        return list(self._metrics.keys())

    def clear(self):
        """Clear all registered metrics."""
        self._metrics.clear()

