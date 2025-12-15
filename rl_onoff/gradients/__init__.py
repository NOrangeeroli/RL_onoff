"""Gradient utilities for RL-ON/OFF.

This package provides:
- Random projection utilities for high-dimensional gradients
- Helpers to extract and project LoRA-specific gradients
"""

from .projectors import (
    AbstractProjector,
    NoOpProjector,
    BasicSingleBlockProjector,
    BasicProjector,
    CudaProjector,
    ChunkedCudaProjector,
    ProjectionType,
)
from .lora_extractor import LoraBGradientProjector

__all__ = [
    "AbstractProjector",
    "NoOpProjector",
    "BasicSingleBlockProjector",
    "BasicProjector",
    "CudaProjector",
    "ChunkedCudaProjector",
    "ProjectionType",
    "LoraBGradientProjector",
]


