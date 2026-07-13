"""Core module for AutoMLChain.

This module contains the main pipeline orchestration and configuration.
"""

from .config import (
    ProviderType,
    FineTuningMethod,
    TaskType,
    HyperParams,
    Budget,
    ProviderConfig,
    PipelineConfig,
    DatasetConfig,
    EvaluationConfig,
    ConfigManager,
)
from .exceptions import (
    AutoMLChainError,
    ValidationError,
    DatasetError,
    TrainingError,
    ProviderError,
    ConfigurationError,
    APIKeyError,
    AutoMLChainTimeoutError,
    TimeoutError,  # Deprecated alias for AutoMLChainTimeoutError
    JobCancelledError,
)
from .pipeline import AutoMLPipeline

__all__ = [
    # Config
    "ProviderType",
    "FineTuningMethod",
    "TaskType",
    "HyperParams",
    "Budget",
    "ProviderConfig",
    "PipelineConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ConfigManager",
    # Exceptions
    "AutoMLChainError",
    "ValidationError",
    "DatasetError",
    "TrainingError",
    "ProviderError",
    "ConfigurationError",
    "APIKeyError",
    "AutoMLChainTimeoutError",
    "TimeoutError",  # Deprecated alias
    "JobCancelledError",
    # Pipeline
    "AutoMLPipeline",
]
