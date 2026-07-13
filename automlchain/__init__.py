"""AutoMLChain - Fine-tuning made accessible.

A Python library for automated fine-tuning of LLMs.
Orchestrates the complete fine-tuning workflow with auto-optimization.

Example:
    >>> from automlchain import AutoMLPipeline
    >>> pipeline = AutoMLPipeline(provider="replicate")
    >>> pipeline.upload_dataset("data.jsonl")
    >>> result = pipeline.train()
"""

__version__ = "0.1.0"

# Core
from .core import (
    AutoMLPipeline,
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
    AutoMLChainError,
    ValidationError,
    DatasetError,
    TrainingError,
    ProviderError,
    ConfigurationError,
    APIKeyError,
    AutoMLChainTimeoutError,
    TimeoutError,  # Deprecated alias
    JobCancelledError,
)

# Datasets
from .datasets import (
    Dataset,
    DatasetManager,
    DatasetStats,
    ValidationResult,
)

# Prompts
from .prompts import (
    PromptTemplate,
    PromptEngine,
)

# Training
from .training import (
    TrainingOrchestrator,
    TrainingJobInfo,
    TrainingResult,
    ProgressCallback,
    LoggingCallback,
)

# Evaluation
from .evaluation import (
    EvaluationSuite,
    EvalResult,
    RMSE,
    F1,
    MAE,
    Accuracy,
)

# Providers
from .providers import (
    BaseProvider,
    ProviderRegistry,
    ReplicateProvider,
    MockProvider,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "AutoMLPipeline",
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
    # Datasets
    "Dataset",
    "DatasetManager",
    "DatasetStats",
    "ValidationResult",
    # Prompts
    "PromptTemplate",
    "PromptEngine",
    # Training
    "TrainingOrchestrator",
    "TrainingJobInfo",
    "TrainingResult",
    "ProgressCallback",
    "LoggingCallback",
    # Evaluation
    "EvaluationSuite",
    "EvalResult",
    "RMSE",
    "F1",
    "MAE",
    "Accuracy",
    # Providers
    "BaseProvider",
    "ProviderRegistry",
    "ReplicateProvider",
    "MockProvider",
]
