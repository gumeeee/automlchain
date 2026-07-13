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

# Optional providers (may not be installed)
try:
    from .providers import TogetherProvider
except ImportError:
    TogetherProvider = None  # type: ignore

try:
    from .providers import LocalProvider
except ImportError:
    LocalProvider = None  # type: ignore

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
    "TogetherProvider",  # May be None if together not installed
    "LocalProvider",  # May be None if dependencies not installed
]
