"""Training module for AutoMLChain.

Handles training orchestration, job management, and callbacks.
"""

from .callbacks import (
    BaseCallback,
    CallbackEvent,
    ProgressCallback,
    LoggingCallback,
    WebhookCallback,
    CallbackManager,
)
from .jobs import (
    JobState,
    JobStateTransition,
    TrainingMetrics,
    TrainingCheckpoint,
    TrainingJobInfo,
    TrainingResult,
)
from .orchestrator import TrainingOrchestrator

__all__ = [
    # Callbacks
    "BaseCallback",
    "CallbackEvent",
    "ProgressCallback",
    "LoggingCallback",
    "WebhookCallback",
    "CallbackManager",
    # Jobs
    "JobState",
    "JobStateTransition",
    "TrainingMetrics",
    "TrainingCheckpoint",
    "TrainingJobInfo",
    "TrainingResult",
    # Orchestrator
    "TrainingOrchestrator",
]
