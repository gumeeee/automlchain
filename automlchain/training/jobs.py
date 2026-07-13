"""Training job definitions for AutoMLChain."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobState(str, Enum):
    """Possible states for a training job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStateTransition:
    """Defines valid state transitions."""

    TRANSITIONS = {
        JobState.PENDING: {JobState.QUEUED, JobState.CANCELLED},
        JobState.QUEUED: {JobState.RUNNING, JobState.CANCELLED},
        JobState.RUNNING: {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED},
        JobState.COMPLETED: set(),  # Terminal state
        JobState.FAILED: set(),  # Terminal state
        JobState.CANCELLED: set(),  # Terminal state
    }

    @classmethod
    def can_transition(cls, from_state: JobState, to_state: JobState) -> bool:
        """Check if a state transition is valid."""
        return to_state in cls.TRANSITIONS.get(from_state, set())


@dataclass
class TrainingMetrics:
    """Metrics collected during training.

    Attributes:
        loss: Current loss value.
        learning_rate: Current learning rate.
        epoch: Current epoch number.
        step: Current step number.
        gpu_memory: GPU memory usage in MB.
        throughput: Tokens per second.
    """

    loss: float | None = None
    learning_rate: float | None = None
    epoch: int = 0
    step: int = 0
    gpu_memory: float | None = None
    throughput: float | None = None
    custom: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "epoch": self.epoch,
            "step": self.step,
            "gpu_memory": self.gpu_memory,
            "throughput": self.throughput,
            **self.custom,
        }


@dataclass
class TrainingCheckpoint:
    """Represents a training checkpoint.

    Attributes:
        checkpoint_id: Unique identifier.
        path: Path to checkpoint files.
        epoch: Epoch at which checkpoint was saved.
        step: Step at which checkpoint was saved.
        metrics: Metrics at checkpoint time.
        created_at: Timestamp when checkpoint was created.
    """

    checkpoint_id: str
    path: str
    epoch: int
    step: int
    metrics: TrainingMetrics | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "path": self.path,
            "epoch": self.epoch,
            "step": self.step,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "created_at": self.created_at,
        }


@dataclass
class TrainingJobInfo:
    """Extended information about a training job.

    Attributes:
        job_id: Unique job identifier.
        state: Current job state.
        model: Model being trained.
        provider: Provider name.
        hyperparameters: Training configuration.
        created_at: When job was created.
        started_at: When training started.
        completed_at: When training completed.
        estimated_completion: Estimated completion time.
        metrics: Current training metrics.
        checkpoints: List of saved checkpoints.
        error: Error message if failed.
    """

    job_id: str
    state: JobState
    model: str
    provider: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    started_at: str | None = None
    completed_at: str | None = None
    estimated_completion: str | None = None
    metrics: TrainingMetrics | None = None
    checkpoints: list[TrainingCheckpoint] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}

    @property
    def progress(self) -> float:
        """Calculate progress percentage."""
        if self.state == JobState.COMPLETED:
            return 100.0
        if self.state in {JobState.PENDING, JobState.QUEUED}:
            return 0.0

        if self.metrics and self.state == JobState.RUNNING:
            total_epochs = self.hyperparameters.get("epochs", 1)
            if total_epochs > 0:
                epoch_progress = self.metrics.epoch / total_epochs
                return min(100.0, epoch_progress * 100)

        return 0.0

    @property
    def duration_seconds(self) -> float | None:
        """Calculate job duration in seconds."""
        if not self.started_at:
            return None

        start = time.mktime(time.strptime(self.started_at, "%Y-%m-%d %H:%M:%S"))

        if self.completed_at:
            end = time.mktime(time.strptime(self.completed_at, "%Y-%m-%d %H:%M:%S"))
        else:
            end = time.time()

        return end - start

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "model": self.model,
            "provider": self.provider,
            "hyperparameters": self.hyperparameters,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "estimated_completion": self.estimated_completion,
            "progress": self.progress,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class TrainingResult:
    """Result of a completed training job.

    Attributes:
        job_id: Job identifier.
        status: Final status.
        model_path: Path to trained model.
        checkpoint_url: URL to download checkpoints.
        metrics: Final training metrics.
        evaluation_results: Results of evaluation if performed.
        cost: Total training cost.
        duration_seconds: Training duration.
    """

    job_id: str
    status: JobState
    model_path: str | None = None
    checkpoint_url: str | None = None
    metrics: TrainingMetrics | None = None
    evaluation_results: dict[str, float] | None = None
    cost: float | None = None
    duration_seconds: float | None = None
    logs: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Check if training was successful."""
        return self.status == JobState.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "model_path": self.model_path,
            "checkpoint_url": self.checkpoint_url,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "evaluation_results": self.evaluation_results,
            "cost": self.cost,
            "duration_seconds": self.duration_seconds,
            "logs": self.logs,
        }
