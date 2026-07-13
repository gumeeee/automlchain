"""Base provider interface for AutoMLChain."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TrainingJob:
    """Represents a training job.

    Attributes:
        job_id: Unique identifier for the job.
        provider: Name of the provider.
        model: Model being trained.
        status: Current job status.
        created_at: Timestamp when job was created.
        started_at: Timestamp when training started.
        completed_at: Timestamp when training completed.
        cost: Estimated cost in USD.
        checkpoint_url: URL to download checkpoints.
        error: Error message if job failed.
    """

    job_id: str
    provider: str
    model: str
    status: str = "pending"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    started_at: str | None = None
    completed_at: str | None = None
    cost: float | None = None
    checkpoint_url: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in ("completed", "failed", "cancelled")

    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status in ("queued", "running", "processing")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "cost": self.cost,
            "checkpoint_url": self.checkpoint_url,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class JobStatus:
    """Status of a training job.

    Attributes:
        status: Current status (pending, queued, running, completed, failed, cancelled).
        progress: Progress percentage (0-100).
        epoch: Current epoch.
        total_epochs: Total number of epochs.
        step: Current step.
        total_steps: Total number of steps.
        loss: Current loss value.
        metrics: Additional metrics.
        logs: Recent log lines.
        error: Error message if failed.
    """

    status: str
    progress: float = 0.0
    epoch: int = 0
    total_epochs: int = 1
    step: int = 0
    total_steps: int = 0
    loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "progress": self.progress,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "step": self.step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "metrics": self.metrics,
            "logs": self.logs,
            "error": self.error,
        }


@dataclass
class DeployedModel:
    """Represents a deployed model.

    Attributes:
        model_id: Provider-assigned model ID.
        endpoint: Inference endpoint URL.
        provider: Provider name.
        created_at: When the model was deployed.
        status: Deployment status.
        cost_per_1k_tokens: Cost per 1000 tokens in USD.
    """

    model_id: str
    endpoint: str
    provider: str
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    status: str = "ready"
    cost_per_1k_tokens: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, input_text: str, **kwargs: Any) -> Any:
        """Run inference on the model.

        Args:
            input_text: Input text for prediction.
            **kwargs: Additional inference parameters.

        Returns:
            Model prediction.
        """
        raise NotImplementedError("Use inference client for predictions")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "endpoint": self.endpoint,
            "provider": self.provider,
            "created_at": self.created_at,
            "status": self.status,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "metadata": self.metadata,
        }


class BaseProvider(ABC):
    """Abstract base class for fine-tuning providers.

    Defines the interface that all providers must implement.

    Example:
        >>> class MyProvider(BaseProvider):
        ...     def train(self, config):
        ...         ...
    """

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initialize the provider.

        Args:
            api_key: API key for authentication.
            **kwargs: Additional provider-specific configuration.
        """
        self.api_key = api_key
        self.config = kwargs
        self._client: Any = None

    @property
    def name(self) -> str:
        """Provider name."""
        return self.__class__.__name__.replace("Provider", "").lower()

    @abstractmethod
    def train(
        self,
        *,
        model: str,
        training_file: str,
        hyperparameters: dict[str, Any] | None = None,
        webhook_url: str | None = None,
        **kwargs: Any,
    ) -> TrainingJob:
        """Start a training job.

        Args:
            model: Model identifier (e.g., "meta/llama-3-8b").
            training_file: URL or path to training data.
            hyperparameters: Training hyperparameters.
            webhook_url: Optional webhook for status updates.
            **kwargs: Additional provider-specific parameters.

        Returns:
            TrainingJob with job_id for tracking.
        """
        ...

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a training job.

        Args:
            job_id: ID of the job to check.

        Returns:
            JobStatus with current state.
        """
        ...

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Cancel a running training job.

        Args:
            job_id: ID of the job to cancel.
        """
        ...

    @abstractmethod
    def deploy(
        self,
        *,
        model_path: str | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> DeployedModel:
        """Deploy a fine-tuned model.

        Args:
            model_path: Path to model files (for local models).
            job_id: Training job ID to deploy.
            **kwargs: Additional deployment options.

        Returns:
            DeployedModel with inference endpoint.
        """
        ...

    def get_training_cost(
        self,
        model: str,
        epochs: int,
        batch_size: int,
    ) -> float:
        """Estimate training cost.

        Args:
            model: Model identifier.
            epochs: Number of epochs.
            batch_size: Batch size.

        Returns:
            Estimated cost in USD.
        """
        from .pricing import PricingProvider

        pricing = PricingProvider(self.name)
        return pricing.estimate_training_cost(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
        )

    def validate_model(self, model: str) -> bool:
        """Check if a model is available.

        Args:
            model: Model identifier.

        Returns:
            True if model is available.
        """
        # Override in subclasses with actual validation
        return True

    def get_client(self) -> Any:
        """Get or create the HTTP client.

        Returns:
            Configured HTTP client.
        """
        if self._client is None:
            import httpx
            self._client = httpx.Client(
                headers={"Authorization": f"Token {self.api_key}"},
                timeout=60.0,
            )
        return self._client
