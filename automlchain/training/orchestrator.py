"""Training orchestrator for AutoMLChain.

Coordinates training across providers with progress tracking.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import structlog

from .callbacks import BaseCallback, CallbackManager, CallbackEvent
from .jobs import JobState, TrainingJobInfo, TrainingResult, TrainingMetrics
from ..core.config import HyperParams
from ..core.exceptions import TrainingError, JobCancelledError, AutoMLChainTimeoutError
from ..providers.base import BaseProvider, TrainingJob, JobStatus
from ..datasets.types import Dataset
from ..prompts.templates import PromptTemplate

if TYPE_CHECKING:
    from ..prompts.templates import PromptTemplate

logger = structlog.get_logger(__name__)


class TrainingOrchestrator:
    """Orchestrates training jobs across providers.

    Manages job lifecycle, progress polling, and callback dispatch.

    Example:
        >>> from automlchain.providers import MockProvider
        >>> provider = MockProvider()
        >>> orchestrator = TrainingOrchestrator(provider)
        >>> job = orchestrator.train(dataset, config)
        >>> result = orchestrator.wait_for_completion(job.job_id)
    """

    def __init__(
        self,
        provider: BaseProvider,
        *,
        callbacks: list[BaseCallback] | None = None,
        poll_interval: float = 5.0,
    ) -> None:
        """
        Args:
            provider: Provider instance for training.
            callbacks: List of callbacks for progress updates.
            poll_interval: Seconds between status polls.
        """
        self.provider = provider
        self.poll_interval = poll_interval
        self.callback_manager = CallbackManager(callbacks)

        # Track jobs
        self._jobs: dict[str, TrainingJobInfo] = {}
        self._lock = threading.Lock()

    def train(
        self,
        dataset: Dataset,
        template: PromptTemplate | None,
        hyperparams: HyperParams,
        model: str,
        **kwargs: Any,
    ) -> TrainingJob:
        """Start a training job.

        Args:
            dataset: Dataset to train on.
            template: Optional prompt template.
            hyperparams: Training hyperparameters.
            model: Model identifier.
            **kwargs: Additional training options.

        Returns:
            TrainingJob with job_id for tracking.
        """
        logger.info("starting_training", model=model)

        # Convert dataset to training file
        training_file = self._prepare_training_data(dataset, template)

        # Start training
        try:
            job = self.provider.train(
                model=model,
                training_file=training_file,
                hyperparameters=hyperparams.to_dict(),
                **kwargs,
            )
        except Exception as e:
            raise TrainingError(
                f"Failed to start training: {e}",
                provider=self.provider.name,
                cause=str(e),
            ) from e

        # Track job
        job_info = TrainingJobInfo(
            job_id=job.job_id,
            state=JobState.PENDING,
            model=model,
            provider=self.provider.name,
            hyperparameters=hyperparams.to_dict(),
        )

        with self._lock:
            self._jobs[job.job_id] = job_info

        # Notify callbacks
        self.callback_manager.on_job_started(
            CallbackEvent(
                event_type="job_started",
                job_id=job.job_id,
                data={"model": model, "hyperparameters": hyperparams.to_dict()},
            )
        )

        return job

    def _prepare_training_data(
        self,
        dataset: Dataset,
        template: PromptTemplate | None,
    ) -> str:
        """Prepare dataset for training.

        Converts dataset to provider-specific format.

        Args:
            dataset: Input dataset.
            template: Optional prompt template.

        Returns:
            Path or URL to training data.
        """
        # For now, return the dataset path
        # In real implementation, would upload to provider
        if dataset.path:
            return dataset.path

        # If data is in memory, would upload to temporary storage
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for sample in dataset.data:
                if template:
                    # Format using template
                    formatted = template.format(**sample)
                    sample = {"text": formatted}
                f.write(json.dumps(sample) + "\n")
            return f.name

    def get_status(self, job_id: str) -> TrainingJobInfo:
        """Get the current status of a job.

        Args:
            job_id: ID of the job to check.

        Returns:
            TrainingJobInfo with current state.
        """
        # Check cached state
        with self._lock:
            if job_id in self._jobs:
                cached = self._jobs[job_id]

                # If terminal, return cached
                if cached.is_terminal:
                    return cached

        # Get fresh status from provider
        try:
            status = self.provider.get_job_status(job_id)
        except Exception as e:
            raise TrainingError(
                f"Failed to get job status: {e}",
                provider=self.provider.name,
                job_id=job_id,
                cause=str(e),
            ) from e

        # Map provider status to JobState
        state = self._map_status_to_state(status)

        # Update job info
        metrics = TrainingMetrics(
            loss=status.loss,
            epoch=status.epoch,
            step=status.step,
        )

        with self._lock:
            if job_id in self._jobs:
                job_info = self._jobs[job_id]
                job_info.state = state
                job_info.metrics = metrics

                if status.started_at and not job_info.started_at:
                    job_info.started_at = status.started_at

                if status.error:
                    job_info.error = status.error

                return job_info
            else:
                # Create new job info if not tracked
                job_info = TrainingJobInfo(
                    job_id=job_id,
                    state=state,
                    model="unknown",  # Unknown without original training call
                    provider=self.provider.name,
                    metrics=metrics,
                    error=status.error,
                )
                self._jobs[job_id] = job_info
                return job_info

    def _map_status_to_state(self, status: JobStatus) -> JobState:
        """Map provider status to JobState."""
        status_map = {
            "pending": JobState.PENDING,
            "queued": JobState.QUEUED,
            "running": JobState.RUNNING,
            "completed": JobState.COMPLETED,
            "failed": JobState.FAILED,
            "cancelled": JobState.CANCELLED,
            "canceled": JobState.CANCELLED,
        }
        return status_map.get(status.status.lower(), JobState.PENDING)

    def cancel(self, job_id: str) -> None:
        """Cancel a running job.

        Args:
            job_id: ID of the job to cancel.

        Raises:
            JobCancelledError: If cancellation fails.
        """
        logger.info("cancelling_job", job_id=job_id)

        try:
            self.provider.cancel_job(job_id)
        except Exception as e:
            raise JobCancelledError(job_id) from e

        # Update local state
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].state = JobState.CANCELLED

        # Notify callbacks
        self.callback_manager.on_job_cancelled(
            CallbackEvent(
                event_type="job_cancelled",
                job_id=job_id,
            )
        )

    def wait_for_completion(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> TrainingResult:
        """Wait for a job to complete.

        Polls status until job reaches terminal state or timeout.

        Args:
            job_id: ID of the job to wait for.
            timeout: Maximum seconds to wait. None = wait forever.
            poll_interval: Seconds between polls. None = use default.

        Returns:
            TrainingResult with final job status.

        Raises:
            AutoMLChainTimeoutError: If timeout is reached.
            JobCancelledError: If job is cancelled.
        """
        poll_interval = poll_interval or self.poll_interval
        start_time = time.time()

        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise AutoMLChainTimeoutError(
                    f"Job {job_id} did not complete within {timeout}s",
                    seconds=timeout,
                )

            # Get status
            job_info = self.get_status(job_id)

            # Dispatch progress callback
            if not job_info.is_terminal:
                self.callback_manager.on_job_progress(
                    CallbackEvent(
                        event_type="job_progress",
                        job_id=job_id,
                        data={
                            "progress": job_info.progress,
                            "epoch": job_info.metrics.epoch if job_info.metrics else 0,
                            "total_epochs": job_info.hyperparameters.get("epochs", 1),
                            "loss": job_info.metrics.loss if job_info.metrics else None,
                        },
                    )
                )

            # Check terminal states
            if job_info.state == JobState.COMPLETED:
                result = TrainingResult(
                    job_id=job_id,
                    status=job_info.state,
                    metrics=job_info.metrics,
                    duration_seconds=job_info.duration_seconds,
                )

                self.callback_manager.on_job_completed(
                    CallbackEvent(
                        event_type="job_completed",
                        job_id=job_id,
                        data={
                            "duration_seconds": result.duration_seconds,
                        },
                    )
                )

                return result

            if job_info.state == JobState.FAILED:
                self.callback_manager.on_job_failed(
                    CallbackEvent(
                        event_type="job_failed",
                        job_id=job_id,
                        data={"error": job_info.error},
                    )
                )

                raise TrainingError(
                    f"Training failed: {job_info.error}",
                    provider=self.provider.name,
                    job_id=job_id,
                )

            if job_info.state == JobState.CANCELLED:
                raise JobCancelledError(job_id)

            # Wait before next poll
            time.sleep(poll_interval)

    def add_callback(self, callback: BaseCallback) -> None:
        """Add a progress callback."""
        self.callback_manager.add(callback)

    def remove_callback(self, callback: BaseCallback) -> None:
        """Remove a progress callback."""
        self.callback_manager.remove(callback)

    def get_job(self, job_id: str) -> TrainingJobInfo | None:
        """Get job info from cache."""
        with self._lock:
            return self._jobs.get(job_id)
