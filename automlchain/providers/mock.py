"""Mock provider for testing and development."""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog

from .base import BaseProvider, TrainingJob, JobStatus, DeployedModel

logger = structlog.get_logger(__name__)


class MockProvider(BaseProvider):
    """Mock provider for testing without API calls.

    Simulates training jobs and deployments locally.

    Example:
        >>> provider = MockProvider(api_key="mock-key")
        >>> job = provider.train(model="meta/llama-3-8b", training_file="data.jsonl")
        >>> status = provider.get_job_status(job.job_id)
    """

    def __init__(
        self,
        *,
        api_key: str = "mock-key",
        simulate_duration: float = 10.0,
        failure_rate: float = 0.0,
    ) -> None:
        """
        Args:
            api_key: API key (ignored for mock).
            simulate_duration: Simulated training duration in seconds.
            failure_rate: Probability of simulated failures (0.0 to 1.0).
        """
        super().__init__(api_key)
        self.simulate_duration = simulate_duration
        self.failure_rate = failure_rate
        self._jobs: dict[str, dict[str, Any]] = {}
        self._deployed_models: dict[str, DeployedModel] = {}

    @property
    def name(self) -> str:
        return "mock"

    def train(
        self,
        *,
        model: str,
        training_file: str,
        hyperparameters: dict[str, Any] | None = None,
        webhook_url: str | None = None,
        **kwargs: Any,
    ) -> TrainingJob:
        """Create a mock training job.

        Args:
            model: Model identifier.
            training_file: Training file path.
            hyperparameters: Training hyperparameters.
            webhook_url: Webhook URL (ignored).
            **kwargs: Additional parameters.

        Returns:
            Mock TrainingJob.
        """
        job_id = f"mock_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        job = TrainingJob(
            job_id=job_id,
            provider="mock",
            model=model,
            status="queued",
            metadata={
                "training_file": training_file,
                "hyperparameters": hyperparameters or {},
                "start_time": start_time,
            },
        )

        self._jobs[job_id] = {
            "job": job,
            "start_time": start_time,
            "progress": 0.0,
        }

        logger.info(
            "mock_training_started",
            job_id=job_id,
            model=model,
        )

        return job

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get mock job status with simulated progress.

        Args:
            job_id: ID of the job to check.

        Returns:
            JobStatus with simulated progress.
        """
        if job_id not in self._jobs:
            return JobStatus(
                status="not_found",
                error=f"Job {job_id} not found",
            )

        job_data = self._jobs[job_id]
        job = job_data["job"]
        start_time = job_data["start_time"]

        elapsed = time.time() - start_time
        progress = min(100.0, (elapsed / self.simulate_duration) * 100)

        # Simulate completion or failure
        if progress >= 100:
            if self.failure_rate > 0 and hash(job_id) % 100 < self.failure_rate * 100:
                job.status = "failed"
                job.error = "Simulated training failure"
                status = JobStatus(
                    status="failed",
                    progress=100.0,
                    error=job.error,
                )
            else:
                job.status = "completed"
                job.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
                job.checkpoint_url = f"https://mock.example.com/checkpoints/{job_id}"
                status = JobStatus(
                    status="completed",
                    progress=100.0,
                    epoch=3,
                    total_epochs=3,
                    loss=0.1,
                )
        else:
            job.status = "running"
            job.started_at = job.started_at or time.strftime("%Y-%m-%d %H:%M:%S")

            # Simulate training metrics
            current_epoch = max(1, int((progress / 100) * 3))
            current_step = int((progress / 100) * 1000)
            current_loss = max(0.1, 1.0 - (progress / 100) * 0.9)

            status = JobStatus(
                status="running",
                progress=progress,
                epoch=current_epoch,
                total_epochs=3,
                step=current_step,
                total_steps=1000,
                loss=current_loss,
            )

        return status

    def cancel_job(self, job_id: str) -> None:
        """Cancel a mock training job.

        Args:
            job_id: ID of the job to cancel.
        """
        if job_id not in self._jobs:
            logger.warning("mock_cancel_not_found", job_id=job_id)
            return

        job = self._jobs[job_id]["job"]
        job.status = "cancelled"

        logger.info("mock_job_cancelled", job_id=job_id)

    def deploy(
        self,
        *,
        model_path: str | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> DeployedModel:
        """Create a mock deployed model.

        Args:
            model_path: Model path (ignored).
            job_id: Training job ID.
            **kwargs: Additional parameters.

        Returns:
            Mock DeployedModel.
        """
        model_id = f"mock_model_{uuid.uuid4().hex[:12]}"

        deployed = DeployedModel(
            model_id=model_id,
            endpoint=f"https://mock.example.com/models/{model_id}",
            provider="mock",
            status="ready",
            cost_per_1k_tokens=0.001,
            metadata={
                "job_id": job_id,
                "model_path": model_path,
            },
        )

        self._deployed_models[model_id] = deployed

        logger.info("mock_model_deployed", model_id=model_id)
        return deployed

    def predict(self, model_id: str, input_text: str) -> dict[str, Any]:
        """Run mock prediction.

        Args:
            model_id: ID of deployed model.
            input_text: Input text.

        Returns:
            Mock prediction result.
        """
        return {
            "output": f"Mock response to: {input_text[:50]}...",
            "model_id": model_id,
            "tokens_used": len(input_text.split()) * 2,
        }

    def clear_jobs(self) -> None:
        """Clear all mock jobs."""
        self._jobs.clear()

    def clear_models(self) -> None:
        """Clear all deployed models."""
        self._deployed_models.clear()
