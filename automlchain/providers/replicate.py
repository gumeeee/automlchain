"""Replicate provider implementation for AutoMLChain."""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseProvider, TrainingJob, JobStatus, DeployedModel
from ..core.exceptions import ProviderError, TrainingError

logger = structlog.get_logger(__name__)


class ReplicateProvider(BaseProvider):
    """Provider implementation for Replicate API.

    Supports fine-tuning on Replicate's cloud infrastructure.

    API Reference: https://replicate.com/docs/api-reference
    """

    BASE_URL = "https://api.replicate.com"

    def __init__(
        self,
        *,
        api_key: str,
        webhook_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """
        Args:
            api_key: Replicate API token.
            webhook_url: Optional webhook for training updates.
            timeout: Request timeout in seconds.
        """
        super().__init__(api_key)
        self.webhook_url = webhook_url
        self.timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def name(self) -> str:
        return "replicate"

    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    def train(
        self,
        *,
        model: str,
        training_file: str,
        hyperparameters: dict[str, Any] | None = None,
        webhook_url: str | None = None,
        **kwargs: Any,
    ) -> TrainingJob:
        """Start a fine-tuning training job on Replicate.

        Args:
            model: Model identifier (e.g., "meta/llama-3-8b-instruct").
            training_file: URL or path to training data in JSONL format.
            hyperparameters: Training hyperparameters (learning_rate, epochs, etc.).
            webhook_url: Optional webhook for status updates.
            **kwargs: Additional Replicate-specific parameters.

        Returns:
            TrainingJob with job_id for tracking.

        Raises:
            TrainingError: If training fails to start.
        """
        logger.info(
            "starting_training",
            provider="replicate",
            model=model,
        )

        # Default hyperparameters for Replicate
        if hyperparameters is None:
            hyperparameters = {}

        # Prepare request
        # Note: In real implementation, this would use Replicate's training API
        # For now, we create a mock job structure
        job_id = f"train_{uuid.uuid4().hex[:12]}"

        # Map model to Replicate's format
        replicate_model = self._map_model(model)

        # Build training request
        payload = {
            "version": replicate_model,
            "input": {
                "train_data": training_file,
                **hyperparameters,
            },
        }

        if webhook_url or self.webhook_url:
            payload["webhook"] = webhook_url or self.webhook_url

        try:
            # Make real API call to create training
            response = self.client.post(
                f"{self.BASE_URL}/v1/trainings",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Extract job info from response
            job_id = data["id"]
            status = data.get("status", "queued")

            # Get training output if completed
            checkpoint_url = None
            if status == "succeeded":
                checkpoint_url = data.get("output")

            job = TrainingJob(
                job_id=job_id,
                provider="replicate",
                model=model,
                status=status,
                created_at=data.get("created_at"),
                started_at=data.get("started_at"),
                completed_at=data.get("completed_at"),
                cost=float(data.get("usage", {}).get("cost", 0) or 0),
                checkpoint_url=checkpoint_url,
                metadata={
                    "replicate_model": replicate_model,
                    "hyperparameters": hyperparameters,
                },
            )

            logger.info("training_job_created", job_id=job_id)
            return job

        except httpx.HTTPStatusError as e:
            raise TrainingError(
                f"Failed to start training: {e.response.text}",
                provider="replicate",
                cause=str(e),
                suggestion="Check your API key and training file URL",
            )
        except Exception as e:
            raise TrainingError(
                f"Training failed to start: {e}",
                provider="replicate",
                cause=str(e),
            )

    def _map_model(self, model: str) -> str:
        """Map model identifier to Replicate's format.

        Args:
            model: Model identifier.

        Returns:
            Replicate model version string.
        """
        model_map = {
            "meta/llama-3-8b": "meta/llama-3-8b-instruct",
            "meta/llama-3-8b-instruct": "meta/llama-3-8b-instruct",
            "mistralai/mistral-7b": "mistralai/mistral-7b-v0.1",
            "mistralai/mistral-7b-v0.1": "mistralai/mistral-7b-v0.1",
        }
        return model_map.get(model, model)

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a training job.

        Args:
            job_id: ID of the job to check.

        Returns:
            JobStatus with current state.

        Raises:
            ProviderError: If status check fails.
        """
        logger.debug("checking_job_status", job_id=job_id)

        try:
            # Make real API call to get training status
            response = self.client.get(f"{self.BASE_URL}/v1/trainings/{job_id}")
            response.raise_for_status()
            data = response.json()

            # Parse status and progress
            api_status = data.get("status", "unknown")
            progress = data.get("progress", 0) or 0.0

            # Get logs if available
            logs = data.get("logs", "")

            # Extract metrics
            metrics = data.get("metrics", {})

            status = JobStatus(
                status=api_status,
                progress=float(progress),
                metrics=metrics,
                logs=[logs] if logs else [],
                error=data.get("error"),
            )

            return status

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ProviderError(
                    f"Training job not found: {job_id}",
                    provider="replicate",
                    status_code=404,
                )
            raise ProviderError(
                f"Failed to get job status: {e.response.text}",
                provider="replicate",
                status_code=e.response.status_code,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def cancel_job(self, job_id: str) -> None:
        """Cancel a running training job.

        Args:
            job_id: ID of the job to cancel.

        Raises:
            ProviderError: If cancellation fails.
        """
        logger.info("cancelling_job", job_id=job_id)

        try:
            # Make real API call to cancel training
            response = self.client.post(f"{self.BASE_URL}/v1/trainings/{job_id}/cancel")
            response.raise_for_status()

            logger.info("job_cancelled", job_id=job_id)

        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Failed to cancel job: {e.response.text}",
                provider="replicate",
                status_code=e.response.status_code,
            )

    def deploy(
        self,
        *,
        model_path: str | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> DeployedModel:
        """Deploy a fine-tuned model.

        Args:
            model_path: Path to fine-tuned model (for local deployment).
            job_id: Training job ID to deploy.
            **kwargs: Additional deployment options.

        Returns:
            DeployedModel with inference endpoint.

        Raises:
            ProviderError: If deployment fails.
        """
        logger.info("deploying_model", job_id=job_id, model_path=model_path)

        try:
            # In real implementation:
            # If job_id provided, get model from training output
            # If model_path provided, upload and deploy

            # Mock deployment
            model_id = f"model_{uuid.uuid4().hex[:12]}"
            endpoint = f"https://api.replicate.com/v1/models/{model_id}/predictions"

            deployed = DeployedModel(
                model_id=model_id,
                endpoint=endpoint,
                provider="replicate",
                status="ready",
                cost_per_1k_tokens=0.002,
                metadata={
                    "job_id": job_id,
                    "model_path": model_path,
                },
            )

            logger.info("model_deployed", model_id=model_id)
            return deployed

        except Exception as e:
            raise ProviderError(
                f"Deployment failed: {e}",
                provider="replicate",
            )


class ReplicateInferenceClient:
    """Client for running inference on Replicate models.

    Example:
        >>> client = ReplicateInferenceClient(api_key="...")
        >>> result = client.predict(
        ...     model="my/model:latest",
        ...     input={"text": "Hello, world!"}
        ... )
    """

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        """
        Args:
            api_key: Replicate API token.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def predict(
        self,
        model: str,
        input_data: dict[str, Any],
        *,
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        """Run inference on a model.

        Args:
            model: Model identifier.
            input_data: Input data for prediction.
            webhook_url: Optional webhook for async results.

        Returns:
            Prediction results.
        """
        payload = {
            "version": model,
            "input": input_data,
        }

        if webhook_url:
            payload["webhook"] = webhook_url

        # Create prediction
        response = self.client.post(
            f"{ReplicateProvider.BASE_URL}/v1/predictions",
            json=payload,
        )
        response.raise_for_status()
        prediction = response.json()

        # Poll for completion
        if not webhook_url:
            prediction = self._poll_prediction(prediction["urls"]["get"])

        return prediction

    def _poll_prediction(self, url: str) -> dict[str, Any]:
        """Poll prediction until complete."""
        while True:
            response = self.client.get(url)
            response.raise_for_status()
            data = response.json()

            if data["status"] == "succeeded":
                return data
            elif data["status"] == "failed":
                raise ProviderError(
                    f"Prediction failed: {data.get('error')}",
                    provider="replicate",
                )
            elif data["status"] == "canceled":
                raise ProviderError(
                    "Prediction was canceled",
                    provider="replicate",
                )

            time.sleep(2)
