"""Together AI provider implementation for AutoMLChain."""

from __future__ import annotations

import time
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseProvider, TrainingJob, JobStatus, DeployedModel
from ..core.exceptions import ProviderError, TrainingError

logger = structlog.get_logger(__name__)


class TogetherProvider(BaseProvider):
    """Provider implementation for Together AI API.

    Together AI provides simple fine-tuning with LoRA support.
    Much simpler than Replicate for fine-tuning workflows.

    API Reference: https://docs.together.ai/
    """

    BASE_URL = "https://api.together.ai"

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str = "Qwen/Qwen3.5-9B",
        webhook_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        """
        Args:
            api_key: Together AI API token.
            default_model: Default model for fine-tuning.
            webhook_url: Optional webhook for training updates.
            timeout: Request timeout in seconds.
        """
        super().__init__(api_key)
        self.default_model = default_model
        self.webhook_url = webhook_url
        self.timeout = timeout
        self._client: Any = None

    @property
    def name(self) -> str:
        return "together"

    @property
    def client(self) -> Any:
        """Get or create Together AI client."""
        if self._client is None:
            try:
                from together import Together
                self._client = Together(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "together package not installed. Install with: pip install together"
                )
        return self._client

    def upload_dataset(self, file_path: str) -> str:
        """Upload a dataset file to Together AI.

        Args:
            file_path: Path to the JSONL dataset file.

        Returns:
            File ID for the uploaded dataset.
        """
        logger.info("uploading_dataset", file=file_path)

        try:
            response = self.client.files.upload(
                file=file_path,
                purpose="fine-tune",
            )
            file_id = response.id
            logger.info("dataset_uploaded", file_id=file_id)

            # Wait for processing to complete
            self._wait_for_file_processing(file_id)

            return file_id

        except Exception as e:
            raise TrainingError(
                f"Failed to upload dataset: {e}",
                provider="together",
                cause=str(e),
            )

    def _wait_for_file_processing(self, file_id: str, timeout: int = 300) -> None:
        """Wait for file processing to complete.

        Args:
            file_id: File ID to check.
            timeout: Maximum wait time in seconds.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            meta = self.client.files.retrieve(file_id)

            if meta.processing_status == "COMPLETED":
                logger.info("file_processing_complete", file_id=file_id)
                return

            if meta.processing_status == "INVALID_FORMAT":
                raise TrainingError(
                    f"Invalid dataset format: {meta.validation_report}",
                    provider="together",
                )

            if meta.processing_status == "FAILED":
                raise TrainingError(
                    f"File processing failed: {meta.processing_status}",
                    provider="together",
                )

            time.sleep(5)

        raise TrainingError(
            "File processing timed out",
            provider="together",
        )

    def train(
        self,
        *,
        model: str,
        training_file: str,
        hyperparameters: dict[str, Any] | None = None,
        webhook_url: str | None = None,
        **kwargs: Any,
    ) -> TrainingJob:
        """Start a fine-tuning training job on Together AI.

        Args:
            model: Model identifier (e.g., "Qwen/Qwen3.5-9B").
            training_file: File ID or path to training data in JSONL format.
            hyperparameters: Training hyperparameters.
            webhook_url: Optional webhook for status updates.
            **kwargs: Additional Together-specific parameters.

        Returns:
            TrainingJob with job_id for tracking.

        Raises:
            TrainingError: If training fails to start.
        """
        logger.info(
            "starting_training",
            provider="together",
            model=model,
        )

        # Handle file path vs file ID
        if not training_file.startswith("file-"):
            # It's a file path, upload it
            training_file = self.upload_dataset(training_file)

        # Default hyperparameters for Together
        if hyperparameters is None:
            hyperparameters = {}

        # Build training request
        train_params = {
            "training_file": training_file,
            "model": model,
            "n_epochs": hyperparameters.get("epochs", 3),
            "learning_rate": hyperparameters.get("learning_rate", 1e-5),
            "batch_size": hyperparameters.get("batch_size", "max"),
            "warmup_ratio": hyperparameters.get("warmup_ratio", 0.0),
            "lora": hyperparameters.get("lora", True),
        }

        # Optional parameters
        if hyperparameters.get("lora_rank"):
            train_params["lora_rank"] = hyperparameters["lora_rank"]
        if hyperparameters.get("lora_alpha"):
            train_params["lora_alpha"] = hyperparameters["lora_alpha"]
        if hyperparameters.get("weight_decay"):
            train_params["weight_decay"] = hyperparameters["weight_decay"]
        if hyperparameters.get("max_grad_norm"):
            train_params["max_grad_norm"] = hyperparameters["max_grad_norm"]
        if hyperparameters.get("suffix"):
            train_params["suffix"] = hyperparameters["suffix"]

        if webhook_url or self.webhook_url:
            train_params["webhook"] = webhook_url or self.webhook_url

        # Merge any additional kwargs
        train_params.update(kwargs)

        try:
            # Create fine-tuning job
            job_response = self.client.fine_tuning.create(**train_params)
            job_id = job_response.id

            job = TrainingJob(
                job_id=job_id,
                provider="together",
                model=model,
                status="pending",
                metadata={
                    "training_file": training_file,
                    "hyperparameters": hyperparameters,
                },
            )

            logger.info("training_job_created", job_id=job_id)
            return job

        except Exception as e:
            raise TrainingError(
                f"Failed to start training: {e}",
                provider="together",
                cause=str(e),
                suggestion="Check your API key and training file",
            )

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a fine-tuning job.

        Args:
            job_id: ID of the job to check.

        Returns:
            JobStatus with current state.

        Raises:
            ProviderError: If status check fails.
        """
        logger.debug("checking_job_status", job_id=job_id)

        try:
            status_response = self.client.fine_tuning.retrieve(id=job_id)

            # Map Together status to our status
            together_status = status_response.status
            status_map = {
                "pending": "queued",
                "queued": "queued",
                "running": "running",
                "uploading": "processing",
                "completed": "completed",
                "error": "failed",
                "cancelled": "cancelled",
            }
            mapped_status = status_map.get(together_status, together_status)

            # Extract progress from events
            progress = 0.0
            current_epoch = 0
            total_epochs = status_response.n_epochs or 1
            logs = []

            # Try to get events for progress
            try:
                events = self.client.fine_tuning.list_events(id=job_id)
                for event in events.data[-10:]:  # Last 10 events
                    logs.append(event.message)
                    if "Epoch completed" in event.message:
                        current_epoch += 1
                        progress = min((current_epoch / total_epochs) * 100, 99)

                if together_status == "completed":
                    progress = 100.0
                    current_epoch = total_epochs
            except Exception:
                pass

            status = JobStatus(
                status=mapped_status,
                progress=float(progress),
                epoch=current_epoch,
                total_epochs=total_epochs,
                logs=logs[-5:] if logs else [],
                error=status_response.error,
            )

            return status

        except Exception as e:
            raise ProviderError(
                f"Failed to get job status: {e}",
                provider="together",
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def cancel_job(self, job_id: str) -> None:
        """Cancel a running fine-tuning job.

        Args:
            job_id: ID of the job to cancel.

        Raises:
            ProviderError: If cancellation fails.
        """
        logger.info("cancelling_job", job_id=job_id)

        try:
            self.client.fine_tuning.cancel(id=job_id)
            logger.info("job_cancelled", job_id=job_id)

        except Exception as e:
            raise ProviderError(
                f"Failed to cancel job: {e}",
                provider="together",
            )

    def deploy(
        self,
        *,
        model_path: str | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> DeployedModel:
        """Deploy a fine-tuned model.

        Note: Together AI deploys models automatically after training.
        This method is for reference and custom deployments.

        Args:
            model_path: Path to fine-tuned model files.
            job_id: Training job ID to deploy the output from.
            **kwargs: Additional deployment options.

        Returns:
            DeployedModel with inference endpoint.
        """
        logger.info("deploying_model", job_id=job_id, model_path=model_path)

        try:
            # Get model name from job if job_id provided
            if job_id:
                status = self.client.fine_tuning.retrieve(id=job_id)
                model_name = status.x_model_output_name
            else:
                model_name = model_path

            if not model_name:
                raise ProviderError(
                    "No model to deploy. Provide job_id or model_path.",
                    provider="together",
                )

            # For Together, the model is already deployed after training
            # Just create the DeployedModel wrapper
            deployed = DeployedModel(
                model_id=model_name,
                endpoint="https://api.together.ai/v1/chat/completions",
                provider="together",
                status="ready",
                cost_per_1k_tokens=0.001,  # Approximate Together inference cost
                metadata={
                    "job_id": job_id,
                    "model_name": model_name,
                },
            )

            logger.info("model_deployed", model_id=model_name)
            return deployed

        except Exception as e:
            raise ProviderError(
                f"Deployment failed: {e}",
                provider="together",
            )

    def infer(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Run inference on a deployed model.

        Args:
            prompt: Input prompt.
            model: Model name (uses default if not specified).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional inference parameters.

        Returns:
            Generated text response.
        """
        model = model or self.default_model

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content

        except Exception as e:
            raise ProviderError(
                f"Inference failed: {e}",
                provider="together",
            )
