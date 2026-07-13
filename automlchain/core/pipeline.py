"""Main pipeline orchestration for AutoMLChain."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import structlog

from .config import PipelineConfig, ProviderConfig

if TYPE_CHECKING:
    from ..datasets.manager import DatasetManager, Dataset
    from ..providers.base import BaseProvider
    from ..training.orchestrator import TrainingOrchestrator, TrainingJob
    from ..evaluation.suite import EvaluationSuite, EvalResult
    from ..prompts.engine import PromptEngine, PromptTemplate


logger = structlog.get_logger(__name__)


class AutoMLPipeline:
    """Main pipeline for AutoMLChain.

    Orchestrates the complete fine-tuning workflow:
    1. Upload and validate datasets
    2. Generate prompt templates
    3. Configure and run training
    4. Evaluate model performance
    5. Deploy to production

    Example:
        >>> from automlchain import AutoMLPipeline
        >>> pipeline = AutoMLPipeline(provider="replicate")
        >>> pipeline.upload_dataset("reviews.jsonl")
        >>> result = pipeline.train()
        >>> print(result.job_id)
    """

    def __init__(
        self,
        provider: str | ProviderConfig = "replicate",
        *,
        config: PipelineConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the pipeline.

        Args:
            provider: Provider name ("replicate") or ProviderConfig object.
            config: Optional PipelineConfig. If None, created from provider.
            **kwargs: Additional arguments passed to PipelineConfig.
        """
        self._setup_logging()

        # Handle provider argument
        from .config import ProviderType
        from ..providers.base import BaseProvider

        if isinstance(provider, str):
            # Map string to ProviderType
            if provider.lower() == "mock":
                provider_config = ProviderConfig(provider_type=ProviderType.REPLICATE, api_key="mock-key")
            else:
                provider_config = ProviderConfig(provider_type=ProviderType(provider))
            provider_name = provider
        elif isinstance(provider, BaseProvider):
            # Already a provider instance (e.g., MockProvider)
            # Store it directly and use its name
            self._provider = provider
            provider_config = ProviderConfig(provider_type=ProviderType.REPLICATE, api_key=provider.api_key)
            provider_name = provider.name
        else:
            # Already a ProviderConfig
            provider_config = provider
            provider_name = getattr(provider, 'provider_type', 'unknown')
            if hasattr(provider_name, 'value'):
                provider_name = provider_name.value

        # Build config
        if config is None:
            self.config = PipelineConfig(
                provider=provider_config,
                **{k: v for k, v in kwargs.items() if v is not None},
            )
        else:
            self.config = config

        # Initialize components
        self._provider_registry: Any = None
        self._provider: Any = None
        self._dataset_manager: Any = None
        self._prompt_engine: Any = None
        self._training_orchestrator: Any = None
        self._evaluation_suite: Any = None

        # State
        self._current_dataset: Any | None = None
        self._current_template: Any | None = None
        self._current_job: Any | None = None
        self._deployed_model: Any | None = None

        logger.info(
            "pipeline_initialized",
            provider=provider_name,
        )

    def _setup_logging(self) -> None:
        """Configure structured logging."""
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    @property
    def provider_registry(self) -> Any:
        """Lazy-load provider registry."""
        if self._provider_registry is None:
            from ..providers.registry import ProviderRegistry
            self._provider_registry = ProviderRegistry()
        return self._provider_registry

    @property
    def provider(self) -> BaseProvider:
        """Get the configured provider instance."""
        # If provider was passed as instance, return it directly
        if self._provider is not None:
            return self._provider

        # Otherwise, get from registry
        from ..providers.registry import ProviderRegistry
        registry = ProviderRegistry()

        # Handle both ProviderConfig objects and provider name strings
        if isinstance(self.config.provider, str):
            provider_name = self.config.provider
        else:
            provider_name = self.config.provider.provider_type.value

        self._provider = registry.get(
            provider_name,
            api_key=self.config.provider.get_api_key() if hasattr(self.config.provider, 'get_api_key') else None,
        )
        return self._provider

    @property
    def dataset_manager(self) -> DatasetManager:
        """Lazy-load dataset manager."""
        if self._dataset_manager is None:
            from ..datasets.manager import DatasetManager
            self._dataset_manager = DatasetManager()
        return self._dataset_manager

    @property
    def prompt_engine(self) -> PromptEngine:
        """Lazy-load prompt engine."""
        if self._prompt_engine is None:
            from ..prompts.engine import PromptEngine
            self._prompt_engine = PromptEngine()
        return self._prompt_engine

    @property
    def training_orchestrator(self) -> TrainingOrchestrator:
        """Lazy-load training orchestrator."""
        if self._training_orchestrator is None:
            from ..training.orchestrator import TrainingOrchestrator
            self._training_orchestrator = TrainingOrchestrator(
                provider=self.provider,
                callbacks=self.config.callbacks,
            )
        return self._training_orchestrator

    @property
    def evaluation_suite(self) -> EvaluationSuite:
        """Lazy-load evaluation suite."""
        if self._evaluation_suite is None:
            from ..evaluation.suite import EvaluationSuite
            self._evaluation_suite = EvaluationSuite()
        return self._evaluation_suite

    def upload_dataset(
        self,
        path: str,
        format: str = "auto",
        **kwargs: Any,
    ) -> Dataset:
        """Upload and validate a dataset.

        Args:
            path: Path to the dataset file.
            format: Format of the file (jsonl, csv, parquet, auto).
            **kwargs: Additional arguments for DatasetManager.

        Returns:
            Validated Dataset object.

        Raises:
            DatasetError: If dataset validation fails.
        """
        logger.info("uploading_dataset", path=path, format=format)

        dataset = self.dataset_manager.upload(path, format=format, **kwargs)
        validation_result = self.dataset_manager.validate(dataset)

        if not validation_result.is_valid:
            error_messages = [
                f"{err.field}: {err.message}"
                for err in validation_result.errors
            ]
            from .exceptions import ValidationError
            raise ValidationError(
                f"Dataset validation failed: {', '.join(error_messages)}",
                cause="invalid_dataset",
            )

        self._current_dataset = dataset
        logger.info(
            "dataset_uploaded",
            n_samples=len(dataset),
            format=format,
        )
        return dataset

    def create_template(
        self,
        template: str,
        name: str = "default",
        **kwargs: Any,
    ) -> PromptTemplate:
        """Create a prompt template.

        Args:
            template: Template string with {variable} placeholders.
            name: Name for this template.
            **kwargs: Additional arguments.

        Returns:
            PromptTemplate instance.
        """
        logger.info("creating_template", name=name)
        prompt_template = self.prompt_engine.create_template(template, name, **kwargs)
        self._current_template = prompt_template
        return prompt_template

    def train(
        self,
        *,
        dataset: Dataset | None = None,
        template: PromptTemplate | None = None,
        **kwargs: Any,
    ) -> TrainingJob:
        """Start a training job.

        Args:
            dataset: Dataset to train on. Uses current dataset if None.
            template: Prompt template to use. Uses current template if None.
            **kwargs: Override hyperparameters.

        Returns:
            TrainingJob instance with job_id for tracking.

        Raises:
            TrainingError: If training fails to start.
        """
        dataset = dataset or self._current_dataset
        template = template or self._current_template

        if dataset is None:
            from .exceptions import ValidationError
            raise ValidationError(
                "No dataset provided. Call upload_dataset() first.",
                field="dataset",
            )

        logger.info("starting_training", dataset_size=len(dataset))

        # Extract hyperparameters from kwargs (exclude non-hyperparams)
        from .config import HyperParams

        hyperparam_fields = {
            "learning_rate", "lora_rank", "lora_alpha", "lora_dropout",
            "batch_size", "epochs", "warmup_steps", "max_seq_length", "weight_decay"
        }
        hyperparams_kwargs = {k: v for k, v in kwargs.items() if k in hyperparam_fields}
        hyperparams = HyperParams(**hyperparams_kwargs) if hyperparams_kwargs else self.config.hyperparameters

        job = self.training_orchestrator.train(
            dataset=dataset,
            template=template,
            hyperparams=hyperparams,
            model=self.config.provider.model,
        )

        self._current_job = job
        logger.info("training_started", job_id=job.job_id)
        return job

    def get_status(self, job_id: str | None = None) -> Any:
        """Get the status of a training job.

        Args:
            job_id: Job ID to check. Uses current job if None.

        Returns:
            JobStatus object with current state.
        """
        job_id = job_id or (self._current_job.job_id if self._current_job else None)

        if not job_id:
            from .exceptions import ValidationError
            raise ValidationError("No job_id provided and no current job")

        return self.training_orchestrator.get_status(job_id)

    def cancel(self, job_id: str | None = None) -> None:
        """Cancel a training job.

        Args:
            job_id: Job ID to cancel. Uses current job if None.

        Raises:
            JobCancelledError: If job is already completed or failed.
        """
        job_id = job_id or (self._current_job.job_id if self._current_job else None)

        if not job_id:
            from .exceptions import ValidationError
            raise ValidationError("No job_id provided and no current job")

        logger.info("cancelling_job", job_id=job_id)
        self.training_orchestrator.cancel(job_id)

        if self._current_job and self._current_job.job_id == job_id:
            self._current_job = None

    def evaluate(
        self,
        predictions: list[Any],
        references: list[Any],
        metrics: list[str] | None = None,
    ) -> EvalResult:
        """Evaluate model predictions.

        Args:
            predictions: Model predictions.
            references: Ground truth values.
            metrics: List of metric names. Uses config defaults if None.

        Returns:
            EvalResult with scores for each metric.
        """
        if metrics:
            for metric in metrics:
                self.evaluation_suite.add_metric(metric)
        else:
            # Add default metrics from config
            from ..evaluation.metrics import RMSE, F1, MAE, Accuracy
            self.evaluation_suite.add_metric("rmse", RMSE())
            self.evaluation_suite.add_metric("f1", F1())
            self.evaluation_suite.add_metric("mae", MAE())
            self.evaluation_suite.add_metric("accuracy", Accuracy())

        logger.info("evaluating", n_samples=len(predictions))
        return self.evaluation_suite.evaluate(predictions, references)

    def deploy(self, **kwargs: Any) -> Any:
        """Deploy the trained model.

        Args:
            **kwargs: Deployment options.

        Returns:
            DeployedModel instance with prediction endpoint.
        """
        if not self._current_job:
            from .exceptions import ValidationError
            raise ValidationError("No training job found. Run train() first.")

        logger.info("deploying_model")
        self._deployed_model = self.provider.deploy(
            job_id=self._current_job.job_id,
            **kwargs,
        )
        return self._deployed_model

    @property
    def model(self) -> Any:
        """Get the deployed model for inference."""
        if self._deployed_model is None:
            from .exceptions import AutoMLChainError
            raise AutoMLChainError(
                "No deployed model. Run deploy() first or use provider directly.",
            )
        return self._deployed_model


__all__ = ["AutoMLPipeline"]
