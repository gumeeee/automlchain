"""Configuration dataclasses for AutoMLChain.

Uses frozen dataclasses for immutability and type safety.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ProviderType(str, Enum):
    """Supported fine-tuning providers."""

    REPLICATE = "replicate"
    TOGETHER = "together"  # Phase 2
    ANYSCALE = "anyscale"  # Phase 2
    LOCAL = "local"  # Local GPU training


class FineTuningMethod(str, Enum):
    """Supported fine-tuning methods."""

    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"


class TaskType(str, Enum):
    """Task types for model evaluation."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TEXT_GENERATION = "text_generation"


@dataclass(frozen=True)
class HyperParams:
    """Hyperparameters for fine-tuning.

    All values have sane defaults that work for most use cases.
    Adjust based on dataset size and model.

    Attributes:
        learning_rate: Learning rate for optimizer. Range: 1e-6 to 1e-2.
        lora_rank: LoRA rank dimension. Higher = more capacity, more params.
        lora_alpha: LoRA alpha parameter. Usually 2x lora_rank.
        lora_dropout: Dropout probability for LoRA layers.
        batch_size: Number of samples per batch. Adjust based on GPU memory.
        epochs: Number of training passes over dataset.
        warmup_steps: Steps for learning rate warmup.
        max_seq_length: Maximum sequence length in tokens.
        weight_decay: L2 regularization strength.
    """

    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 4
    epochs: int = 3
    warmup_steps: int = 100
    max_seq_length: int = 2048
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        """Validate hyperparameter ranges."""
        if not 1e-6 <= self.learning_rate <= 1e-2:
            raise ValueError(
                f"learning_rate must be between 1e-6 and 1e-2, got {self.learning_rate}"
            )
        if not 8 <= self.lora_rank <= 128:
            raise ValueError(
                f"lora_rank must be between 8 and 128, got {self.lora_rank}"
            )
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError(
                f"lora_dropout must be between 0 and 1, got {self.lora_dropout}"
            )
        if not 1 <= self.batch_size <= 64:
            raise ValueError(
                f"batch_size must be between 1 and 64, got {self.batch_size}"
            )
        if not 1 <= self.epochs <= 10:
            raise ValueError(
                f"epochs must be between 1 and 10, got {self.epochs}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "learning_rate": self.learning_rate,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "warmup_steps": self.warmup_steps,
            "max_seq_length": self.max_seq_length,
            "weight_decay": self.weight_decay,
        }


@dataclass(frozen=True)
class Budget:
    """Budget constraints for training runs.

    Attributes:
        max_cost: Maximum cost in USD. None = unlimited.
        max_duration_seconds: Maximum training duration. None = unlimited.
        max_trials: Maximum number of optimization trials (Phase 2).
    """

    max_cost: float | None = None
    max_duration_seconds: int | None = None
    max_trials: int | None = None

    def is_within_budget(self, cost: float | None = None, duration: int | None = None) -> bool:
        """Check if cost/duration are within budget."""
        if self.max_cost is not None and cost is not None and cost > self.max_cost:
            return False
        if self.max_duration_seconds is not None and duration is not None:
            if duration > self.max_duration_seconds:
                return False
        return True


@dataclass
class ProviderConfig:
    """Configuration for a specific provider.

    Attributes:
        provider_type: The type of provider (replicate, together, etc.).
        api_key: API key for the provider. If None, reads from environment.
        model: Default model identifier.
        webhook_url: Optional webhook for status updates.
        extra: Additional provider-specific configuration.
    """

    provider_type: ProviderType
    api_key: str | None = None
    model: str = "meta/llama-3-8b-instruct"
    webhook_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def get_api_key(self) -> str:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key

        env_vars = {
            ProviderType.REPLICATE: "REPLICATE_API_TOKEN",
            ProviderType.TOGETHER: "TOGETHER_API_KEY",
            ProviderType.ANYSCALE: "ANYSCALE_API_KEY",
        }

        env_var = env_vars.get(self.provider_type)
        if not env_var:
            raise ValueError(f"No environment variable defined for {self.provider_type}")

        key = os.environ.get(env_var)
        if not key:
            from .exceptions import APIKeyError
            raise APIKeyError(
                self.provider_type.value,
                env_var=env_var,
            )
        return key


@dataclass
class PipelineConfig:
    """Main configuration for AutoMLChain pipeline.

    Attributes:
        provider: Provider configuration.
        hyperparameters: Training hyperparameters.
        budget: Budget constraints.
        callbacks: List of callback instances for progress tracking.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_format: Log format (text, json).
    """

    provider: ProviderConfig
    hyperparameters: HyperParams = field(default_factory=HyperParams)
    budget: Budget = field(default_factory=Budget)
    callbacks: list[Any] = field(default_factory=list)
    log_level: str = "INFO"
    log_format: str = "text"

    def get_log_config(self) -> dict[str, Any]:
        """Get logging configuration for structlog."""
        return {
            "log_level": self.log_level.lower(),
            "log_format": self.log_format,
        }


@dataclass
class DatasetConfig:
    """Configuration for dataset processing.

    Attributes:
        input_field: Field name for input text.
        output_field: Field name for expected output.
        encoding: File encoding. Auto-detect if None.
        max_samples: Maximum samples to use. None = all.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for reproducibility.
    """

    input_field: str = "input"
    output_field: str = "output"
    encoding: str | None = None
    max_samples: int | None = None
    shuffle: bool = False
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.

    Attributes:
        metrics: List of metric names to compute.
        test_size: Fraction of data to use for testing.
        random_seed: Random seed for train/test split.
    """

    metrics: list[str] = field(default_factory=lambda: ["rmse", "f1"])
    test_size: float = 0.2
    random_seed: int = 42


@dataclass
class ConfigManager:
    """Manages configuration from multiple sources.

    Priority (low to high):
    1. Defaults in code
    2. Config file (~/.automlchain/config.yaml)
    3. Environment variables
    4. Runtime arguments
    """

    config_dir: Path = field(
        default_factory=lambda: Path.home() / ".automlchain"
    )
    config_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.config_file = self.config_dir / "config.yaml"

    def load(self) -> PipelineConfig:
        """Load configuration from all sources."""
        # Start with defaults
        config = self._load_defaults()

        # Override with config file if exists
        if self.config_file.exists():
            config = self._merge_config_file(config)

        # Override with environment variables
        config = self._merge_env_vars(config)

        return config

    def _load_defaults(self) -> PipelineConfig:
        """Load default configuration."""
        return PipelineConfig(
            provider=ProviderConfig(provider_type=ProviderType.REPLICATE),
        )

    def _merge_config_file(self, config: PipelineConfig) -> PipelineConfig:
        """Merge configuration from config file."""
        # TODO: Implement YAML loading
        return config

    def _merge_env_vars(self, config: PipelineConfig) -> PipelineConfig:
        """Merge configuration from environment variables."""
        if provider_type := os.environ.get("AUTOMLCHAIN_PROVIDER"):
            try:
                config.provider.provider_type = ProviderType(provider_type.lower())
            except ValueError:
                pass  # Keep default

        if model := os.environ.get("AUTOMLCHAIN_MODEL"):
            config.provider.model = model

        if log_level := os.environ.get("AUTOMLCHAIN_LOG_LEVEL"):
            config.log_level = log_level.upper()

        return config

    def save(self, config: PipelineConfig) -> None:
        """Save configuration to config file."""
        # TODO: Implement YAML saving
        pass
