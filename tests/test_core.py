"""Tests for core module - config and exceptions."""

import pytest
from automlchain.core.config import (
    HyperParams,
    Budget,
    ProviderType,
    TaskType,
    FineTuningMethod,
)
from automlchain.core.exceptions import (
    AutoMLChainError,
    AutoMLChainTimeoutError,
    ValidationError,
    DatasetError,
    TrainingError,
    ProviderError,
    ConfigurationError,
    APIKeyError,
    JobCancelledError,
)


class TestHyperParams:
    """Tests for HyperParams dataclass."""

    def test_default_values(self):
        """Test default hyperparameters."""
        params = HyperParams()
        assert params.learning_rate == 1e-4
        assert params.lora_rank == 16
        assert params.batch_size == 4
        assert params.epochs == 3

    def test_custom_values(self):
        """Test custom hyperparameters."""
        params = HyperParams(
            learning_rate=5e-5,
            lora_rank=32,
            batch_size=8,
            epochs=5,
        )
        assert params.learning_rate == 5e-5
        assert params.lora_rank == 32
        assert params.batch_size == 8
        assert params.epochs == 5

    def test_validation_learning_rate_too_high(self):
        """Test validation rejects learning rate > 1."""
        with pytest.raises(ValueError, match="learning_rate must be"):
            HyperParams(learning_rate=2.0)

    def test_validation_learning_rate_negative(self):
        """Test validation rejects negative learning rate."""
        with pytest.raises(ValueError, match="learning_rate must be"):
            HyperParams(learning_rate=-1e-5)

    def test_validation_lora_rank_too_low(self):
        """Test validation rejects lora_rank < 1."""
        with pytest.raises(ValueError, match="lora_rank must be"):
            HyperParams(lora_rank=0)

    def test_validation_epochs_negative(self):
        """Test validation rejects negative epochs."""
        with pytest.raises(ValueError, match="epochs must be"):
            HyperParams(epochs=-1)

    def test_validation_batch_size_invalid(self):
        """Test validation rejects invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be"):
            HyperParams(batch_size=0)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        params = HyperParams(learning_rate=1e-4, lora_rank=16)
        d = params.to_dict()
        assert d["learning_rate"] == 1e-4
        assert d["lora_rank"] == 16
        assert "learning_rate" in d
        assert "lora_rank" in d


class TestBudget:
    """Tests for Budget dataclass."""

    def test_default_values(self):
        """Test default budget is unlimited."""
        budget = Budget()
        assert budget.max_cost is None
        assert budget.max_duration_seconds is None
        assert budget.max_trials is None

    def test_custom_values(self):
        """Test custom budget constraints."""
        budget = Budget(
            max_cost=100.0,
            max_duration_seconds=3600,
            max_trials=10,
        )
        assert budget.max_cost == 100.0
        assert budget.max_duration_seconds == 3600
        assert budget.max_trials == 10

    def test_is_within_budget_cost(self):
        """Test cost budget checking."""
        budget = Budget(max_cost=50.0)
        assert budget.is_within_budget(cost=25.0)
        assert budget.is_within_budget(cost=50.0)
        assert not budget.is_within_budget(cost=75.0)

    def test_is_within_budget_duration(self):
        """Test duration budget checking."""
        budget = Budget(max_duration_seconds=3600)
        assert budget.is_within_budget(duration=1800)
        assert budget.is_within_budget(duration=3600)
        assert not budget.is_within_budget(duration=7200)

    def test_is_within_budget_unlimited(self):
        """Test unlimited budget always returns True."""
        budget = Budget()
        assert budget.is_within_budget(cost=1_000_000)
        assert budget.is_within_budget(duration=1_000_000)


class TestEnums:
    """Tests for enum types."""

    def test_provider_type_values(self):
        """Test ProviderType enum values."""
        assert ProviderType.REPLICATE.value == "replicate"
        assert ProviderType.TOGETHER.value == "together"
        assert ProviderType.ANYSCALE.value == "anyscale"

    def test_task_type_values(self):
        """Test TaskType enum values."""
        assert TaskType.CLASSIFICATION.value == "classification"
        assert TaskType.REGRESSION.value == "regression"

    def test_fine_tuning_method_values(self):
        """Test FineTuningMethod enum values."""
        assert FineTuningMethod.LORA.value == "lora"
        assert FineTuningMethod.QLORA.value == "qlora"
        assert FineTuningMethod.FULL.value == "full"


class TestAutoMLChainError:
    """Tests for AutoMLChainError base exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = AutoMLChainError("Test error")
        assert error.message == "Test error"
        assert str(error) == "Test error"

    def test_error_with_cause(self):
        """Test error with cause."""
        error = AutoMLChainError(
            "Operation failed",
            cause="Network timeout",
        )
        assert "Operation failed" in str(error)
        assert "Network timeout" in str(error)

    def test_error_with_suggestion(self):
        """Test error with suggestion."""
        error = AutoMLChainError(
            "API key missing",
            suggestion="Set REPLICATE_API_TOKEN environment variable",
        )
        assert "API key missing" in str(error)
        assert "REPLICATE_API_TOKEN" in str(error)

    def test_error_with_context(self):
        """Test error with extra context."""
        error = AutoMLChainError(
            "Training failed",
            job_id="job123",
            provider="replicate",
        )
        assert error.context["job_id"] == "job123"
        assert error.context["provider"] == "replicate"


class TestTimeoutError:
    """Tests for AutoMLChainTimeoutError."""

    def test_timeout_error(self):
        """Test timeout error creation."""
        error = AutoMLChainTimeoutError(
            "Job timed out after 60 seconds",
            seconds=60.0,
        )
        assert error.seconds == 60.0
        assert "60 seconds" in str(error)

    def test_timeout_error_alias(self):
        """Test backward compatibility alias."""
        from automlchain.core.exceptions import TimeoutError

        error = TimeoutError("Test timeout", seconds=30)
        assert error.seconds == 30
        assert isinstance(error, AutoMLChainTimeoutError)


class TestAPIKeyError:
    """Tests for APIKeyError."""

    def test_api_key_error(self):
        """Test API key error creation."""
        error = APIKeyError("replicate", env_var="REPLICATE_API_TOKEN")
        assert error.provider == "replicate"
        assert error.env_var == "REPLICATE_API_TOKEN"
        assert "replicate" in str(error)
        assert "REPLICATE_API_TOKEN" in str(error)


class TestJobCancelledError:
    """Tests for JobCancelledError."""

    def test_job_cancelled_error(self):
        """Test job cancelled error creation."""
        error = JobCancelledError("job_abc123")
        assert error.job_id == "job_abc123"
        assert "job_abc123" in str(error)


class TestValidationError:
    """Tests for ValidationError."""

    def test_validation_error(self):
        """Test validation error with field info."""
        error = ValidationError(
            "Invalid value",
            field="learning_rate",
            value=2.0,
        )
        assert error.field == "learning_rate"
        assert error.value == 2.0


class TestTrainingError:
    """Tests for TrainingError."""

    def test_training_error(self):
        """Test training error with context."""
        error = TrainingError(
            "Training failed",
            provider="replicate",
            job_id="job_123",
        )
        assert error.provider == "replicate"
        assert error.job_id == "job_123"
