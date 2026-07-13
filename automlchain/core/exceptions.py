"""Custom exceptions for AutoMLChain."""


class AutoMLChainError(Exception):
    """Base exception for all AutoMLChain errors."""

    def __init__(
        self,
        message: str,
        *,
        cause: str | None = None,
        suggestion: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.suggestion = suggestion
        self.context = kwargs

    def __str__(self) -> str:
        parts = [self.message]
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class ValidationError(AutoMLChainError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: object | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value


class DatasetError(AutoMLChainError):
    """Raised for dataset-related errors."""

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        format: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(message, **kwargs)
        self.path = path
        self.format = format


class TrainingError(AutoMLChainError):
    """Raised for training-related errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        job_id: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(message, **kwargs)
        self.provider = provider
        self.job_id = job_id


class ProviderError(AutoMLChainError):
    """Raised when a provider API call fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(message, **kwargs)
        self.provider = provider
        self.status_code = status_code


class ConfigurationError(AutoMLChainError):
    """Raised for configuration-related errors."""


class APIKeyError(ConfigurationError):
    """Raised when required API key is missing."""

    def __init__(self, provider: str, *, env_var: str | None = None) -> None:
        message = f"API key not found for provider '{provider}'"
        if env_var:
            message += f". Set the {env_var} environment variable."
        super().__init__(message)
        self.provider = provider
        self.env_var = env_var


class AutoMLChainTimeoutError(AutoMLChainError):
    """Raised when an operation times out.

    Note: Named AutoMLChainTimeoutError to avoid shadowing Python's built-in TimeoutError.
    """

    def __init__(self, message: str, *, seconds: float | None = None) -> None:
        super().__init__(message)
        self.seconds = seconds


# Alias for backward compatibility (deprecated, use AutoMLChainTimeoutError)
TimeoutError = AutoMLChainTimeoutError


class JobCancelledError(AutoMLChainError):
    """Raised when a job is cancelled."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Job {job_id} was cancelled")
        self.job_id = job_id
