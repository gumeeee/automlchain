"""Provider registry for AutoMLChain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from .base import BaseProvider

logger = structlog.get_logger(__name__)


class ProviderRegistry:
    """Registry for fine-tuning providers.

    Allows registration and retrieval of provider implementations.

    Example:
        >>> registry = ProviderRegistry()
        >>> provider = registry.get("replicate", api_key="...")
        >>> job = provider.train(model="meta/llama-3-8b", ...)
    """

    _providers: dict[str, type] = {}
    _instances: dict[str, "BaseProvider"] = {}

    @classmethod
    def register(
        cls,
        name: str,
        provider_class: type["BaseProvider"],
    ) -> type["BaseProvider"]:
        """Register a provider class.

        Args:
            name: Provider name (e.g., "replicate").
            provider_class: Provider class that extends BaseProvider.

        Returns:
            The provider class (for decorator usage).
        """
        cls._providers[name.lower()] = provider_class
        logger.info("provider_registered", name=name)
        return provider_class

    @classmethod
    def get(
        cls,
        name: str,
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> "BaseProvider":
        """Get a provider instance.

        Args:
            name: Provider name.
            api_key: API key for authentication.
            **kwargs: Additional provider configuration.

        Returns:
            Provider instance.

        Raises:
            ValueError: If provider is not registered or API key is missing.
        """
        name = name.lower()

        # Check cache first
        cache_key = f"{name}:{api_key}"
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # Get provider class
        if name not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Provider '{name}' not found. Available: {available}"
            )

        provider_class = cls._providers[name]

        # Require API key
        if api_key is None:
            raise ValueError(
                f"API key required for provider '{name}'. "
                f"Set the environment variable for this provider."
            )

        # Create instance
        instance = provider_class(api_key=api_key, **kwargs)
        cls._instances[cache_key] = instance

        logger.info("provider_initialized", name=name)
        return instance

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names.
        """
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name.

        Returns:
            True if registered.
        """
        return name.lower() in cls._providers

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the provider instance cache."""
        cls._instances.clear()


# Register built-in providers
from .replicate import ReplicateProvider  # noqa: E402

ProviderRegistry.register("replicate", ReplicateProvider)

# Register Together provider
try:
    from .together import TogetherProvider  # noqa: E402

    ProviderRegistry.register("together", TogetherProvider)
except ImportError:
    logger.warning("together_not_installed", message="Together AI provider not available. Install with: pip install together")

# Register Local provider (always available)
from .local import LocalProvider  # noqa: E402

ProviderRegistry.register("local", LocalProvider)
