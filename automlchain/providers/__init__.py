"""Providers module for AutoMLChain.

Abstraction layer for different fine-tuning providers.
"""

from .base import BaseProvider, TrainingJob, JobStatus, DeployedModel
from .registry import ProviderRegistry
from .replicate import ReplicateProvider, ReplicateInferenceClient
from .mock import MockProvider
from .pricing import PricingProvider, PricingCache

# Optional providers (may not be installed)
try:
    from .together import TogetherProvider
except ImportError:
    TogetherProvider = None  # type: ignore

try:
    from .local import LocalProvider
except ImportError:
    LocalProvider = None  # type: ignore

__all__ = [
    # Base classes
    "BaseProvider",
    "TrainingJob",
    "JobStatus",
    "DeployedModel",
    # Registry
    "ProviderRegistry",
    # Providers
    "ReplicateProvider",
    "ReplicateInferenceClient",
    "MockProvider",
    "TogetherProvider",  # May be None if not installed
    "LocalProvider",  # May be None if not installed
    # Pricing
    "PricingProvider",
    "PricingCache",
]
