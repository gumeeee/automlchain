"""Pricing module for AutoMLChain.

Provides dynamic cost estimation for training and inference.
Implements hybrid approach: API fetch with local fallback.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Default costs (fallback when API unavailable)
# Source: Provider pricing pages (update periodically)
DEFAULT_COSTS: dict[str, dict[str, float]] = {
    "replicate": {
        # Training costs per 1M tokens
        "meta/llama-3-8b": 0.20,
        "meta/llama-3-70b": 0.90,
        "mistralai/mistral-7b": 0.20,
        "mistralai/mixtral-8x7b": 0.50,
        # Default for unknown models
        "_default": 0.25,
    },
    "together": {
        # Training costs per 1M tokens
        "meta-llama/Llama-3-8b": 0.20,
        "meta-llama/Llama-3-70b": 0.90,
        "mistralai/Mistral-7B-v0.1": 0.20,
        "_default": 0.25,
    },
    "anyscale": {
        # Training costs per 1M tokens
        "meta-llama/Llama-3-8b": 0.18,
        "meta-llama/Llama-3-70b": 0.85,
        "mistralai/Mistral-7B-v0.1": 0.18,
        "_default": 0.22,
    },
}

# Inference costs per 1M output tokens
DEFAULT_INFERENCE_COSTS: dict[str, dict[str, float]] = {
    "replicate": {
        "meta/llama-3-8b": 0.20,
        "meta/llama-3-70b": 2.75,
        "mistralai/mistral-7b": 0.20,
        "_default": 0.50,
    },
    "together": {
        "meta-llama/Llama-3-8b": 0.20,
        "meta-llama/Llama-3-70b": 2.75,
        "mistralai/Mistral-7B-v0.1": 0.20,
        "_default": 0.50,
    },
    "anyscale": {
        "meta-llama/Llama-3-8b": 0.18,
        "meta-llama/Llama-3-70b": 2.50,
        "mistralai/Mistral-7B-v0.1": 0.18,
        "_default": 0.45,
    },
}


@dataclass
class CostEntry:
    """A single cost entry for a model.

    Attributes:
        model: Model identifier.
        training_cost_per_1m: Training cost per 1M tokens.
        inference_cost_per_1m: Inference cost per 1M output tokens.
        source: Where the cost was obtained ("api", "local", "default").
        fetched_at: Timestamp when cost was fetched.
    """

    model: str
    training_cost_per_1m: float
    inference_cost_per_1m: float
    source: str = "default"
    fetched_at: float = field(default_factory=time.time)


@dataclass
class PricingCache:
    """Cache for model pricing data.

    Attributes:
        training_costs: Training costs per model.
        inference_costs: Inference costs per model.
        provider: Provider name.
        last_updated: Timestamp of last update.
    """

    training_costs: dict[str, float] = field(default_factory=dict)
    inference_costs: dict[str, float] = field(default_factory=dict)
    provider: str = ""
    last_updated: float = field(default_factory=time.time)

    def get_training_cost(self, model: str) -> float:
        """Get training cost for model."""
        return self.training_costs.get(model, self.training_costs.get("_default", 0.25))

    def get_inference_cost(self, model: str) -> float:
        """Get inference cost for model."""
        return self.inference_costs.get(model, self.inference_costs.get("_default", 0.50))


class PricingProvider:
    """Provider for model pricing information.

    Implements hybrid approach:
    1. Try to fetch from API
    2. Fall back to local cache
    3. Use default values if all else fails

    Example:
        >>> pricing = PricingProvider("replicate")
        >>> cost = pricing.get_training_cost("meta/llama-3-8b")
        >>> print(f"${cost:.4f} per 1M tokens")
    """

    def __init__(
        self,
        provider: str,
        *,
        cache_ttl: float = 3600.0,  # 1 hour
        cache_dir: Path | None = None,
    ) -> None:
        """
        Args:
            provider: Provider name (e.g., "replicate", "together").
            cache_ttl: Cache time-to-live in seconds.
            cache_dir: Directory for persistent cache. Defaults to ~/.automlchain/.
        """
        self.provider = provider.lower()
        self.cache_ttl = cache_ttl
        self._cache: PricingCache | None = None

        # Setup cache directory
        if cache_dir:
            self._cache_dir = cache_dir
        else:
            self._cache_dir = Path.home() / ".automlchain" / "pricing"
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Load from file if available
        self._load_cache()

    def _get_cache_file(self) -> Path:
        """Get cache file path for this provider."""
        return self._cache_dir / f"{self.provider}_pricing.json"

    def _load_cache(self) -> None:
        """Load cache from file if valid."""
        cache_file = self._get_cache_file()
        if not cache_file.exists():
            # Initialize with defaults
            self._cache = PricingCache(provider=self.provider)
            self._update_with_defaults()
            return

        try:
            with open(cache_file) as f:
                data = json.load(f)

            # Check if cache is still valid
            last_updated = data.get("last_updated", 0)
            if time.time() - last_updated < self.cache_ttl:
                self._cache = PricingCache(
                    training_costs=data.get("training_costs", {}),
                    inference_costs=data.get("inference_costs", {}),
                    provider=self.provider,
                    last_updated=last_updated,
                )
                logger.info("pricing_cache_loaded", provider=self.provider)
            else:
                # Cache expired, reload defaults
                self._cache = PricingCache(provider=self.provider)
                self._update_with_defaults()
                logger.info("pricing_cache_expired", provider=self.provider)

        except (json.JSONDecodeError, IOError) as e:
            logger.warning("pricing_cache_load_failed", error=str(e))
            self._cache = PricingCache(provider=self.provider)
            self._update_with_defaults()

    def _save_cache(self) -> None:
        """Save cache to file."""
        if self._cache is None:
            return

        cache_file = self._get_cache_file()
        try:
            self._cache.last_updated = time.time()
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "provider": self.provider,
                        "training_costs": self._cache.training_costs,
                        "inference_costs": self._cache.inference_costs,
                        "last_updated": self._cache.last_updated,
                    },
                    f,
                    indent=2,
                )
            logger.debug("pricing_cache_saved", provider=self.provider)
        except IOError as e:
            logger.warning("pricing_cache_save_failed", error=str(e))

    def _update_with_defaults(self) -> None:
        """Update cache with default values for provider."""
        if self._cache is None:
            self._cache = PricingCache(provider=self.provider)

        defaults = DEFAULT_COSTS.get(self.provider, DEFAULT_COSTS["replicate"])
        inference_defaults = DEFAULT_INFERENCE_COSTS.get(
            self.provider, DEFAULT_INFERENCE_COSTS["replicate"]
        )

        for model, cost in defaults.items():
            if model != "_default":
                self._cache.training_costs[model] = cost

        for model, cost in inference_defaults.items():
            if model != "_default":
                self._cache.inference_costs[model] = cost

        # Set default
        self._cache.training_costs["_default"] = defaults.get("_default", 0.25)
        self._cache.inference_costs["_default"] = inference_defaults.get("_default", 0.50)

    async def fetch_from_api(self, api_key: str) -> bool:
        """Fetch latest pricing from provider API.

        Args:
            api_key: API key for authentication.

        Returns:
            True if fetch succeeded, False otherwise.
        """
        if self.provider == "replicate":
            return await self._fetch_replicate_pricing(api_key)
        elif self.provider == "together":
            return await self._fetch_together_pricing(api_key)
        else:
            logger.warning("pricing_api_not_supported", provider=self.provider)
            return False

    async def _fetch_replicate_pricing(self, api_key: str) -> bool:
        """Fetch pricing from Replicate API."""
        import httpx

        try:
            async with httpx.AsyncClient():
                # Replicate doesn't have a public pricing API endpoint
                # We would need to scrape or maintain manually
                # For now, just use defaults
                logger.debug("replicate_pricing_api_unavailable")
                return False
        except Exception as e:
            logger.warning("replicate_pricing_fetch_failed", error=str(e))
            return False

    async def _fetch_together_pricing(self, api_key: str) -> bool:
        """Fetch pricing from Together AI API."""
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.together.xyz/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )

                if response.status_code == 200:
                    # TODO: Extract pricing from model info when API exposes it
                    # models = response.json()
                    # for model in models:
                    #     model_id = model.get("id", "")
                    return False
                return False

        except Exception as e:
            logger.warning("together_pricing_fetch_failed", error=str(e))
            return False

    def get_training_cost(self, model: str) -> float:
        """Get training cost per 1M tokens.

        Args:
            model: Model identifier.

        Returns:
            Cost in USD per 1M tokens.
        """
        if self._cache is None:
            self._update_with_defaults()

        return self._cache.get_training_cost(model)

    def get_inference_cost(self, model: str) -> float:
        """Get inference cost per 1M output tokens.

        Args:
            model: Model identifier.

        Returns:
            Cost in USD per 1M tokens.
        """
        if self._cache is None:
            self._update_with_defaults()

        return self._cache.get_inference_cost(model)

    def estimate_training_cost(
        self,
        model: str,
        epochs: int,
        batch_size: int,
        dataset_size_tokens: int = 1_000_000,
    ) -> float:
        """Estimate total training cost.

        Args:
            model: Model identifier.
            epochs: Number of training epochs.
            batch_size: Batch size.
            dataset_size_tokens: Approximate dataset size in tokens.

        Returns:
            Estimated cost in USD.
        """
        cost_per_token = self.get_training_cost(model)

        # Simplified estimation
        # Real cost depends on GPU time, which varies by model size
        effective_tokens = dataset_size_tokens * epochs * (8 / max(batch_size, 1))
        return cost_per_token * (effective_tokens / 1_000_000)

    def estimate_inference_cost(
        self,
        model: str,
        num_tokens: int,
    ) -> float:
        """Estimate inference cost.

        Args:
            model: Model identifier.
            num_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        cost_per_token = self.get_inference_cost(model)
        return cost_per_token * (num_tokens / 1000)  # per 1K tokens

    def refresh(self) -> bool:
        """Refresh pricing from cache file.

        Returns:
            True if cache was refreshed.
        """
        old_cache = self._cache
        self._load_cache()

        if self._cache is not None and self._cache is not old_cache:
            return True
        return False

    def update(self, model: str, training_cost: float, inference_cost: float) -> None:
        """Manually update cost for a model.

        Args:
            model: Model identifier.
            training_cost: Training cost per 1M tokens.
            inference_cost: Inference cost per 1M tokens.
        """
        if self._cache is None:
            self._cache = PricingCache(provider=self.provider)

        self._cache.training_costs[model] = training_cost
        self._cache.inference_costs[model] = inference_cost
        self._save_cache()

        logger.info(
            "pricing_updated",
            provider=self.provider,
            model=model,
            training_cost=training_cost,
            inference_cost=inference_cost,
        )
