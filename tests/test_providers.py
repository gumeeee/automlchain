"""Tests for providers module."""

import pytest
import time
from automlchain.providers import (
    MockProvider,
    ProviderRegistry,
    PricingProvider,
)


class TestMockProvider:
    """Tests for MockProvider."""

    def test_create_provider(self):
        """Test provider creation."""
        provider = MockProvider(api_key="test_key")
        assert provider.name == "mock"
        assert provider.api_key == "test_key"

    def test_train_returns_job(self):
        """Test that train returns a TrainingJob."""
        provider = MockProvider(api_key="test", simulate_duration=0.1)
        job = provider.train(
            model="meta/llama-3-8b",
            training_file="test.jsonl",
        )
        assert job.job_id is not None
        assert job.model == "meta/llama-3-8b"
        # Status can be pending or queued depending on timing

    def test_get_job_status(self):
        """Test getting job status."""
        provider = MockProvider(api_key="test", simulate_duration=0.1)
        job = provider.train(
            model="meta/llama-3-8b",
            training_file="test.jsonl",
        )
        status = provider.get_job_status(job.job_id)
        assert status.status in ["pending", "queued", "running", "completed"]


class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    def test_register_provider(self):
        """Test that replicate is registered by default."""
        providers = ProviderRegistry.list_providers()
        assert "replicate" in providers

    def test_get_replicate_provider(self):
        """Test getting replicate provider."""
        provider = ProviderRegistry.get("replicate", api_key="test")
        assert provider is not None


class TestPricingProvider:
    """Tests for PricingProvider."""

    def test_create_pricing_provider(self):
        """Test creating a pricing provider."""
        pricing = PricingProvider("replicate")
        assert pricing.provider == "replicate"

    def test_get_training_cost(self):
        """Test getting training cost for known model."""
        pricing = PricingProvider("replicate")
        cost = pricing.get_training_cost("meta/llama-3-8b")
        assert cost > 0
        assert isinstance(cost, float)

    def test_get_training_cost_unknown_model(self):
        """Test getting training cost for unknown model uses default."""
        pricing = PricingProvider("replicate")
        cost = pricing.get_training_cost("unknown/model-v99")
        assert cost > 0
        assert isinstance(cost, float)

    def test_get_inference_cost(self):
        """Test getting inference cost."""
        pricing = PricingProvider("replicate")
        cost = pricing.get_inference_cost("meta/llama-3-8b")
        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_training_cost(self):
        """Test training cost estimation."""
        pricing = PricingProvider("replicate")
        cost = pricing.estimate_training_cost(
            model="meta/llama-3-8b",
            epochs=3,
            batch_size=4,
            dataset_size_tokens=1_000_000,
        )
        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_inference_cost(self):
        """Test inference cost estimation."""
        pricing = PricingProvider("replicate")
        cost = pricing.estimate_inference_cost(
            model="meta/llama-3-8b",
            num_tokens=100,
        )
        assert cost >= 0
        assert isinstance(cost, float)

    def test_update_cost(self):
        """Test manually updating cost."""
        pricing = PricingProvider("replicate")
        pricing.update(
            model="custom/model",
            training_cost=1.0,
            inference_cost=2.0,
        )
        assert pricing.get_training_cost("custom/model") == 1.0


class TestLocalProvider:
    """Tests for LocalProvider."""

    def test_create_provider(self):
        """Test provider creation."""
        from automlchain.providers import LocalProvider

        provider = LocalProvider(model="TinyLlama/TinyLlama-1.1B")
        assert provider.name == "local"
        assert provider.model == "TinyLlama/TinyLlama-1.1B"

    def test_provider_not_requires_api_key(self):
        """Test that LocalProvider works without API key."""
        from automlchain.providers import LocalProvider

        provider = LocalProvider()
        assert provider.api_key is None or provider.api_key == "local"

    def test_train_without_dependencies(self):
        """Test train() handles missing dependencies."""
        from automlchain.providers import LocalProvider

        provider = LocalProvider()

        # Check if dependencies are available
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            dependencies_available = True
        except ImportError:
            dependencies_available = False

        if not dependencies_available:
            # Should raise ImportError without dependencies
            with pytest.raises(ImportError):
                provider.train(
                    model="TinyLlama/TinyLlama-1.1B",
                    training_file="nonexistent.jsonl",
                )
        else:
            # Should raise TrainingError for missing file
            with pytest.raises(Exception):  # Could be FileNotFoundError or TrainingError
                provider.train(
                    model="TinyLlama/TinyLlama-1.1B",
                    training_file="nonexistent.jsonl",
                )


class TestTogetherProvider:
    """Tests for TogetherProvider."""

    def test_create_provider_requires_api_key(self):
        """Test that TogetherProvider requires API key."""
        from automlchain.providers import ProviderRegistry

        with pytest.raises(ValueError, match="API key required"):
            ProviderRegistry.get("together")

    def test_together_registered(self):
        """Test that Together provider is registered."""
        providers = ProviderRegistry.list_providers()
        assert "together" in providers

    def test_together_optional_import(self):
        """Test that TogetherProvider handles missing package gracefully."""
        from automlchain.providers import ProviderRegistry

        # Try to get together - will fail without package
        # but won't crash the module
        assert ProviderRegistry.is_registered("together")
