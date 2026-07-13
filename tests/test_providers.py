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
