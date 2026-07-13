"""Tests for training module."""

import pytest
import time
from unittest.mock import MagicMock
from automlchain.training import (
    ProgressCallback,
    LoggingCallback,
)
from automlchain.training.jobs import (
    JobState,
    JobStateTransition,
    TrainingJobInfo,
    TrainingResult,
    TrainingMetrics,
)


class TestJobState:
    """Tests for JobState enum."""

    def test_job_state_values(self):
        """Test JobState values."""
        assert JobState.PENDING.value == "pending"
        assert JobState.QUEUED.value == "queued"
        assert JobState.RUNNING.value == "running"
        assert JobState.COMPLETED.value == "completed"
        assert JobState.FAILED.value == "failed"

    def test_job_state_transitions(self):
        """Test valid state transitions."""
        assert JobStateTransition.can_transition(JobState.PENDING, JobState.QUEUED)
        assert JobStateTransition.can_transition(JobState.QUEUED, JobState.RUNNING)
        assert not JobStateTransition.can_transition(JobState.COMPLETED, JobState.RUNNING)


class TestTrainingJobInfo:
    """Tests for TrainingJobInfo dataclass."""

    def test_create_job_info(self):
        """Test creating job info."""
        info = TrainingJobInfo(
            job_id="test_job_123",
            state=JobState.RUNNING,
            model="meta/llama-3-8b",
            provider="mock",
        )
        assert info.job_id == "test_job_123"
        assert info.state == JobState.RUNNING
        assert info.model == "meta/llama-3-8b"

    def test_is_terminal(self):
        """Test terminal state detection."""
        info = TrainingJobInfo(
            job_id="test",
            state=JobState.RUNNING,
            model="test",
            provider="mock",
        )
        assert not info.is_terminal

        info.state = JobState.COMPLETED
        assert info.is_terminal

        info.state = JobState.FAILED
        assert info.is_terminal

    def test_progress(self):
        """Test progress calculation."""
        info = TrainingJobInfo(
            job_id="test",
            state=JobState.RUNNING,
            model="test",
            provider="mock",
            hyperparameters={"epochs": 3},
        )
        assert info.progress == 0.0

    def test_to_dict(self):
        """Test serialization."""
        info = TrainingJobInfo(
            job_id="test",
            state=JobState.COMPLETED,
            model="test",
            provider="mock",
        )
        d = info.to_dict()
        assert d["job_id"] == "test"
        assert d["state"] == "completed"


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating training metrics."""
        metrics = TrainingMetrics(
            loss=0.5,
            epoch=2,
            step=100,
        )
        assert metrics.loss == 0.5
        assert metrics.epoch == 2

    def test_to_dict(self):
        """Test serialization."""
        metrics = TrainingMetrics(loss=0.5, epoch=1)
        d = metrics.to_dict()
        assert d["loss"] == 0.5
        assert d["epoch"] == 1


class TestProgressCallback:
    """Tests for ProgressCallback."""

    def test_callback_creation(self):
        """Test creating a progress callback."""
        callback = ProgressCallback()
        assert callback is not None

    def test_on_job_started(self):
        """Test on_job_started event."""
        callback = ProgressCallback()
        event = MagicMock()
        event.data = {"job_id": "test", "model": "test"}
        callback.on_job_started(event)
        # Should not raise

    def test_on_job_completed(self):
        """Test on_job_completed event."""
        callback = ProgressCallback()
        event = MagicMock()
        event.data = {"job_id": "test", "result": {}}
        callback.on_job_completed(event)
        # Should not raise


class TestLoggingCallback:
    """Tests for LoggingCallback."""

    def test_callback_creation(self):
        """Test creating a logging callback."""
        callback = LoggingCallback()
        assert callback is not None

    def test_on_job_started(self):
        """Test on_job_started logs."""
        callback = LoggingCallback()
        event = MagicMock()
        event.data = {"job_id": "test_job", "model": "llama-3-8b"}
        callback.on_job_started(event)
        # Should log without raising


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_create_result(self):
        """Test creating a training result."""
        result = TrainingResult(
            job_id="test_job",
            status=JobState.COMPLETED,
        )
        assert result.job_id == "test_job"
        assert result.status == JobState.COMPLETED

    def test_to_dict(self):
        """Test serialization."""
        result = TrainingResult(
            job_id="test",
            status=JobState.COMPLETED,
        )
        d = result.to_dict()
        assert d["job_id"] == "test"
