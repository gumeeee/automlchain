"""Callbacks for training progress and status updates."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CallbackEvent:
    """Event data passed to callbacks.

    Attributes:
        event_type: Type of event (job_started, job_progress, job_completed, etc.).
        job_id: ID of the job.
        data: Event-specific data.
        timestamp: Event timestamp.
    """

    event_type: str
    job_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class BaseCallback(ABC):
    """Abstract base class for training callbacks.

    Implement this interface to receive training updates.

    Example:
        >>> class MyCallback(BaseCallback):
        ...     def on_job_progress(self, event: CallbackEvent):
        ...         print(f"Progress: {event.data['progress']}%")
    """

    @abstractmethod
    def on_job_started(self, event: CallbackEvent) -> None:
        """Called when a job starts."""
        ...

    @abstractmethod
    def on_job_progress(self, event: CallbackEvent) -> None:
        """Called periodically during job execution."""
        ...

    @abstractmethod
    def on_job_completed(self, event: CallbackEvent) -> None:
        """Called when a job completes successfully."""
        ...

    @abstractmethod
    def on_job_failed(self, event: CallbackEvent) -> None:
        """Called when a job fails."""
        ...

    @abstractmethod
    def on_job_cancelled(self, event: CallbackEvent) -> None:
        """Called when a job is cancelled."""
        ...


class ProgressCallback(BaseCallback):
    """Callback that prints progress to stdout.

    Example:
        >>> pipeline = AutoMLPipeline(callbacks=[ProgressCallback()])
    """

    def __init__(
        self,
        *,
        show_loss: bool = True,
        show_metrics: bool = True,
        refresh_rate: float = 1.0,
    ) -> None:
        """
        Args:
            show_loss: Show loss value in progress.
            show_metrics: Show additional metrics.
            refresh_rate: Minimum seconds between updates.
        """
        self.show_loss = show_loss
        self.show_metrics = show_metrics
        self.refresh_rate = refresh_rate
        self._last_update: float = 0

    def on_job_started(self, event: CallbackEvent) -> None:
        """Print job start message."""
        logger.info("job_started", job_id=event.job_id)
        print(f"[START] Training job {event.job_id} started")

    def on_job_progress(self, event: CallbackEvent) -> None:
        """Print progress bar."""
        # Rate limit updates
        now = time.time()
        if now - self._last_update < self.refresh_rate:
            return
        self._last_update = now

        data = event.data
        progress = data.get("progress", 0)
        epoch = data.get("epoch", 0)
        total_epochs = data.get("total_epochs", 0)
        loss = data.get("loss")

        # Build progress string
        bar_length = 30
        filled = int(bar_length * progress / 100)
        bar = "=" * filled + "-" * (bar_length - filled)

        parts = [f"\r[{bar}] {progress:.1f}%"]
        if total_epochs > 0:
            parts.append(f" Epoch {epoch}/{total_epochs}")

        if self.show_loss and loss is not None:
            parts.append(f" Loss: {loss:.4f}")

        if self.show_metrics:
            metrics = data.get("metrics", {})
            if metrics:
                metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
                parts.append(f" ({', '.join(metric_strs)})")

        print("".join(parts), end="", flush=True)

    def on_job_completed(self, event: CallbackEvent) -> None:
        """Print completion message."""
        print()  # New line after progress bar
        data = event.data
        duration = data.get("duration_seconds", 0)
        cost = data.get("cost")

        print(f"[DONE] Training job {event.job_id} completed")
        print(f"  Duration: {duration:.1f}s")

        if cost is not None:
            print(f"  Cost: ${cost:.4f}")

        if event.job_id:
            print(f"  Job ID: {event.job_id}")

    def on_job_failed(self, event: CallbackEvent) -> None:
        """Print error message."""
        print()  # New line after progress bar
        error = event.data.get("error", "Unknown error")
        print(f"[ERROR] Training job {event.job_id} failed")
        print(f"  Error: {error}")

    def on_job_cancelled(self, event: CallbackEvent) -> None:
        """Print cancellation message."""
        print()  # New line after progress bar
        print(f"[CANCELLED] Training job {event.job_id} was cancelled")


class LoggingCallback(BaseCallback):
    """Callback that logs events using structlog.

    Example:
        >>> pipeline = AutoMLPipeline(callbacks=[LoggingCallback()])
    """

    def on_job_started(self, event: CallbackEvent) -> None:
        logger.info("training_job_started", job_id=event.job_id)

    def on_job_progress(self, event: CallbackEvent) -> None:
        logger.debug(
            "training_progress",
            job_id=event.job_id,
            progress=event.data.get("progress"),
            epoch=event.data.get("epoch"),
            loss=event.data.get("loss"),
        )

    def on_job_completed(self, event: CallbackEvent) -> None:
        logger.info(
            "training_completed",
            job_id=event.job_id,
            duration=event.data.get("duration_seconds"),
            cost=event.data.get("cost"),
        )

    def on_job_failed(self, event: CallbackEvent) -> None:
        logger.error(
            "training_failed",
            job_id=event.job_id,
            error=event.data.get("error"),
        )

    def on_job_cancelled(self, event: CallbackEvent) -> None:
        logger.warning("training_cancelled", job_id=event.job_id)


class WebhookCallback(BaseCallback):
    """Callback that sends events to a webhook URL.

    Example:
        >>> callback = WebhookCallback("https://myapp.com/webhooks/training")
        >>> pipeline = AutoMLPipeline(callbacks=[callback])
    """

    def __init__(
        self,
        webhook_url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            webhook_url: URL to send webhook events.
            headers: Additional HTTP headers.
        """
        import httpx
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self._client = httpx.Client(timeout=30.0)

    def _send(self, payload: dict[str, Any]) -> None:
        """Send webhook payload."""
        try:
            self._client.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
            )
        except Exception as e:
            logger.warning("webhook_send_failed", error=str(e))

    def on_job_started(self, event: CallbackEvent) -> None:
        self._send({
            "event": "job_started",
            "job_id": event.job_id,
            "timestamp": event.timestamp,
        })

    def on_job_progress(self, event: CallbackEvent) -> None:
        self._send({
            "event": "job_progress",
            "job_id": event.job_id,
            "timestamp": event.timestamp,
            **event.data,
        })

    def on_job_completed(self, event: CallbackEvent) -> None:
        self._send({
            "event": "job_completed",
            "job_id": event.job_id,
            "timestamp": event.timestamp,
            **event.data,
        })

    def on_job_failed(self, event: CallbackEvent) -> None:
        self._send({
            "event": "job_failed",
            "job_id": event.job_id,
            "timestamp": event.timestamp,
            **event.data,
        })

    def on_job_cancelled(self, event: CallbackEvent) -> None:
        self._send({
            "event": "job_cancelled",
            "job_id": event.job_id,
            "timestamp": event.timestamp,
        })


class CallbackManager:
    """Manages multiple callbacks.

    Dispatches events to all registered callbacks.
    """

    def __init__(self, callbacks: list[BaseCallback] | None = None) -> None:
        """
        Args:
            callbacks: List of callbacks to manage.
        """
        self.callbacks: list[BaseCallback] = callbacks or []

    def add(self, callback: BaseCallback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def remove(self, callback: BaseCallback) -> None:
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def on_job_started(self, event: CallbackEvent) -> None:
        """Notify all callbacks of job start."""
        for callback in self.callbacks:
            try:
                callback.on_job_started(event)
            except Exception as e:
                logger.warning("callback_error", error=str(e))

    def on_job_progress(self, event: CallbackEvent) -> None:
        """Notify all callbacks of progress."""
        for callback in self.callbacks:
            try:
                callback.on_job_progress(event)
            except Exception as e:
                logger.warning("callback_error", error=str(e))

    def on_job_completed(self, event: CallbackEvent) -> None:
        """Notify all callbacks of completion."""
        for callback in self.callbacks:
            try:
                callback.on_job_completed(event)
            except Exception as e:
                logger.warning("callback_error", error=str(e))

    def on_job_failed(self, event: CallbackEvent) -> None:
        """Notify all callbacks of failure."""
        for callback in self.callbacks:
            try:
                callback.on_job_failed(event)
            except Exception as e:
                logger.warning("callback_error", error=str(e))

    def on_job_cancelled(self, event: CallbackEvent) -> None:
        """Notify all callbacks of cancellation."""
        for callback in self.callbacks:
            try:
                callback.on_job_cancelled(event)
            except Exception as e:
                logger.warning("callback_error", error=str(e))
