"""Evaluation suite for AutoMLChain.

Evaluates model predictions against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from .metrics import BaseMetric, RMSE, F1, MAE, Accuracy, get_metric

logger = structlog.get_logger(__name__)


@dataclass
class MetricResult:
    """Result of computing a single metric.

    Attributes:
        name: Metric name.
        value: Computed metric value.
        higher_is_better: Whether higher values are better.
        details: Additional details about the computation.
    """

    name: str
    value: float
    higher_is_better: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "higher_is_better": self.higher_is_better,
            "details": self.details,
        }


@dataclass
class EvalResult:
    """Result of evaluation with multiple metrics.

    Attributes:
        metrics: Results for each metric.
        predictions: Input predictions.
        references: Input references.
        summary: Summary statistics.
    """

    metrics: list[MetricResult] = field(default_factory=list)
    predictions: list[Any] = field(default_factory=list)
    references: list[Any] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def get_metric(self, name: str) -> MetricResult | None:
        """Get a metric result by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def best_metric(self) -> MetricResult | None:
        """Get the best scoring metric (for accuracy-like metrics)."""
        candidates = [m for m in self.metrics if m.higher_is_better]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.value)

    def worst_metric(self) -> MetricResult | None:
        """Get the worst scoring metric (for accuracy-like metrics)."""
        candidates = [m for m in self.metrics if m.higher_is_better]
        if not candidates:
            return None
        return min(candidates, key=lambda m: m.value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "summary": self.summary,
            "n_samples": len(self.predictions),
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = ["Evaluation Results:", "-" * 40]
        for metric in self.metrics:
            indicator = "↑" if metric.higher_is_better else "↓"
            lines.append(f"  {metric.name}: {metric.value:.4f} {indicator}")
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Result of comparing multiple models.

    Attributes:
        model_scores: Dict mapping model names to EvalResult.
        rankings: Dict mapping metric names to ranked models.
    """

    model_scores: dict[str, EvalResult] = field(default_factory=dict)
    rankings: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    _metric_preference: dict[str, bool] = field(default_factory=dict)  # higher_is_better per metric

    def add_model(self, model_name: str, result: EvalResult) -> None:
        """Add a model's evaluation result."""
        self.model_scores[model_name] = result

        # Update rankings
        for metric_result in result.metrics:
            metric_name = metric_result.name
            if metric_name not in self.rankings:
                self.rankings[metric_name] = []
                self._metric_preference[metric_name] = metric_result.higher_is_better

            self.rankings[metric_name].append((model_name, metric_result.value))

        # Sort rankings based on whether higher is better for each metric
        for metric_name in self.rankings:
            higher_is_better = self._metric_preference.get(metric_name, True)
            self.rankings[metric_name] = sorted(
                self.rankings[metric_name],
                key=lambda x: x[1],
                reverse=higher_is_better,  # Higher first if higher_is_better, lower first otherwise
            )

    def get_ranking(self, metric_name: str) -> list[tuple[str, float]]:
        """Get ranking for a specific metric.

        Returns:
            List of (model_name, score) tuples, sorted best first.
        """
        return self.rankings.get(metric_name, [])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_scores": {
                name: result.to_dict()
                for name, result in self.model_scores.items()
            },
            "rankings": self.rankings,
        }


class EvaluationSuite:
    """Suite for evaluating model predictions.

    Supports multiple metrics and model comparison.

    Example:
        >>> suite = EvaluationSuite()
        >>> suite.add_metric("rmse", RMSE())
        >>> suite.add_metric("f1", F1())
        >>> result = suite.evaluate(predictions, references)
        >>> print(result)
    """

    def __init__(
        self,
        metrics: list[tuple[str, BaseMetric]] | None = None,
    ) -> None:
        """
        Args:
            metrics: Initial list of (name, metric) tuples.
        """
        self._metrics: dict[str, BaseMetric] = {}
        self._higher_is_better: dict[str, bool] = {}

        # Add default metrics
        self._add_default_metrics()

        # Add provided metrics
        if metrics:
            for name, metric in metrics:
                self.add_metric(name, metric)

    def _add_default_metrics(self) -> None:
        """Add default metrics for MVP."""
        self.add_metric("rmse", RMSE())
        self.add_metric("mae", MAE())
        self.add_metric("accuracy", Accuracy())
        self.add_metric("f1", F1())

    def add_metric(
        self,
        name: str,
        metric: BaseMetric | str,
        *,
        higher_is_better: bool | None = None,
    ) -> None:
        """Add a metric to the suite.

        Args:
            name: Name for this metric.
            metric: Metric instance or metric name string.
            higher_is_better: Whether higher values are better.
                Auto-detected for built-in metrics if None.
        """
        if isinstance(metric, str):
            metric = get_metric(metric)

        self._metrics[name] = metric

        # Auto-detect higher_is_better
        if higher_is_better is None:
            # Error metrics: lower is better
            # Score metrics: higher is better
            if name.lower() in ("rmse", "mae", "error"):
                self._higher_is_better[name] = False
            else:
                self._higher_is_better[name] = True
        else:
            self._higher_is_better[name] = higher_is_better

        logger.debug("metric_added", name=name)

    def remove_metric(self, name: str) -> None:
        """Remove a metric from the suite."""
        self._metrics.pop(name, None)
        self._higher_is_better.pop(name, None)

    def list_metrics(self) -> list[str]:
        """List all configured metrics."""
        return list(self._metrics.keys())

    def evaluate(
        self,
        predictions: list[Any],
        references: list[Any],
        *,
        return_predictions: bool = False,
    ) -> EvalResult:
        """Evaluate predictions against references.

        Args:
            predictions: Model predictions.
            references: Ground truth values.
            return_predictions: Include predictions/references in result.

        Returns:
            EvalResult with computed metrics.
        """
        logger.info(
            "evaluating",
            n_samples=len(predictions),
            n_metrics=len(self._metrics),
        )

        metric_results: list[MetricResult] = []
        errors: list[str] = []

        for name, metric in self._metrics.items():
            try:
                value = metric.compute(predictions, references)
                metric_results.append(
                    MetricResult(
                        name=name,
                        value=value,
                        higher_is_better=self._higher_is_better.get(name, True),
                    )
                )
            except Exception as e:
                error_msg = f"{name}: {e}"
                errors.append(error_msg)
                logger.warning("metric_error", name=name, error=str(e))

        # Build summary
        summary: dict[str, Any] = {
            "n_samples": len(predictions),
            "n_metrics": len(metric_results),
            "n_errors": len(errors),
        }

        if errors:
            summary["errors"] = errors

        result = EvalResult(
            metrics=metric_results,
            predictions=predictions if return_predictions else [],
            references=references if return_predictions else [],
            summary=summary,
        )

        logger.info(
            "evaluation_complete",
            n_metrics=len(metric_results),
        )

        return result

    def compare(
        self,
        models: dict[str, tuple[list[Any], list[Any]]],
    ) -> ComparisonResult:
        """Compare multiple models.

        Args:
            models: Dict mapping model names to (predictions, references) tuples.

        Returns:
            ComparisonResult with rankings.
        """
        logger.info("comparing_models", n_models=len(models))

        comparison = ComparisonResult()

        for model_name, (predictions, references) in models.items():
            result = self.evaluate(predictions, references)
            comparison.add_model(model_name, result)

        return comparison


def evaluate_predictions(
    predictions: list[Any],
    references: list[Any],
    metrics: list[str] | None = None,
) -> EvalResult:
    """Quick evaluation function.

    Args:
        predictions: Model predictions.
        references: Ground truth.
        metrics: List of metric names. Uses defaults if None.

    Returns:
        EvalResult with computed metrics.
    """
    suite = EvaluationSuite()

    if metrics:
        for metric in metrics:
            suite.add_metric(metric, metric)

    return suite.evaluate(predictions, references)
