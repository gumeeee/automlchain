"""Evaluation module for AutoMLChain.

Handles model evaluation and metrics computation.
"""

from .metrics import (
    BaseMetric,
    RMSE,
    MAE,
    Accuracy,
    F1,
    Precision,
    Recall,
    BUILTIN_METRICS,
    get_metric,
)
from .suite import (
    MetricResult,
    EvalResult,
    ComparisonResult,
    EvaluationSuite,
    evaluate_predictions,
)

__all__ = [
    # Metrics
    "BaseMetric",
    "RMSE",
    "MAE",
    "Accuracy",
    "F1",
    "Precision",
    "Recall",
    "BUILTIN_METRICS",
    "get_metric",
    # Suite
    "MetricResult",
    "EvalResult",
    "ComparisonResult",
    "EvaluationSuite",
    "evaluate_predictions",
]
