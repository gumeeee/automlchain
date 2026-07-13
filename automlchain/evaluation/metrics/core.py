"""Built-in evaluation metrics for AutoMLChain."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics.

    All metrics must implement the compute method.

    Example:
        >>> class MyMetric(BaseMetric):
        ...     def compute(self, y_true, y_pred):
        ...         return some_score
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self.__doc__ or self.name

    @abstractmethod
    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> float:
        """Compute the metric.

        Args:
            predictions: Model predictions.
            references: Ground truth values.

        Returns:
            Metric value (higher is better for accuracy, lower for error).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RMSE(BaseMetric):
    """Root Mean Squared Error metric.

    Measures the square root of the average squared differences
    between predictions and references.

    Best for: Regression tasks, rating predictions

    Example:
        >>> metric = RMSE()
        >>> score = metric.compute([3.0, 4.0], [2.5, 4.5])
    """

    @property
    def name(self) -> str:
        return "rmse"

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> float:
        """Compute RMSE."""
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions, "
                f"{len(references)} references"
            )

        if not predictions:
            return 0.0

        squared_errors = []

        for pred, ref in zip(predictions, references):
            try:
                pred_val = float(pred)
                ref_val = float(ref)
                squared_errors.append((pred_val - ref_val) ** 2)
            except (ValueError, TypeError):
                # Skip non-numeric values
                continue

        if not squared_errors:
            raise ValueError("No valid numeric pairs found")

        mse = sum(squared_errors) / len(squared_errors)
        return float(mse ** 0.5)


class MAE(BaseMetric):
    """Mean Absolute Error metric.

    Measures the average absolute difference between
    predictions and references.

    Best for: Regression tasks where outliers should be penalized less

    Example:
        >>> metric = MAE()
        >>> score = metric.compute([3.0, 4.0], [2.5, 4.5])
    """

    @property
    def name(self) -> str:
        return "mae"

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> float:
        """Compute MAE."""
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions, "
                f"{len(references)} references"
            )

        if not predictions:
            return 0.0

        absolute_errors = []

        for pred, ref in zip(predictions, references):
            try:
                pred_val = float(pred)
                ref_val = float(ref)
                absolute_errors.append(abs(pred_val - ref_val))
            except (ValueError, TypeError):
                continue

        if not absolute_errors:
            raise ValueError("No valid numeric pairs found")

        return float(sum(absolute_errors) / len(absolute_errors))


class Accuracy(BaseMetric):
    """Accuracy metric.

    Measures the proportion of correct predictions.

    Best for: Classification tasks with balanced classes

    Example:
        >>> metric = Accuracy()
        >>> score = metric.compute(["cat", "dog", "cat"], ["cat", "dog", "dog"])
    """

    @property
    def name(self) -> str:
        return "accuracy"

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> float:
        """Compute accuracy."""
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions, "
                f"{len(references)} references"
            )

        if not predictions:
            return 0.0

        correct = sum(
            1 for pred, ref in zip(predictions, references)
            if self._normalize(pred) == self._normalize(ref)
        )

        return float(correct / len(predictions))

    def _normalize(self, value: Any) -> str:
        """Normalize value for comparison."""
        if isinstance(value, str):
            return value.lower().strip()
        return str(value)


class F1(BaseMetric):
    """F1 Score metric.

    Harmonic mean of precision and recall.

    Best for: Classification tasks with imbalanced classes

    Example:
        >>> metric = F1()
        >>> score = metric.compute(["cat", "dog", "cat"], ["cat", "dog", "dog"])
    """

    def __init__(self, average: str = "macro") -> None:
        """
        Args:
            average: Averaging method ("macro", "micro", "weighted").
        """
        self.average = average

    @property
    def name(self) -> str:
        return f"f1_{self.average}"

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> float:
        """Compute F1 score."""
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions, "
                f"{len(references)} references"
            )

        if not predictions:
            return 0.0

        # Get unique classes
        all_classes = set(predictions) | set(references)

        if len(all_classes) == 2:
            # Binary classification
            return self._binary_f1(predictions, references)
        else:
            # Multi-class
            return self._multiclass_f1(predictions, references, all_classes)

    def _binary_f1(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> float:
        """Compute binary F1 score."""
        # Find positive class (prefer reference values)
        positive_class = None
        for ref in references:
            if ref is not None:
                positive_class = self._normalize(ref)
                break

        if positive_class is None:
            return 0.0

        # Calculate TP, FP, FN
        tp = fp = fn = 0

        for pred, ref in zip(predictions, references):
            pred_norm = self._normalize(pred)
            ref_norm = self._normalize(ref)

            if pred_norm == ref_norm == positive_class:
                tp += 1
            elif pred_norm == positive_class and ref_norm != positive_class:
                fp += 1
            elif pred_norm != positive_class and ref_norm == positive_class:
                fn += 1

        return self._compute_f1_from_components(tp, fp, fn)

    def _multiclass_f1(
        self,
        predictions: list[Any],
        references: list[Any],
        classes: set[Any],
    ) -> float:
        """Compute multi-class F1 score."""
        class_metrics: dict[str, dict[str, int]] = {}

        for cls in classes:
            cls_norm = self._normalize(cls)
            class_metrics[cls_norm] = {"tp": 0, "fp": 0, "fn": 0}

        for pred, ref in zip(predictions, references):
            pred_norm = self._normalize(pred)
            ref_norm = self._normalize(ref)

            if pred_norm == ref_norm:
                class_metrics[ref_norm]["tp"] += 1
            else:
                class_metrics[pred_norm]["fp"] += 1
                class_metrics[ref_norm]["fn"] += 1

        if self.average == "micro":
            # Micro: aggregate TP, FP, FN globally
            total_tp = sum(m["tp"] for m in class_metrics.values())
            total_fp = sum(m["fp"] for m in class_metrics.values())
            total_fn = sum(m["fn"] for m in class_metrics.values())
            return self._compute_f1_from_components(total_tp, total_fp, total_fn)

        elif self.average == "weighted":
            # Weighted: weighted average by support
            f1_scores = []
            total_support = 0

            for cls, metrics in class_metrics.items():
                support = metrics["tp"] + metrics["fn"]
                f1 = self._compute_f1_from_components(
                    metrics["tp"], metrics["fp"], metrics["fn"]
                )
                f1_scores.append((f1, support))
                total_support += support

            if total_support == 0:
                return 0.0

            weighted_sum = sum(f1 * support for f1, support in f1_scores)
            return weighted_sum / total_support

        else:
            # Macro: unweighted average
            f1_scores = []

            for cls, metrics in class_metrics.items():
                f1 = self._compute_f1_from_components(
                    metrics["tp"], metrics["fp"], metrics["fn"]
                )
                f1_scores.append(f1)

            if not f1_scores:
                return 0.0

            return sum(f1_scores) / len(f1_scores)

    def _compute_f1_from_components(
        self,
        tp: int,
        fp: int,
        fn: int,
    ) -> float:
        """Compute F1 from TP, FP, FN."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def _normalize(self, value: Any) -> str:
        """Normalize value for comparison."""
        if isinstance(value, str):
            return value.lower().strip()
        return str(value)


class Precision(BaseMetric):
    """Precision metric.

    Measures the proportion of positive predictions that are correct.
    """

    @property
    def name(self) -> str:
        return "precision"

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> float:
        """Compute precision."""
        if len(predictions) != len(references):
            raise ValueError("Length mismatch")

        if not predictions:
            return 0.0

        all_classes = set(predictions) | set(references)
        if len(all_classes) == 2:
            # Binary
            positive = None
            for r in references:
                if r is not None:
                    positive = self._normalize(r)
                    break
            if positive is None:
                return 0.0
            tp = fp = 0
            for pred, ref in zip(predictions, references):
                if self._normalize(pred) == positive:
                    if self._normalize(ref) == positive:
                        tp += 1
                    else:
                        fp += 1
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        else:
            # Multi-class macro
            scores = []
            for cls in all_classes:
                cls_norm = self._normalize(cls)
                tp = sum(1 for p, r in zip(predictions, references)
                        if self._normalize(p) == cls_norm and self._normalize(r) == cls_norm)
                fp = sum(1 for p, r in zip(predictions, references)
                        if self._normalize(p) == cls_norm and self._normalize(r) != cls_norm)
                if tp + fp > 0:
                    scores.append(tp / (tp + fp))
            return sum(scores) / len(scores) if scores else 0.0

    def _normalize(self, value: Any) -> str:
        if isinstance(value, str):
            return value.lower().strip()
        return str(value)


class Recall(BaseMetric):
    """Recall metric.

    Measures the proportion of actual positives that were identified.
    """

    @property
    def name(self) -> str:
        return "recall"

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
    ) -> float:
        """Compute recall."""
        if len(predictions) != len(references):
            raise ValueError("Length mismatch")

        if not predictions:
            return 0.0

        all_classes = set(predictions) | set(references)
        if len(all_classes) == 2:
            # Binary
            positive = None
            for r in references:
                if r is not None:
                    positive = self._normalize(r)
                    break
            if positive is None:
                return 0.0
            tp = fn = 0
            for pred, ref in zip(predictions, references):
                if self._normalize(ref) == positive:
                    if self._normalize(pred) == positive:
                        tp += 1
                    else:
                        fn += 1
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            scores = []
            for cls in all_classes:
                cls_norm = self._normalize(cls)
                tp = sum(1 for p, r in zip(predictions, references)
                        if self._normalize(p) == cls_norm and self._normalize(r) == cls_norm)
                fn = sum(1 for p, r in zip(predictions, references)
                        if self._normalize(p) != cls_norm and self._normalize(r) == cls_norm)
                if tp + fn > 0:
                    scores.append(tp / (tp + fn))
            return sum(scores) / len(scores) if scores else 0.0

    def _normalize(self, value: Any) -> str:
        if isinstance(value, str):
            return value.lower().strip()
        return str(value)


# Registry of built-in metrics
BUILTIN_METRICS: dict[str, type[BaseMetric]] = {
    "rmse": RMSE,
    "mae": MAE,
    "accuracy": Accuracy,
    "f1": F1,
    "f1_macro": lambda: F1(average="macro"),
    "f1_micro": lambda: F1(average="micro"),
    "f1_weighted": lambda: F1(average="weighted"),
    "precision": Precision,
    "recall": Recall,
}


def get_metric(name: str) -> BaseMetric:
    """Get a metric instance by name.

    Args:
        name: Metric name.

    Returns:
        Metric instance.

    Raises:
        ValueError: If metric not found.
    """
    if name not in BUILTIN_METRICS:
        raise ValueError(
            f"Unknown metric: {name}. Available: {list(BUILTIN_METRICS.keys())}"
        )

    metric_class = BUILTIN_METRICS[name]
    return metric_class() if callable(metric_class) else metric_class
