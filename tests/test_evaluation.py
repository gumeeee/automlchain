"""Tests for evaluation module."""

import pytest
from automlchain.evaluation import (
    EvaluationSuite,
    RMSE,
    MAE,
    Accuracy,
    F1,
    get_metric,
)


class TestMetrics:
    """Tests for individual metrics."""

    def test_rmse(self):
        """Test RMSE metric."""
        metric = RMSE()
        predictions = [3.0, 4.0, 5.0]
        references = [3.5, 4.5, 5.5]
        score = metric.compute(predictions, references)
        assert score >= 0
        assert isinstance(score, float)

    def test_rmse_perfect(self):
        """Test RMSE with perfect predictions."""
        metric = RMSE()
        predictions = [1.0, 2.0, 3.0]
        references = [1.0, 2.0, 3.0]
        score = metric.compute(predictions, references)
        assert score == 0.0

    def test_rmse_length_mismatch(self):
        """Test RMSE raises on length mismatch."""
        metric = RMSE()
        with pytest.raises(ValueError, match="Length mismatch"):
            metric.compute([1.0, 2.0], [1.0])

    def test_mae(self):
        """Test MAE metric."""
        metric = MAE()
        predictions = [3.0, 4.0, 5.0]
        references = [3.5, 4.5, 5.5]
        score = metric.compute(predictions, references)
        assert score >= 0

    def test_mae_perfect(self):
        """Test MAE with perfect predictions."""
        metric = MAE()
        predictions = [1.0, 2.0, 3.0]
        references = [1.0, 2.0, 3.0]
        score = metric.compute(predictions, references)
        assert score == 0.0

    def test_accuracy(self):
        """Test Accuracy metric."""
        metric = Accuracy()
        predictions = ["cat", "dog", "cat"]
        references = ["cat", "dog", "bird"]
        score = metric.compute(predictions, references)
        assert 0 <= score <= 1
        assert score == 2/3

    def test_accuracy_perfect(self):
        """Test Accuracy with perfect predictions."""
        metric = Accuracy()
        predictions = ["cat", "dog", "bird"]
        references = ["cat", "dog", "bird"]
        score = metric.compute(predictions, references)
        assert score == 1.0

    def test_accuracy_case_insensitive(self):
        """Test Accuracy is case insensitive."""
        metric = Accuracy()
        predictions = ["CAT", "Dog"]
        references = ["cat", "DOG"]
        score = metric.compute(predictions, references)
        assert score == 1.0

    def test_f1_binary(self):
        """Test F1 for binary classification."""
        metric = F1()
        predictions = ["positive", "negative", "positive", "negative"]
        references = ["positive", "positive", "positive", "negative"]
        score = metric.compute(predictions, references)
        assert 0 <= score <= 1

    def test_f1_macro(self):
        """Test F1 with macro averaging."""
        metric = F1(average="macro")
        predictions = ["a", "b", "c", "a"]
        references = ["a", "b", "c", "a"]
        score = metric.compute(predictions, references)
        assert score == 1.0

    def test_f1_with_none_values(self):
        """Test F1 handles None values correctly (regression fix)."""
        metric = F1()
        predictions = ["cat", "dog", None, "cat"]
        references = ["cat", "dog", "bird", "cat"]
        score = metric.compute(predictions, references)
        assert 0 <= score <= 1

    def test_f1_all_none_references(self):
        """Test F1 when all references are None."""
        metric = F1()
        predictions = ["cat", "dog", "bird"]
        references = [None, None, None]
        score = metric.compute(predictions, references)
        assert score == 0.0


class TestGetMetric:
    """Tests for get_metric factory function."""

    def test_get_rmse(self):
        """Test getting RMSE metric."""
        metric = get_metric("rmse")
        assert isinstance(metric, RMSE)

    def test_get_mae(self):
        """Test getting MAE metric."""
        metric = get_metric("mae")
        assert isinstance(metric, MAE)

    def test_get_accuracy(self):
        """Test getting Accuracy metric."""
        metric = get_metric("accuracy")
        assert isinstance(metric, Accuracy)

    def test_get_f1(self):
        """Test getting F1 metric."""
        metric = get_metric("f1")
        assert isinstance(metric, F1)

    def test_get_unknown_metric(self):
        """Test getting unknown metric raises error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("unknown_metric")


class TestEvaluationSuite:
    """Tests for EvaluationSuite class."""

    def test_default_metrics(self):
        """Test suite has default metrics."""
        suite = EvaluationSuite()
        metrics = suite.list_metrics()
        assert "rmse" in metrics
        assert "f1" in metrics

    def test_add_metric(self):
        """Test adding a metric."""
        suite = EvaluationSuite()
        suite.add_metric("custom", RMSE())
        assert "custom" in suite.list_metrics()

    def test_remove_metric(self):
        """Test removing a metric."""
        suite = EvaluationSuite()
        suite.remove_metric("rmse")
        metrics = suite.list_metrics()
        assert "rmse" not in metrics

    def test_evaluate(self):
        """Test basic evaluation."""
        suite = EvaluationSuite()
        predictions = ["cat", "dog", "cat"]
        references = ["cat", "dog", "bird"]
        result = suite.evaluate(predictions, references)

        assert result.get_metric("accuracy") is not None
        assert result.get_metric("f1") is not None
        assert result.summary["n_samples"] == 3

    def test_evaluate_with_numeric(self):
        """Test evaluation with numeric predictions."""
        suite = EvaluationSuite()
        predictions = [3.0, 4.0, 5.0]
        references = [3.5, 4.5, 5.5]
        result = suite.evaluate(predictions, references)

        assert result.get_metric("rmse") is not None
        assert result.get_metric("mae") is not None

    def test_best_metric(self):
        """Test getting best metric."""
        suite = EvaluationSuite()
        predictions = ["cat", "dog", "cat"]
        references = ["cat", "dog", "bird"]
        result = suite.evaluate(predictions, references)

        best = result.best_metric()
        assert best is not None

    def test_worst_metric(self):
        """Test getting worst metric."""
        suite = EvaluationSuite()
        predictions = ["cat", "dog", "cat"]
        references = ["cat", "dog", "bird"]
        result = suite.evaluate(predictions, references)

        worst = result.worst_metric()
        assert worst is not None

    def test_result_to_dict(self):
        """Test result serialization."""
        suite = EvaluationSuite()
        predictions = ["cat", "dog"]
        references = ["cat", "dog"]
        result = suite.evaluate(predictions, references)

        d = result.to_dict()
        assert "metrics" in d
        assert "summary" in d


class TestComparisonResult:
    """Tests for model comparison."""

    def test_compare_models(self):
        """Test comparing multiple models."""
        suite = EvaluationSuite()
        models = {
            "model_a": (["cat", "dog", "bird"], ["cat", "dog", "bird"]),
            "model_b": (["cat", "cat", "cat"], ["cat", "dog", "bird"]),
        }

        comparison = suite.compare(models)

        assert "model_a" in comparison.model_scores
        assert "model_b" in comparison.model_scores

    def test_ranking(self):
        """Test model ranking."""
        suite = EvaluationSuite()
        models = {
            "model_a": (["cat", "dog", "bird"], ["cat", "dog", "bird"]),
            "model_b": (["bird", "bird", "bird"], ["cat", "dog", "bird"]),
        }

        comparison = suite.compare(models)
        ranking = comparison.get_ranking("accuracy")

        assert len(ranking) == 2
        # First should be better
        assert ranking[0][0] == "model_a"

    def test_ranking_for_error_metrics(self):
        """Test ranking for error metrics (lower is better)."""
        suite = EvaluationSuite()
        models = {
            "model_a": ([3.0, 3.0, 3.0], [3.0, 3.0, 3.0]),  # RMSE = 0
            "model_b": ([5.0, 5.0, 5.0], [3.0, 3.0, 3.0]),  # RMSE > 0
        }

        comparison = suite.compare(models)
        ranking = comparison.get_ranking("rmse")

        # For RMSE (lower is better), model_a should be first
        assert ranking[0][0] == "model_a"
        assert ranking[0][1] < ranking[1][1]
