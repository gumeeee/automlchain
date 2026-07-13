"""Pytest configuration and fixtures for AutoMLChain tests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Set test environment variables
os.environ.setdefault("REPLICATE_API_TOKEN", "test_replicate_token")
os.environ.setdefault("TOGETHER_API_KEY", "test_together_key")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_jsonl_file(temp_dir: Path) -> Path:
    """Create a sample JSONL file for testing."""
    file_path = temp_dir / "sample.jsonl"
    data = [
        {"input": "Great product", "output": "positive", "score": 5},
        {"input": "Terrible service", "output": "negative", "score": 1},
        {"input": "It was okay", "output": "neutral", "score": 3},
        {"input": "Amazing quality", "output": "positive", "score": 5},
        {"input": "Not worth it", "output": "negative", "score": 2},
    ]
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return file_path


@pytest.fixture
def sample_csv_file(temp_dir: Path) -> Path:
    """Create a sample CSV file for testing."""
    file_path = temp_dir / "sample.csv"
    with open(file_path, "w") as f:
        f.write("input,output,score\n")
        f.write("Great product,positive,5\n")
        f.write("Terrible service,negative,1\n")
        f.write("It was okay,neutral,3\n")
    return file_path


@pytest.fixture
def sample_json_file(temp_dir: Path) -> Path:
    """Create a sample JSON file for testing."""
    file_path = temp_dir / "sample.json"
    data = [
        {"input": "Great product", "output": "positive", "score": 5},
        {"input": "Terrible service", "output": "negative", "score": 1},
        {"input": "It was okay", "output": "neutral", "score": 3},
    ]
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def mock_provider():
    """Create a MockProvider instance for testing."""
    from automlchain.providers import MockProvider

    return MockProvider(
        api_key="test_key",
        simulate_duration=0.1,
        failure_rate=0.0,
    )


@pytest.fixture
def sample_dataset():
    """Create a sample Dataset for testing."""
    from automlchain.datasets import Dataset

    return Dataset(
        data=[
            {"input": "Great product", "output": "positive", "score": 5},
            {"input": "Terrible service", "output": "negative", "score": 1},
            {"input": "It was okay", "output": "neutral", "score": 3},
        ],
        format="jsonl",
    )


@pytest.fixture
def sample_predictions() -> list[str]:
    """Sample predictions for evaluation tests."""
    return ["positive", "negative", "neutral", "positive", "negative"]


@pytest.fixture
def sample_references() -> list[str]:
    """Sample references for evaluation tests."""
    return ["positive", "negative", "positive", "positive", "negative"]


@pytest.fixture
def sample_numeric_predictions() -> list[float]:
    """Sample numeric predictions for regression tests."""
    return [4.5, 1.2, 3.0, 5.0, 2.8]


@pytest.fixture
def sample_numeric_references() -> list[float]:
    """Sample numeric references for regression tests."""
    return [5.0, 1.0, 3.5, 4.5, 3.0]
