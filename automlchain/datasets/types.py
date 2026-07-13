"""Dataset management for AutoMLChain.

Supports JSONL, CSV, and Parquet formats with auto-detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator



@dataclass
class Dataset:
    """Represents a dataset for fine-tuning.

    Attributes:
        data: List of data samples as dictionaries.
        format: Original file format.
        path: Source file path.
        metadata: Additional metadata about the dataset.
    """

    data: list[dict[str, Any]]
    format: str
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over dataset samples."""
        return iter(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single sample by index."""
        return self.data[index]

    def get_field(self, field_name: str) -> list[Any]:
        """Get all values for a specific field.

        Args:
            field_name: Name of the field to extract.

        Returns:
            List of values for the field.
        """
        return [sample.get(field_name) for sample in self.data]

    def filter(
        self,
        predicate: Any,
    ) -> Dataset:
        """Filter samples based on a predicate function.

        Args:
            predicate: Function that takes a sample and returns bool.

        Returns:
            New Dataset with filtered samples.
        """
        filtered_data = [s for s in self.data if predicate(s)]
        return Dataset(
            data=filtered_data,
            format=self.format,
            path=self.path,
            metadata={**self.metadata, "filtered": True},
        )

    def sample(self, n: int, seed: int | None = None) -> Dataset:
        """Get a random sample of the dataset.

        Args:
            n: Number of samples to return.
            seed: Random seed for reproducibility.

        Returns:
            New Dataset with sampled data.
        """
        import random
        if seed is not None:
            random.seed(seed)

        sampled = random.sample(self.data, min(n, len(self.data)))
        return Dataset(
            data=sampled,
            format=self.format,
            path=self.path,
            metadata={**self.metadata, "sampled": True, "sample_size": n},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset to dictionary representation."""
        return {
            "data": self.data,
            "format": self.format,
            "path": self.path,
            "metadata": self.metadata,
            "n_samples": len(self.data),
        }


@dataclass
class DatasetStats:
    """Statistics about a dataset."""

    n_rows: int
    n_cols: int
    fields: list[str]
    field_types: dict[str, str]
    missing_values: dict[str, int]
    encoding: str | None = None
    file_size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "fields": self.fields,
            "field_types": self.field_types,
            "missing_values": self.missing_values,
            "encoding": self.encoding,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass
class ValidationError:
    """A single validation error."""

    field: str
    message: str
    row: int | None = None
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "message": self.message,
            "row": self.row,
            "value": self.value,
        }


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: DatasetStats | None = None

    def add_error(
        self,
        field: str,
        message: str,
        row: int | None = None,
        value: Any = None,
    ) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(field, message, row, value))
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "stats": self.stats.to_dict() if self.stats else None,
        }


@dataclass
class DatasetVersion:
    """Represents a versioned snapshot of a dataset."""

    version_id: str
    dataset: Dataset
    created_at: str | None = None
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "checksum": self.checksum,
            "dataset": self.dataset.to_dict(),
        }
