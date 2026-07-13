"""Dataset manager for AutoMLChain.

Handles dataset upload, validation, transformation, and versioning.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import structlog

from .types import Dataset, DatasetStats, ValidationResult, DatasetVersion
from .validators import DatasetValidator
from .formats import get_parser, detect_format
from ..core.exceptions import DatasetError

logger = structlog.get_logger(__name__)


class DatasetManager:
    """Manages datasets for fine-tuning.

    Handles upload, validation, format conversion, and versioning.

    Example:
        >>> manager = DatasetManager()
        >>> dataset = manager.upload("reviews.jsonl")
        >>> result = manager.validate(dataset)
        >>> if result.is_valid:
        ...     stats = manager.stats(dataset)
        ...     print(f"Loaded {stats.n_rows} samples")
    """

    def __init__(
        self,
        *,
        default_required_fields: list[str] | None = None,
        default_encoding: str | None = None,
    ) -> None:
        """
        Args:
            default_required_fields: Default required fields for validation.
            default_encoding: Default file encoding. Auto-detect if None.
        """
        self.default_required_fields = default_required_fields or ["input", "output"]
        self.default_encoding = default_encoding
        self._validator = DatasetValidator(
            required_fields=self.default_required_fields,
        )

    def upload(
        self,
        path: str,
        format: str = "auto",
        **kwargs: Any,
    ) -> Dataset:
        """Upload and parse a dataset from file.

        Args:
            path: Path to the dataset file.
            format: Format of the file (jsonl, csv, parquet, auto).
            **kwargs: Additional arguments for parsing.

        Returns:
            Dataset object with parsed data.

        Raises:
            DatasetError: If file cannot be read or parsed.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise DatasetError(
                f"Dataset file not found: {path}",
                path=str(path),
            )

        if not file_path.is_file():
            raise DatasetError(
                f"Path is not a file: {path}",
                path=str(path),
            )

        logger.info("loading_dataset", path=path)

        try:
            # Auto-detect format if needed
            if format == "auto":
                format = detect_format(file_path)
                logger.info("detected_format", format=format)

            # Get appropriate parser
            parser = get_parser(format)

            # Parse file
            data = parser.parse(file_path, encoding=self.default_encoding)

            logger.info(
                "dataset_loaded",
                n_samples=len(data),
                format=format,
            )

            return Dataset(
                data=data,
                format=format,
                path=str(file_path),
                metadata={
                    "original_path": str(file_path.absolute()),
                },
            )

        except Exception as e:
            raise DatasetError(
                f"Failed to parse dataset: {e}",
                path=str(path),
                format=format,
                cause=str(e),
            ) from e

    def validate(self, dataset: Dataset) -> ValidationResult:
        """Validate a dataset.

        Args:
            dataset: Dataset to validate.

        Returns:
            ValidationResult with any errors or warnings.
        """
        logger.info("validating_dataset", n_samples=len(dataset))

        result = self._validator.validate(dataset.data)

        # Add stats to result
        result.stats = self.stats(dataset)

        if result.is_valid:
            logger.info("dataset_valid", n_samples=len(dataset))
        else:
            logger.warning(
                "dataset_invalid",
                n_errors=len(result.errors),
                n_warnings=len(result.warnings),
            )

        return result

    def stats(self, dataset: Dataset) -> DatasetStats:
        """Get statistics about a dataset.

        Args:
            dataset: Dataset to analyze.

        Returns:
            DatasetStats with information about the dataset.
        """
        if not dataset.data:
            return DatasetStats(
                n_rows=0,
                n_cols=0,
                fields=[],
                field_types={},
                missing_values={},
            )

        # Collect field information
        fields: set[str] = set()
        field_types: dict[str, set[str]] = {}
        missing_values: dict[str, int] = {}

        for sample in dataset.data:
            for field_name, value in sample.items():
                fields.add(field_name)

                if field_name not in field_types:
                    field_types[field_name] = set()

                if value is None:
                    field_types[field_name].add("null")
                else:
                    field_types[field_name].add(type(value).__name__)

        # Count missing values
        n_samples = len(dataset)
        for field_name in fields:
            missing = sum(
                1 for sample in dataset.data
                if field_name not in sample or sample[field_name] is None
            )
            if missing > 0:
                missing_values[field_name] = missing

        # Convert type sets to strings
        field_type_strs: dict[str, str] = {}
        for field_name, types in field_types.items():
            type_list = sorted(t for t in types if t != "null")
            field_type_strs[field_name] = ", ".join(type_list) if type_list else "null"

        # Get file size if path available
        file_size = None
        if dataset.path:
            try:
                file_size = Path(dataset.path).stat().st_size
            except OSError:
                pass

        return DatasetStats(
            n_rows=n_samples,
            n_cols=len(fields),
            fields=sorted(fields),
            field_types=field_type_strs,
            missing_values=missing_values,
            file_size_bytes=file_size,
        )

    def convert(
        self,
        dataset: Dataset,
        to_format: str,
        output_path: str | None = None,
    ) -> Dataset:
        """Convert a dataset to a different format.

        Args:
            dataset: Dataset to convert.
            to_format: Target format (jsonl, csv, parquet).
            output_path: Optional output path. If None, updates in memory.

        Returns:
            Dataset in the new format.

        Raises:
            DatasetError: If conversion fails.
        """
        if to_format == dataset.format:
            return dataset

        logger.info(
            "converting_dataset",
            from_format=dataset.format,
            to_format=to_format,
        )

        try:
            # Get parser for target format
            parser = get_parser(to_format)

            # Determine output path
            if output_path is None and dataset.path:
                original_path = Path(dataset.path)
                output_path = str(
                    original_path.with_suffix(f".{to_format}")
                )

            output_path_obj = Path(output_path) if output_path else None

            if output_path_obj:
                # Write to file
                parser.write(dataset.data, output_path_obj)

                # Return new dataset with updated path
                return Dataset(
                    data=dataset.data,
                    format=to_format,
                    path=str(output_path_obj),
                    metadata={
                        **dataset.metadata,
                        "converted_from": dataset.format,
                    },
                )
            else:
                # Return new dataset without file
                return Dataset(
                    data=dataset.data,
                    format=to_format,
                    path=None,
                    metadata={
                        **dataset.metadata,
                        "converted_from": dataset.format,
                    },
                )

        except Exception as e:
            raise DatasetError(
                f"Failed to convert dataset to {to_format}: {e}",
                format=to_format,
                cause=str(e),
            ) from e

    def version(self, dataset: Dataset) -> DatasetVersion:
        """Create a versioned snapshot of a dataset.

        Args:
            dataset: Dataset to version.

        Returns:
            DatasetVersion with checksum.
        """
        import time

        # Generate checksum from data
        data_str = str(dataset.data)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()[:16]

        # Generate version ID
        version_id = f"v{int(time.time())}"

        return DatasetVersion(
            version_id=version_id,
            dataset=dataset,
            checksum=checksum,
        )

    def filter(
        self,
        dataset: Dataset,
        field: str,
        condition: str,
        value: Any,
    ) -> Dataset:
        """Filter dataset based on field values.

        Args:
            dataset: Dataset to filter.
            field: Field to filter on.
            condition: Comparison condition (eq, ne, gt, lt, ge, le, contains).
            value: Value to compare against.

        Returns:
            Filtered Dataset.
        """
        conditions = {
            "eq": lambda x: x == value,
            "ne": lambda x: x != value,
            "gt": lambda x: x > value,
            "lt": lambda x: x < value,
            "ge": lambda x: x >= value,
            "le": lambda x: x <= value,
            "contains": lambda x: value in str(x) if x else False,
        }

        if condition not in conditions:
            raise ValueError(f"Unknown condition: {condition}")

        predicate = conditions[condition]

        filtered_data = [
            sample
            for sample in dataset.data
            if field in sample and predicate(sample[field])
        ]

        return Dataset(
            data=filtered_data,
            format=dataset.format,
            path=dataset.path,
            metadata={
                **dataset.metadata,
                "filtered": True,
                "filter_field": field,
                "filter_condition": condition,
                "filter_value": value,
            },
        )
