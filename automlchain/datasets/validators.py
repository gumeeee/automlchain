"""Validators for dataset validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import ValidationResult


class BaseValidator(ABC):
    """Base class for dataset validators."""

    @abstractmethod
    def validate(self, data: list[dict[str, Any]]) -> ValidationResult:
        """Validate dataset and return result."""
        ...


class SchemaValidator(BaseValidator):
    """Validates dataset schema.

    Ensures required fields exist in all samples.
    """

    def __init__(
        self,
        required_fields: list[str] | None = None,
        recommended_fields: list[str] | None = None,
        field_types: dict[str, type] | None = None,
    ) -> None:
        """
        Args:
            required_fields: Fields that must exist in every sample.
            recommended_fields: Fields that should exist (warning if missing).
            field_types: Expected types for each field.
        """
        self.required_fields = required_fields or ["input", "output"]
        self.recommended_fields = recommended_fields or []
        self.field_types = field_types or {}

    def validate(self, data: list[dict[str, Any]]) -> ValidationResult:
        """Validate schema requirements."""
        result = ValidationResult(is_valid=True)

        if not data:
            result.add_error("dataset", "Dataset is empty")
            return result

        # Check required fields in first sample
        first_sample = data[0]
        all_fields = set(first_sample.keys())

        for field_name in self.required_fields:
            if field_name not in all_fields:
                result.add_error(
                    field=field_name,
                    message=f"Required field '{field_name}' not found in dataset",
                )

        # Check recommended fields
        for field_name in self.recommended_fields:
            if field_name not in all_fields:
                result.add_warning(
                    f"Recommended field '{field_name}' not found in dataset"
                )

        return result


class EncodingValidator(BaseValidator):
    """Validates and detects encoding issues."""

    def validate(self, data: list[dict[str, Any]]) -> ValidationResult:
        """Check for encoding issues in string fields."""
        result = ValidationResult(is_valid=True)

        # Check for common encoding issues
        for i, sample in enumerate(data[:100]):  # Check first 100 samples
            for field_name, value in sample.items():
                if isinstance(value, str):
                    # Check for replacement character
                    if "\ufffd" in value:
                        result.add_error(
                            field=field_name,
                            message="Contains replacement character (encoding issue)",
                            row=i,
                            value=value[:50] + "..." if len(value) > 50 else value,
                        )

                    # Check for null bytes
                    if "\x00" in value:
                        result.add_warning(
                            f"Field '{field_name}' in row {i} contains null bytes"
                        )

        return result


class MissingValuesValidator(BaseValidator):
    """Validates missing values in dataset."""

    def __init__(
        self,
        max_missing_ratio: float = 0.5,
    ) -> None:
        """
        Args:
            max_missing_ratio: Maximum ratio of missing values allowed per field.
        """
        self.max_missing_ratio = max_missing_ratio

    def validate(self, data: list[dict[str, Any]]) -> ValidationResult:
        """Check for missing values."""
        result = ValidationResult(is_valid=True)

        if not data:
            return result

        n_samples = len(data)
        all_fields = set()
        for sample in data:
            all_fields.update(sample.keys())

        for field_name in all_fields:
            missing_count = sum(
                1 for sample in data
                if field_name not in sample or sample[field_name] is None
            )
            missing_ratio = missing_count / n_samples

            if missing_ratio > self.max_missing_ratio:
                result.add_error(
                    field=field_name,
                    message=f"Too many missing values: {missing_ratio:.1%}",
                    value={"missing": missing_count, "total": n_samples},
                )
            elif missing_ratio > 0:
                result.add_warning(
                    f"Field '{field_name}' has {missing_count} missing values ({missing_ratio:.1%})"
                )

        return result


class DataTypeValidator(BaseValidator):
    """Validates data types of fields."""

    def __init__(
        self,
        expected_types: dict[str, type] | None = None,
    ) -> None:
        """
        Args:
            expected_types: Dict mapping field names to expected types.
        """
        self.expected_types = expected_types or {}

    def validate(self, data: list[dict[str, Any]]) -> ValidationResult:
        """Validate field types."""
        result = ValidationResult(is_valid=True)

        for field_name, expected_type in self.expected_types.items():
            type_errors = 0
            for i, sample in enumerate(data):
                if field_name in sample:
                    value = sample[field_name]
                    if value is not None and not isinstance(value, expected_type):
                        type_errors += 1
                        if type_errors <= 5:  # Limit error reports
                            result.add_error(
                                field=field_name,
                                message=f"Expected {expected_type.__name__}, got {type(value).__name__}",
                                row=i,
                                value=value,
                            )

        return result


class ValueRangeValidator(BaseValidator):
    """Validates numeric values are within expected ranges."""

    def __init__(
        self,
        ranges: dict[str, tuple[Any, Any]] | None = None,
    ) -> None:
        """
        Args:
            ranges: Dict mapping field names to (min, max) tuples.
        """
        self.ranges = ranges or {}

    def validate(self, data: list[dict[str, Any]]) -> ValidationResult:
        """Validate value ranges."""
        result = ValidationResult(is_valid=True)

        for field_name, (min_val, max_val) in self.ranges.items():
            out_of_range = 0
            for i, sample in enumerate(data):
                if field_name in sample:
                    value = sample[field_name]
                    if value is not None:
                        try:
                            value = float(value)
                            if value < min_val or value > max_val:
                                out_of_range += 1
                                if out_of_range <= 5:
                                    result.add_warning(
                                        f"Field '{field_name}' in row {i} has value {value} outside range [{min_val}, {max_val}]"
                                    )
                        except (ValueError, TypeError):
                            pass

        return result


class TextQualityValidator(BaseValidator):
    """Validates text quality in string fields."""

    def __init__(
        self,
        min_length: int = 1,
        max_length: int | None = None,
        check_duplicates: bool = True,
        check_empty: bool = True,
    ) -> None:
        """
        Args:
            min_length: Minimum text length.
            max_length: Maximum text length.
            check_duplicates: Check for duplicate samples.
            check_empty: Check for empty strings.
        """
        self.min_length = min_length
        self.max_length = max_length
        self.check_duplicates = check_duplicates
        self.check_empty = check_empty

    def validate(self, data: list[dict[str, Any]]) -> ValidationResult:
        """Validate text quality."""
        result = ValidationResult(is_valid=True)

        seen_samples = set()
        text_fields = ["input", "output", "text", "instruction", "response"]

        for i, sample in enumerate(data):
            sample_str = str(sample)

            # Check for duplicates
            if self.check_duplicates:
                if sample_str in seen_samples:
                    result.add_warning(f"Row {i} appears to be a duplicate")
                seen_samples.add(sample_str)

            # Check text fields
            for field_name in text_fields:
                if field_name in sample:
                    value = sample[field_name]
                    if isinstance(value, str):
                        length = len(value.strip())

                        if self.check_empty and length < self.min_length:
                            result.add_error(
                                field=field_name,
                                message=f"Text too short: {length} chars (min: {self.min_length})",
                                row=i,
                                value=value,
                            )

                        if self.max_length and length > self.max_length:
                            result.add_warning(
                                f"Field '{field_name}' in row {i} is very long: {length} chars"
                            )

        return result


class DatasetValidator:
    """Complete dataset validation pipeline.

    Runs multiple validators and combines results.
    """

    def __init__(
        self,
        required_fields: list[str] | None = None,
        expected_types: dict[str, type] | None = None,
        min_samples: int = 1,
        max_missing_ratio: float = 0.5,
    ) -> None:
        """
        Args:
            required_fields: Fields that must exist in every sample.
            expected_types: Expected types for fields.
            min_samples: Minimum number of samples required.
            max_missing_ratio: Maximum ratio of missing values allowed.
        """
        self.required_fields = required_fields or ["input", "output"]
        self.expected_types = expected_types or {}
        self.min_samples = min_samples
        self.max_missing_ratio = max_missing_ratio

        # Initialize validators
        self.validators: list[BaseValidator] = [
            SchemaValidator(required_fields=self.required_fields),
            EncodingValidator(),
            MissingValuesValidator(max_missing_ratio=self.max_missing_ratio),
            DataTypeValidator(expected_types=self.expected_types),
            TextQualityValidator(),
        ]

    def validate(self, data: list[dict[str, Any]]) -> ValidationResult:
        """Run all validators and combine results."""
        result = ValidationResult(is_valid=True)

        # Check minimum samples
        if len(data) < self.min_samples:
            result.add_error(
                field="dataset",
                message=f"Too few samples: {len(data)} (min: {self.min_samples})",
            )
            return result

        # Run all validators
        for validator in self.validators:
            validator_result = validator.validate(data)
            result.errors.extend(validator_result.errors)
            result.warnings.extend(validator_result.warnings)
            if not validator_result.is_valid:
                result.is_valid = False

        return result
