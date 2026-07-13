"""Datasets module for AutoMLChain.

Handles dataset upload, validation, transformation, and versioning.
"""

from .types import (
    Dataset,
    DatasetStats,
    ValidationError,
    ValidationResult,
    DatasetVersion,
)
from .manager import DatasetManager
from .validators import (
    DatasetValidator,
    SchemaValidator,
    EncodingValidator,
    MissingValuesValidator,
    TextQualityValidator,
)
from .formats import (
    FormatParser,
    JSONLParser,
    CSVParser,
    ParquetParser,
    HuggingFaceDatasetParser,
    get_parser,
    detect_format,
)

__all__ = [
    # Types
    "Dataset",
    "DatasetStats",
    "ValidationError",
    "ValidationResult",
    "DatasetVersion",
    # Manager
    "DatasetManager",
    # Validators
    "DatasetValidator",
    "SchemaValidator",
    "EncodingValidator",
    "MissingValuesValidator",
    "TextQualityValidator",
    # Formats
    "FormatParser",
    "JSONLParser",
    "CSVParser",
    "ParquetParser",
    "HuggingFaceDatasetParser",
    "get_parser",
    "detect_format",
]
