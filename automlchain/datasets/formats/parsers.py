"""Format parsers for different dataset file types."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import orjson


class FormatParser(ABC):
    """Base class for format parsers."""

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """List of formats this parser supports."""
        ...

    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        """Check if this parser can handle the given file."""
        ...

    @abstractmethod
    def parse(self, path: Path, encoding: str | None = None) -> list[dict[str, Any]]:
        """Parse the file and return list of records."""
        ...

    @abstractmethod
    def write(
        self,
        data: list[dict[str, Any]],
        path: Path,
        encoding: str | None = None,
    ) -> None:
        """Write data to file in this format."""
        ...


class JSONLParser(FormatParser):
    """Parser for JSON Lines format.

    Each line is a valid JSON object.
    """

    @property
    def supported_formats(self) -> list[str]:
        return ["jsonl", "ndjson", "jsonl.gz"]

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in [".jsonl", ".ndjson", ".gz"]

    def parse(self, path: Path, encoding: str | None = None) -> list[dict[str, Any]]:
        """Parse JSONL file, auto-detecting encoding."""
        # Try UTF-8 first
        enc = encoding or "utf-8"

        if path.suffix == ".gz":
            # Handle gzipped JSONL
            import gzip
            with gzip.open(path, "rt", encoding=enc) as f:
                return [json.loads(line) for line in f if line.strip()]
        else:
            # Try UTF-8, fall back to latin-1 if needed
            try:
                with open(path, "r", encoding=enc) as f:
                    return [json.loads(line) for line in f if line.strip()]
            except UnicodeDecodeError:
                with open(path, "r", encoding="latin-1") as f:
                    return [json.loads(line) for line in f if line.strip()]

    def write(
        self,
        data: list[dict[str, Any]],
        path: Path,
        encoding: str | None = None,
    ) -> None:
        """Write data to JSONL file."""
        enc = encoding or "utf-8"
        with open(path, "w", encoding=enc) as f:
            for record in data:
                f.write(orjson.dumps(record).decode() + "\n")


class CSVParser(FormatParser):
    """Parser for CSV format.

    Supports auto-detection of delimiters and headers.
    """

    @property
    def supported_formats(self) -> list[str]:
        return ["csv", "tsv", "csv.gz"]

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in [".csv", ".tsv", ".gz"]

    def parse(self, path: Path, encoding: str | None = None) -> list[dict[str, Any]]:
        """Parse CSV file with auto-delimiter detection."""
        enc = encoding or "utf-8"

        # Detect delimiter
        delimiter = self._detect_delimiter(path, enc)

        # Handle gzipped CSV
        if path.suffix == ".gz":
            import gzip

            def opener() -> Any:
                return gzip.open(path, "rt", encoding=enc)  # type: ignore[return-value]

        else:

            def opener() -> Any:
                return open(path, "r", encoding=enc)  # type: ignore[return-value]

        with opener() as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return list(reader)

    def _detect_delimiter(self, path: Path, encoding: str) -> str:
        """Auto-detect CSV delimiter."""
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                sample = f.read(4096)

            # Check for tabs
            if "\t" in sample:
                return "\t"

            # Count delimiters
            comma_count = sample.count(",")
            semicolon_count = sample.count(";")

            if semicolon_count > comma_count:
                return ";"
            return ","
        except Exception:
            return ","

    def write(
        self,
        data: list[dict[str, Any]],
        path: Path,
        encoding: str | None = None,
    ) -> None:
        """Write data to CSV file."""
        if not data:
            return

        enc = encoding or "utf-8"
        fieldnames = list(data[0].keys())

        with open(path, "w", encoding=enc, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


class ParquetParser(FormatParser):
    """Parser for Parquet format.

    Requires pyarrow or pandas with parquet support.
    """

    @property
    def supported_formats(self) -> list[str]:
        return ["parquet", "pq"]

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in [".parquet", ".pq"]

    def parse(self, path: Path, encoding: str | None = None) -> list[dict[str, Any]]:
        """Parse Parquet file."""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict(orient="records")
        except ImportError:
            raise ImportError(
                "pandas is required for Parquet support. "
                "Install with: pip install automlchain[parquet] or pip install pandas pyarrow"
            )

    def write(
        self,
        data: list[dict[str, Any]],
        path: Path,
        encoding: str | None = None,
    ) -> None:
        """Write data to Parquet file."""
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_parquet(path)
        except ImportError:
            raise ImportError(
                "pandas is required for Parquet support. "
                "Install with: pip install automlchain[parquet] or pip install pandas pyarrow"
            )


class HuggingFaceDatasetParser(FormatParser):
    """Parser for HuggingFace datasets.

    Supports loading from local cache or remote.
    """

    @property
    def supported_formats(self) -> list[str]:
        return ["huggingface", "hf"]

    def can_parse(self, path: Path) -> bool:
        return str(path).startswith("hf://") or "huggingface" in str(path)

    def parse(self, path: Path, encoding: str | None = None) -> list[dict[str, Any]]:
        """Load HuggingFace dataset."""
        try:
            from datasets import load_dataset

            dataset_path = str(path).replace("hf://", "")
            ds = load_dataset(dataset_path)
            return list(ds)
        except ImportError:
            raise ImportError(
                "datasets is required for HuggingFace support. "
                "Install with: pip install automlchain[hf] or pip install datasets"
            )

    def write(
        self,
        data: list[dict[str, Any]],
        path: Path,
        encoding: str | None = None,
    ) -> None:
        """HuggingFace datasets are read-only for now."""
        raise NotImplementedError(
            "Writing to HuggingFace format is not supported. "
            "Use JSONL or CSV instead."
        )


# Registry of parsers
PARSERS: list[FormatParser] = [
    JSONLParser(),
    CSVParser(),
    ParquetParser(),
    HuggingFaceDatasetParser(),
]


def get_parser(format: str) -> FormatParser:
    """Get the appropriate parser for a format.

    Args:
        format: Format name (jsonl, csv, parquet, auto).

    Returns:
        FormatParser instance.

    Raises:
        ValueError: If format is not supported.
    """
    format = format.lower().strip()

    if format == "auto":
        raise ValueError("Use detect_format() to determine format first")

    for parser in PARSERS:
        if format in parser.supported_formats:
            return parser

    raise ValueError(f"Unsupported format: {format}. Supported: {PARSERS}")


def detect_format(path: Path) -> str:
    """Auto-detect the format of a file.

    Args:
        path: Path to the file.

    Returns:
        Detected format name.
    """
    suffix = path.suffix.lower()

    format_map = {
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".csv": "csv",
        ".tsv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
    }

    # Check suffix
    if suffix in format_map:
        return format_map[suffix]

    # Check content (for JSONL without extension)
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            first_line = f.readline()
            if first_line.strip():
                json.loads(first_line)
                return "jsonl"
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    # Default to CSV
    return "csv"
