"""Prompt template definitions for AutoMLChain."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptTemplate:
    """Represents a prompt template.

    Supports variable substitution using {variable} syntax.

    Attributes:
        name: Unique name for this template.
        template: Template string with {variable} placeholders.
        variables: List of variable names found in template.
        version: Version identifier.
        metadata: Additional metadata about the template.
    """

    name: str
    template: str
    variables: list[str] = field(default_factory=list)
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Extract variables from template."""
        if not self.variables and self.template:
            self.variables = self._extract_variables(self.template)

    @staticmethod
    def _extract_variables(template: str) -> list[str]:
        """Extract variable names from template.

        Args:
            template: Template string.

        Returns:
            List of variable names.
        """
        pattern = r"\{(\w+)\}"
        return list(set(re.findall(pattern, template)))

    def format(self, **kwargs: Any) -> str:
        """Format the template with provided values.

        Args:
            **kwargs: Variable values to substitute.

        Returns:
            Formatted string.

        Raises:
            KeyError: If a required variable is missing.
        """
        # Check for missing variables
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise KeyError(f"Missing variables: {missing}")

        return self.template.format(**kwargs)

    def validate(self, **kwargs: Any) -> list[str]:
        """Validate that all variables are provided.

        Args:
            **kwargs: Variable values to check.

        Returns:
            List of missing variable names (empty if all present).
        """
        missing = set(self.variables) - set(kwargs.keys())
        return list(missing)

    def get_hash(self) -> str:
        """Get a hash of the template content.

        Returns:
            SHA256 hash of template string.
        """
        return hashlib.sha256(self.template.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "variables": self.variables,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptTemplate:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            template=data["template"],
            variables=data.get("variables", []),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PromptVariant:
    """A variant of a prompt template.

    Represents a variation of a base template for experimentation.
    """

    base_template: PromptTemplate
    variant_id: str
    template: str
    strategy: str  # e.g., "reorder", "add_context", "change_wording"
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs: Any) -> str:
        """Format the variant template."""
        # Extract variables from this variant's template
        pattern = r"\{(\w+)\}"
        variables = list(set(re.findall(pattern, self.template)))

        # Format with provided values
        missing = set(variables) - set(kwargs.keys())
        if missing:
            raise KeyError(f"Missing variables: {missing}")

        return self.template.format(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_template": self.base_template.name,
            "variant_id": self.variant_id,
            "template": self.template,
            "strategy": self.strategy,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class PromptExperiment:
    """Represents a prompt experiment with multiple variants."""

    name: str
    base_template: PromptTemplate
    variants: list[PromptVariant] = field(default_factory=list)
    best_variant: PromptVariant | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_variant(self, variant: PromptVariant) -> None:
        """Add a variant to the experiment."""
        self.variants.append(variant)

    def select_best(self) -> PromptVariant | None:
        """Select the best variant based on scores."""
        scored = [v for v in self.variants if v.score is not None]
        if not scored:
            return None

        self.best_variant = max(scored, key=lambda v: v.score)
        return self.best_variant

    def get_variant(self, variant_id: str) -> PromptVariant | None:
        """Get a variant by ID."""
        for variant in self.variants:
            if variant.variant_id == variant_id:
                return variant
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "base_template": self.base_template.to_dict(),
            "variants": [v.to_dict() for v in self.variants],
            "best_variant": self.best_variant.to_dict() if self.best_variant else None,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
