"""Prompt engine for generating and optimizing prompt templates."""

from __future__ import annotations

import random
import re
import time
from typing import Any

import structlog

from .templates import PromptTemplate, PromptVariant, PromptExperiment

logger = structlog.get_logger(__name__)


class PromptEngine:
    """Engine for prompt template creation and variation.

    Generates template variants for experimentation and optimization.

    Example:
        >>> engine = PromptEngine()
        >>> template = engine.create_template(
        ...     "Classify: {text}\\nCategory:",
        ...     name="classifier"
        ... )
        >>> variants = engine.generate_variants(template, n=5)
    """

    def __init__(
        self,
        *,
        default_variables: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            default_variables: Default variable names to expect.
            seed: Random seed for reproducibility.
        """
        self.default_variables = default_variables or ["instruction", "input", "output"]
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self._experiments: dict[str, PromptExperiment] = {}

    def create_template(
        self,
        template: str,
        name: str = "default",
        **kwargs: Any,
    ) -> PromptTemplate:
        """Create a new prompt template.

        Args:
            template: Template string with {variable} placeholders.
            name: Name for this template.
            **kwargs: Additional metadata.

        Returns:
            PromptTemplate instance.
        """
        # Extract variables
        variables = self._extract_variables(template)

        prompt_template = PromptTemplate(
            name=name,
            template=template,
            variables=variables,
            metadata={
                "created_at": time.time(),
                **kwargs,
            },
        )

        logger.info(
            "template_created",
            name=name,
            variables=variables,
        )

        return prompt_template

    def _extract_variables(self, template: str) -> list[str]:
        """Extract variable names from template."""
        pattern = r"\{(\w+)\}"
        found = list(set(re.findall(pattern, template)))

        # Add any default variables that weren't in template
        for default in self.default_variables:
            if default not in found:
                found.append(default)

        return found

    def generate_variants(
        self,
        template: PromptTemplate,
        n: int = 5,
        strategies: list[str] | None = None,
    ) -> list[PromptTemplate]:
        """Generate template variants for experimentation.

        Args:
            template: Base template to vary.
            n: Number of variants to generate.
            strategies: List of variation strategies to use.
                Options: reorder, add_context, change_wording, few_shot

        Returns:
            List of PromptTemplate variants.
        """
        if strategies is None:
            strategies = ["reorder", "add_context", "change_wording"]

        variants: list[PromptTemplate] = []

        for i in range(n):
            strategy = random.choice(strategies)
            variant = self._generate_variant(template, strategy, i)
            variants.append(variant)

        logger.info(
            "variants_generated",
            base_template=template.name,
            n_variants=len(variants),
            strategies_used=strategies,
        )

        return variants

    def _generate_variant(
        self,
        template: PromptTemplate,
        strategy: str,
        index: int,
    ) -> PromptTemplate:
        """Generate a single variant using specified strategy."""
        variant_methods = {
            "reorder": self._variant_reorder,
            "add_context": self._variant_add_context,
            "change_wording": self._variant_change_wording,
            "few_shot": self._variant_few_shot,
        }

        method = variant_methods.get(strategy, self._variant_change_wording)
        return method(template, index)

    def _variant_reorder(
        self,
        template: PromptTemplate,
        index: int,
    ) -> PromptTemplate:
        """Reorder variables in template."""
        pattern = r"\{(\w+)\}"
        variables_found = re.findall(pattern, template.template)

        if len(variables_found) < 2:
            return PromptTemplate(
                name=f"{template.name}_v{index}",
                template=template.template,
                variables=template.variables,
                metadata={
                    **template.metadata,
                    "strategy": "reorder",
                    "original": template.name,
                },
            )

        # Shuffle variables (with seed for reproducibility)
        shuffled = variables_found.copy()
        random.shuffle(shuffled)

        # Build new template using position-aware replacement
        # Use regex to replace each variable in order of appearance
        new_template = template.template
        mapping = dict(zip(variables_found, shuffled))

        # Replace each unique variable once
        for orig_var, new_var in mapping.items():
            # Only replace if this is the first occurrence of orig_var
            # to avoid replacing all occurrences at once
            if orig_var != new_var:
                # Find first occurrence and replace only that one
                pattern = f"{{{orig_var}}}"
                new_pattern = f"{{{new_var}}}"
                parts = new_template.split(pattern)
                if len(parts) > 1:
                    new_template = new_pattern.join(parts)

        return PromptTemplate(
            name=f"{template.name}_v{index}",
            template=new_template,
            variables=shuffled,
            metadata={
                **template.metadata,
                "strategy": "reorder",
                "original": template.name,
                "reorder_mapping": dict(zip(variables_found, shuffled)),
            },
        )

    def _variant_add_context(
        self,
        template: PromptTemplate,
        index: int,
    ) -> PromptTemplate:
        """Add context prefixes to variables."""
        contexts = [
            "Please {var}",
            "Given the following {var}:",
            "Consider this {var}:",
            "Here is the {var}:",
        ]

        context = random.choice(contexts)
        new_template = template.template

        # Add context before first variable
        first_var = template.variables[0] if template.variables else "input"
        new_template = context.format(var=first_var) + "\\n" + new_template

        return PromptTemplate(
            name=f"{template.name}_v{index}",
            template=new_template,
            variables=template.variables,
            metadata={
                **template.metadata,
                "strategy": "add_context",
                "original": template.name,
            },
        )

    def _variant_change_wording(
        self,
        template: PromptTemplate,
        index: int,
    ) -> PromptTemplate:
        """Change instruction wording."""
        # Common instruction variations
        instruction_map = {
            "classify": ["Classify", "Determine", "Identify the category of"],
            "summarize": ["Summarize", "Give a brief summary of", "Condense"],
            "translate": ["Translate", "Convert to", "Render in"],
            "explain": ["Explain", "Describe", "Tell me about"],
            "generate": ["Generate", "Create", "Produce"],
        }

        new_template = template.template

        # Replace common instruction words
        for key, alternatives in instruction_map.items():
            if key.lower() in new_template.lower():
                pattern = re.compile(re.escape(key), re.IGNORECASE)
                replacement = random.choice(alternatives)
                new_template = pattern.sub(replacement, new_template, count=1)

        return PromptTemplate(
            name=f"{template.name}_v{index}",
            template=new_template,
            variables=template.variables,
            metadata={
                **template.metadata,
                "strategy": "change_wording",
                "original": template.name,
            },
        )

    def _variant_few_shot(
        self,
        template: PromptTemplate,
        index: int,
    ) -> PromptTemplate:
        """Add few-shot examples to template."""
        # Insert example between instruction and input
        example_template = (
            "\\n\\nExample:\\n"
            "Input: {example_input}\\n"
            "Output: {example_output}\\n\\n"
        )

        # Add example template
        new_template = template.template
        parts = new_template.split("{input}")
        if len(parts) == 2:
            new_template = (
                parts[0]
                + example_template
                + "{input}"
                + parts[1]
            )

        variables = template.variables.copy()
        if "example_input" not in variables:
            variables.extend(["example_input", "example_output"])

        return PromptTemplate(
            name=f"{template.name}_v{index}",
            template=new_template,
            variables=variables,
            metadata={
                **template.metadata,
                "strategy": "few_shot",
                "original": template.name,
            },
        )

    def create_experiment(
        self,
        name: str,
        base_template: PromptTemplate,
        n_variants: int = 5,
    ) -> PromptExperiment:
        """Create a prompt experiment with variants.

        Args:
            name: Name for the experiment.
            base_template: Base template to generate variants from.
            n_variants: Number of variants to generate.

        Returns:
            PromptExperiment with variants.
        """
        variants = self.generate_variants(base_template, n=n_variants)

        # Convert to PromptVariant
        prompt_variants: list[PromptVariant] = []
        for i, variant in enumerate(variants):
            prompt_variants.append(
                PromptVariant(
                    base_template=base_template,
                    variant_id=f"{name}_v{i}",
                    template=variant.template,
                    strategy=variant.metadata.get("strategy", "unknown"),
                    metadata=variant.metadata,
                )
            )

        experiment = PromptExperiment(
            name=name,
            base_template=base_template,
            variants=prompt_variants,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self._experiments[name] = experiment
        return experiment

    def get_experiment(self, name: str) -> PromptExperiment | None:
        """Get an experiment by name."""
        return self._experiments.get(name)

    def update_variant_score(
        self,
        experiment_name: str,
        variant_id: str,
        score: float,
    ) -> None:
        """Update the score for a variant."""
        experiment = self._experiments.get(experiment_name)
        if experiment:
            variant = experiment.get_variant(variant_id)
            if variant:
                variant.score = score
                logger.info(
                    "variant_scored",
                    experiment=experiment_name,
                    variant=variant_id,
                    score=score,
                )

    def get_best_variant(
        self,
        experiment_name: str,
    ) -> PromptTemplate | None:
        """Get the best scoring variant from an experiment."""
        experiment = self._experiments.get(experiment_name)
        if not experiment:
            return None

        best = experiment.select_best()
        if best:
            return PromptTemplate(
                name=best.variant_id,
                template=best.template,
                metadata={
                    **best.metadata,
                    "score": best.score,
                    "experiment": experiment_name,
                },
            )
        return None
