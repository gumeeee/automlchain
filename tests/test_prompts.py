"""Tests for prompts module."""

import pytest
from automlchain.prompts import PromptTemplate, PromptEngine


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_create_template(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            name="classification",
            template="Classify: {input}\nCategory: {output}",
        )
        assert template.name == "classification"
        assert len(template.variables) == 2
        assert "input" in template.variables
        assert "output" in template.variables

    def test_template_format(self):
        """Test formatting a template."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}!",
        )
        result = template.format(name="World")
        assert result == "Hello World!"

    def test_template_format_extra_kwargs(self):
        """Test formatting with extra kwargs is ignored."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}!",
        )
        result = template.format(name="World", extra="ignored")
        assert result == "Hello World!"

    def test_template_format_missing_kwargs(self):
        """Test formatting with missing kwargs raises error."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}!",
        )
        with pytest.raises(KeyError):
            template.format()

    def test_to_dict(self):
        """Test serialization."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}!",
        )
        d = template.to_dict()
        assert d["name"] == "test"
        assert d["template"] == "Hello {name}!"
        # Variables are lowercase
        assert "name" in d["variables"]

    def test_get_hash(self):
        """Test hash generation."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}!",
        )
        hash1 = template.get_hash()
        hash2 = template.get_hash()
        assert hash1 == hash2

        template2 = PromptTemplate(
            name="test",
            template="Hello {other}!",
        )
        hash3 = template2.get_hash()
        assert hash1 != hash3


class TestPromptEngine:
    """Tests for PromptEngine class."""

    def test_create_engine(self):
        """Test creating a prompt engine."""
        engine = PromptEngine(seed=42)
        assert engine is not None
        assert engine.seed == 42

    def test_generate_variants(self):
        """Test generating prompt variants."""
        engine = PromptEngine(seed=42)
        template = PromptTemplate(
            name="base",
            template="Classify: {input}\nCategory: {output}",
        )
        variants = engine.generate_variants(template, n=3)
        assert len(variants) == 3
        assert all(isinstance(v, PromptTemplate) for v in variants)
