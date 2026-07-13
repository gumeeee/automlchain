# AutoMLChain

**Fine-tuning made accessible** - A Python library for automated fine-tuning of LLMs.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/gumeeee/automlchain/actions/workflows/tests.yml/badge.svg)](https://github.com/gumeeee/automlchain/actions)
[![codecov](https://codecov.io/gh/gumeeee/automlchain/branch/main/graph/badge.svg)](https://codecov.io/gh/gumeeee/automlchain)

## Overview

AutoMLChain orchestrates the complete fine-tuning workflow — from dataset upload to model evaluation — automating decisions that normally require specialized ML knowledge.

**Tagline:** *"Fine-tuning made accessible"*

**Analogy:** Terraform for fine-tuning — you describe what you want, the library figures out how to get it.

## Key Features

- **Dataset Management** - Upload, validate, and convert datasets (JSONL, CSV, Parquet)
- **Prompt Engineering** - Template creation and variant generation
- **Training Orchestration** - Unified API for multiple fine-tuning providers
- **Auto-Evaluation** - Built-in metrics (RMSE, F1, MAE, Accuracy)
- **CLI Interface** - Command-line tools for training and monitoring
- **Dynamic Pricing** - Real-time cost estimation and tracking
- **Extensible** - Plugin system for new providers and metrics

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install automlchain

# Or using pip
pip install automlchain
```

### Python API

```python
from automlchain import AutoMLPipeline

# Initialize pipeline with Replicate
pipeline = AutoMLPipeline(provider="replicate")

# Upload dataset
pipeline.upload_dataset("reviews.jsonl")

# Train model
job = pipeline.train(
    model="meta/llama-3-8b-instruct",
    epochs=3,
    learning_rate=1e-4,
)

# Wait for completion
result = pipeline.training_orchestrator.wait_for_completion(job.job_id)

# Deploy and use
model = pipeline.deploy()
response = model.predict("This product is amazing!")
```

### CLI

```bash
# Install CLI
uv pip install automlchain

# Start training
automlchain train --dataset data.jsonl --model meta/llama-3-8b

# Check status
automlchain status --job-id abc123

# Cancel job
automlchain cancel --job-id abc123 --force

# Evaluate predictions
automlchain evaluate --predictions preds.jsonl --references refs.jsonl
```

## Supported Providers

| Provider | Status | Notes |
|----------|--------|-------|
| Replicate | ✅ Phase 1 | Primary for inference |
| Together AI | ✅ Phase 2 | Fine-tuning cloud (simpler API) |
| Local | ✅ Phase 2 | Fine-tuning on your GPU (no API costs) |
| Mock | ✅ | For testing without API calls |

### Choosing a Provider

| Use Case | Recommended Provider |
|----------|---------------------|
| Quick test, no GPU | Together AI |
| Production fine-tuning | Together AI |
| Privacy/data on-premise | Local |
| Maximum control | Local |
| Just inference | Replicate |

### Local Provider Requirements

```bash
pip install torch transformers peft accelerate datasets bitsandbytes
```

See [notebooks/test_local_provider.ipynb](notebooks/test_local_provider.ipynb) for a complete example.

## Supported Metrics (MVP)

| Metric | Type | Use Case |
|--------|------|----------|
| RMSE | Regression | Ratings, scores (lower is better) |
| MAE | Regression | Absolute error (lower is better) |
| F1 | Classification | Precision/recall balance |
| Accuracy | Classification | Overall accuracy |

## Architecture

```
automlchain/
├── core/           # Pipeline, configuration, exceptions
├── datasets/       # Dataset management, validation, parsing
│   ├── formats/    # Format-specific parsers
│   └── validators/ # Data validation
├── evaluation/     # Metrics and evaluation suite
│   └── metrics/    # Individual metric implementations
├── prompts/        # Prompt templates and engine
├── providers/      # Provider abstraction layer
│   ├── base.py     # Base provider interface
│   ├── mock.py     # Mock provider for testing
│   ├── replicate.py # Replicate implementation
│   └── pricing.py  # Dynamic cost estimation
├── training/       # Training orchestration
│   ├── callbacks.py # Progress callbacks
│   ├── jobs.py     # Job state management
│   └── orchestrator.py # Training coordination
└── cli/            # Command-line interface
    └── commands/   # CLI command implementations
```

## Requirements

- Python 3.10+
- No GPU required (cloud training)
- API keys for your chosen provider(s)

## Environment Variables

```bash
# Replicate
export REPLICATE_API_TOKEN="your-token-here"

# Optional: Configure defaults
export AUTOMLCHAIN_PROVIDER=replicate
export AUTOMLCHAIN_MODEL=meta/llama-3-8b-instruct
```

## Configuration

AutoMLChain supports configuration from multiple sources (priority high to low):

1. **Runtime** - Passed to Pipeline constructor
2. **Environment** - `AUTOMLCHAIN_*` variables
3. **Config file** - `~/.automlchain/config.yaml`
4. **Defaults** - Sane defaults in code

## Development

> **Note:** This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

```bash
# Clone repository
git clone https://github.com/gumeeee/automlchain.git
cd automlchain

# Install using uv (recommended)
uv sync

# Activate virtual environment
source .venv/bin/activate

# Install dev dependencies
uv sync --dev

# Or using pip
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=automlchain

# Format code
ruff format .

# Lint
ruff check .

# Type check
mypy automlchain/
```

### Testing

```bash
# Unit tests
pytest tests/

# Smoke tests (interactive notebook)
jupyter notebook notebooks/smoke_tests.ipynb

# Real training tests (requires API keys)
jupyter notebook notebooks/real_training_tests.ipynb
```

## Roadmap

### Phase 1: MVP ✅
- ✅ Dataset upload and validation
- ✅ Training via Replicate API
- ✅ Evaluation with built-in metrics
- ✅ CLI (train, status, cancel)
- ✅ Deploy and inference
- ✅ Dynamic pricing
- ✅ Unit tests and smoke tests

### Phase 2: Auto-Optimization
- Hyperparameter search (Grid, Random, Bayesian)
- Budget management
- Early stopping
- Multi-provider abstraction
- Advanced metrics (ROUGE, BLEU)

### Phase 3: Productization
- Web dashboard
- REST API
- Pricing tiers
- Enterprise features

## License

MIT License - see [LICENSE](LICENSE) for details.
