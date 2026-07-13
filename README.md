# AutoMLChain

**Fine-tuning made accessible** - A Python library for automated fine-tuning of LLMs.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
- **Extensible** - Plugin system for new providers and metrics

## Quick Start

### Installation

```bash
pip install automlchain
```

### Python API

```python
from automlchain import AutoMLPipeline

# Initialize pipeline
pipeline = AutoMLPipeline(provider="replicate")

# Upload dataset
pipeline.upload_dataset("reviews.jsonl")

# Train model
result = pipeline.train(
    model="meta/llama-3-8b-instruct",
    epochs=3,
    learning_rate=1e-4,
)

# Deploy and use
model = pipeline.deploy()
response = model.predict("This product is amazing!")
```

### CLI

```bash
# Start training
automlchain train --dataset data.jsonl --model meta/llama-3-8b

# Check status
automlchain status --job-id abc123

# Cancel job
automlchain cancel --job-id abc123
```

## Supported Providers

| Provider | Status | Notes |
|----------|--------|-------|
| Replicate | ✅ Phase 1 | Primary provider for MVP |
| Together AI | 🔜 Phase 2 | Planned |
| Anyscale | 🔜 Phase 2 | Planned |

## Supported Metrics (MVP)

| Metric | Use Case |
|--------|----------|
| RMSE | Regression (ratings, scores) |
| MAE | Regression (absolute error) |
| F1 | Classification (precision/recall balance) |
| Accuracy | Classification (overall accuracy) |

## Architecture

```
automlchain/
├── core/           # Pipeline and configuration
├── datasets/       # Dataset management
├── prompts/       # Prompt templates
├── training/       # Training orchestration
├── evaluation/     # Evaluation metrics
├── providers/      # Provider abstraction
└── cli/           # Command-line interface
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

```bash
# Clone repository
git clone https://github.com/automlchain/automlchain.git
cd automlchain

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format .

# Lint
ruff check .
```

## Roadmap

### Phase 1: MVP (Current)
- ✅ Dataset upload and validation
- ✅ Training via Replicate API
- ✅ Evaluation with built-in metrics
- ✅ CLI (train, status, cancel)
- ✅ Deploy and inference

### Phase 2: Auto-Optimization
- Hyperparameter search (Grid, Random, Bayesian)
- Budget management
- Multi-provider abstraction
- Advanced metrics (ROUGE, BLEU)

### Phase 3: Productization
- Web dashboard
- REST API
- Pricing tiers
- Enterprise features

## License

MIT License - see [LICENSE](LICENSE) for details.
