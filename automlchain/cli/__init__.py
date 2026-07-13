"""CLI module for AutoMLChain.

Provides command-line interface for training, monitoring, and deployment.
"""

from .main import cli, main
from .commands import (
    train_command,
    status_command,
    cancel_command,
    evaluate_command,
    deploy_command,
)

__all__ = [
    "cli",
    "main",
    "train_command",
    "status_command",
    "cancel_command",
    "evaluate_command",
    "deploy_command",
]
