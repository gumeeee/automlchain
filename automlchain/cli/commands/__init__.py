"""CLI commands for AutoMLChain."""

from __future__ import annotations

import json
import sys
from typing import Any

import click
import structlog

from .train import train_command
from .status import status_command
from .cancel import cancel_command
from .evaluate import evaluate_command
from .deploy import deploy_command

logger = structlog.get_logger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="automlchain")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:
    """AutoMLChain - Fine-tuning made accessible.

    A Python library for automated fine-tuning of LLMs.
    """
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level

    # Configure logging
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib, log_level.upper(), structlog.stdlib.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Register commands
cli.add_command(train_command)
cli.add_command(status_command)
cli.add_command(cancel_command)
cli.add_command(evaluate_command)
cli.add_command(deploy_command)


def main() -> None:
    """Entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nAborted.", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("cli_error")
        sys.exit(1)


if __name__ == "__main__":
    main()
