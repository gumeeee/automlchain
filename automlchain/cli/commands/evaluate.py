"""Evaluate command for the CLI."""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from ...evaluation import EvaluationSuite, get_metric


@click.command("evaluate")
@click.option(
    "--predictions",
    "-p",
    "predictions_file",
    required=True,
    type=click.Path(exists=True),
    help="File containing predictions (JSON array)",
)
@click.option(
    "--references",
    "-r",
    "references_file",
    required=True,
    type=click.Path(exists=True),
    help="File containing references (JSON array)",
)
@click.option(
    "--metrics",
    "-m",
    multiple=True,
    default=["rmse", "f1", "mae", "accuracy"],
    help="Metrics to compute",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(),
    help="Output file for results (JSON)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.pass_context
def evaluate_command(
    ctx: click.Context,
    predictions_file: str,
    references_file: str,
    metrics: tuple[str, ...],
    output_file: str | None,
    output_format: str,
) -> None:
    """Evaluate model predictions against references.

    Example:
        $ automlchain evaluate --predictions preds.json --references refs.json --metrics rmse f1
    """
    try:
        # Load predictions
        with open(predictions_file, "r") as f:
            predictions = json.load(f)

        # Load references
        with open(references_file, "r") as f:
            references = json.load(f)

        if not isinstance(predictions, list):
            click.echo("Error: Predictions must be a JSON array", err=True)
            sys.exit(1)

        if not isinstance(references, list):
            click.echo("Error: References must be a JSON array", err=True)
            sys.exit(1)

        if len(predictions) != len(references):
            click.echo(
                f"Error: Length mismatch - {len(predictions)} predictions, "
                f"{len(references)} references",
                err=True,
            )
            sys.exit(1)

        click.echo(f"Evaluating {len(predictions)} samples...")
        click.echo(f"Metrics: {', '.join(metrics)}")

        # Create evaluation suite
        suite = EvaluationSuite()
        for metric_name in metrics:
            try:
                metric = get_metric(metric_name)
                suite.add_metric(metric_name, metric)
            except ValueError:
                click.echo(f"Warning: Unknown metric '{metric_name}', skipping")

        # Run evaluation
        result = suite.evaluate(predictions, references)

        # Output results
        if output_format == "json":
            output_data = result.to_dict()
            output_str = json.dumps(output_data, indent=2)

            if output_file:
                with open(output_file, "w") as f:
                    f.write(output_str)
                click.echo(f"Results written to {output_file}")
            else:
                click.echo(output_str)
        else:
            # Text format
            click.echo("\n" + str(result))

            if output_file:
                with open(output_file, "w") as f:
                    f.write(str(result))
                click.echo(f"\nResults written to {output_file}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
