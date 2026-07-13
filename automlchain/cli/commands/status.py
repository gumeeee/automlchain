"""Status command for the CLI."""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from ...core import AutoMLPipeline
from ...training.jobs import JobState


@click.command("status")
@click.option(
    "--job-id",
    "-j",
    required=True,
    help="Training job ID to check",
)
@click.option(
    "--provider",
    "-p",
    "provider_name",
    default="replicate",
    type=click.Choice(["replicate", "mock"]),
    help="Provider used for the job",
)
@click.option(
    "--watch/--no-watch",
    default=False,
    help="Watch status continuously",
)
@click.option(
    "--interval",
    type=int,
    default=5,
    help="Seconds between status updates when watching",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.pass_context
def status_command(
    ctx: click.Context,
    job_id: str,
    provider_name: str,
    watch: bool,
    interval: int,
    output_json: bool,
) -> None:
    """Check the status of a training job.

    Example:
        $ automlchain status --job-id abc123
        $ automlchain status --job-id abc123 --watch
    """
    if watch:
        _watch_status(job_id, provider_name, interval, output_json)
    else:
        _check_status(job_id, provider_name, output_json)


def _check_status(
    job_id: str,
    provider_name: str,
    output_json: bool,
) -> None:
    """Check status once."""
    try:
        # Get provider
        from ...providers import ProviderRegistry

        api_key = _get_api_key(provider_name)
        provider = ProviderRegistry.get(provider_name, api_key=api_key)

        # Get status
        status = provider.get_job_status(job_id)

        if output_json:
            click.echo(json.dumps(status.to_dict(), indent=2))
            return

        # Format output
        click.echo(f"\nJob Status: {job_id}")
        click.echo("-" * 50)
        click.echo(f"  Status:    {status.status}")
        click.echo(f"  Progress:  {status.progress:.1f}%")

        if status.epoch > 0:
            click.echo(f"  Epoch:     {status.epoch}/{status.total_epochs}")

        if status.loss is not None:
            click.echo(f"  Loss:      {status.loss:.4f}")

        if status.logs:
            click.echo(f"\nRecent Logs:")
            for log in status.logs[-5:]:
                click.echo(f"  {log}")

        if status.error:
            click.echo(f"\n  Error: {status.error}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _watch_status(
    job_id: str,
    provider_name: str,
    interval: int,
    output_json: bool,
) -> None:
    """Watch status continuously."""
    import time

    try:
        from ...providers import ProviderRegistry

        api_key = _get_api_key(provider_name)
        provider = ProviderRegistry.get(provider_name, api_key=api_key)

        terminal_states = {"completed", "failed", "cancelled", "canceled"}
        last_status = None

        click.echo(f"Watching job {job_id}... (Ctrl+C to stop)")
        click.echo("-" * 50)

        while True:
            try:
                status = provider.get_job_status(job_id)

                # Clear line and print new status
                click.echo(f"\r{click.style('Status:', fg='cyan')} {status.status} "
                          f"({status.progress:.1f}%)     ", nl=False)

                if status.status.lower() in terminal_states:
                    click.echo()  # New line
                    if status.status.lower() == "completed":
                        click.echo(click.style("✓ Training completed!", fg="green"))
                    elif status.status.lower() == "failed":
                        click.echo(click.style("✗ Training failed!", fg="red"))
                        if status.error:
                            click.echo(f"  Error: {status.error}")
                    else:
                        click.echo(click.style("⚠ Training cancelled", fg="yellow"))
                    break

                last_status = status
                time.sleep(interval)

            except KeyboardInterrupt:
                click.echo()
                click.echo("Stopped watching.")
                break

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _get_api_key(provider_name: str) -> str:
    """Get API key from environment or prompt."""
    env_vars = {
        "replicate": "REPLICATE_API_TOKEN",
        "mock": "mock-key",
    }

    env_var = env_vars.get(provider_name, "")
    api_key = os.environ.get(env_var, "")

    if not api_key and provider_name != "mock":
        click.echo(f"Error: {env_var} not set", err=True)
        click.echo(f"Set it with: export {env_var}=your_key")
        sys.exit(1)

    return api_key or "mock-key"
