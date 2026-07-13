"""Cancel command for the CLI."""

from __future__ import annotations

import sys
import os

import click


def _get_api_key(provider_name: str) -> str:
    """Get API key from environment."""
    env_vars = {
        "replicate": "REPLICATE_API_TOKEN",
        "mock": "mock-key",
    }

    env_var = env_vars.get(provider_name, "")
    api_key = os.environ.get(env_var, "")

    if not api_key and provider_name != "mock":
        return "mock-key"  # Fallback for testing

    return api_key or "mock-key"


@click.command("cancel")
@click.option(
    "--job-id",
    "-j",
    required=True,
    help="Training job ID to cancel",
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
    "--force",
    "-f",
    is_flag=True,
    help="Skip confirmation prompt",
)
def cancel_command(
    job_id: str,
    provider_name: str,
    force: bool,
) -> None:
    """Cancel a running training job.

    Example:
        $ automlchain cancel --job-id abc123
        $ automlchain cancel --job-id abc123 --force
    """
    # Confirmation prompt (skipped if --force)
    if not force:
        if not click.confirm(
            f"Are you sure you want to cancel job {job_id}?",
        ):
            click.echo("Cancelled.")
            return

    click.echo(f"Cancelling job {job_id}...")

    try:
        from ...providers import ProviderRegistry

        api_key = _get_api_key(provider_name)
        provider = ProviderRegistry.get(provider_name, api_key=api_key)
        provider.cancel_job(job_id)

        click.echo(click.style("✓ Job cancelled successfully!", fg="green"))

    except Exception as e:
        click.echo(f"Error cancelling job: {e}", err=True)
        sys.exit(1)
