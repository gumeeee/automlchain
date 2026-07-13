"""Deploy command for the CLI."""

from __future__ import annotations

import json
import sys
import os

import click

from ...providers import ProviderRegistry


@click.command("deploy")
@click.option(
    "--job-id",
    "-j",
    "job_id",
    required=True,
    help="Training job ID to deploy",
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
    "--output-json",
    is_flag=True,
    help="Output deployment info as JSON",
)
@click.pass_context
def deploy_command(
    ctx: click.Context,
    job_id: str,
    provider_name: str,
    output_json: bool,
) -> None:
    """Deploy a trained model.

    Example:
        $ automlchain deploy --job-id abc123
    """
    click.echo(f"Deploying model from job {job_id}...")

    try:
        api_key = _get_api_key(provider_name)
        provider = ProviderRegistry.get(provider_name, api_key=api_key)

        # Deploy
        deployed = provider.deploy(job_id=job_id)

        click.echo(click.style("✓ Model deployed successfully!", fg="green"))
        click.echo(f"\n  Model ID:  {deployed.model_id}")
        click.echo(f"  Endpoint:  {deployed.endpoint}")
        click.echo(f"  Status:    {deployed.status}")

        if deployed.cost_per_1k_tokens:
            click.echo(f"  Cost:      ${deployed.cost_per_1k_tokens:.4f} / 1K tokens")

        if output_json:
            click.echo(json.dumps(deployed.to_dict(), indent=2))

    except Exception as e:
        click.echo(f"Error deploying model: {e}", err=True)
        sys.exit(1)


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
