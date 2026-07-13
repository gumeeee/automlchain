"""Train command for the CLI."""

from __future__ import annotations

import sys

import click

from ...core import AutoMLPipeline
from ...training.callbacks import ProgressCallback


@click.command("train")
@click.option(
    "--dataset",
    "-d",
    "dataset_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to training dataset (JSONL, CSV, or Parquet)",
)
@click.option(
    "--provider",
    "-p",
    "provider_name",
    default="replicate",
    type=click.Choice(["replicate", "mock"]),
    help="Fine-tuning provider to use",
)
@click.option(
    "--model",
    "-m",
    "model_name",
    default="meta/llama-3-8b-instruct",
    help="Model to fine-tune",
)
@click.option(
    "--template",
    "-t",
    "template_string",
    default=None,
    help="Prompt template (e.g., 'Classify: {input}\\nCategory:')",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=3,
    help="Number of training epochs",
)
@click.option(
    "--learning-rate",
    "-l",
    type=float,
    default=1e-4,
    help="Learning rate",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=4,
    help="Batch size",
)
@click.option(
    "--lora-rank",
    type=int,
    default=16,
    help="LoRA rank dimension",
)
@click.option(
    "--output-json",
    is_flag=True,
    help="Output job info as JSON",
)
@click.option(
    "--wait/--no-wait",
    default=False,
    help="Wait for training to complete",
)
@click.pass_context
def train_command(
    ctx: click.Context,
    dataset_path: str,
    provider_name: str,
    model_name: str,
    template_string: str | None,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    lora_rank: int,
    output_json: bool,
    wait: bool,
) -> None:
    """Start a fine-tuning training job.

    Example:
        $ automlchain train --dataset data.jsonl --model meta/llama-3-8b
    """
    click.echo("Starting training job...")
    click.echo(f"  Dataset: {dataset_path}")
    click.echo(f"  Provider: {provider_name}")
    click.echo(f"  Model: {model_name}")

    try:
        # Initialize pipeline
        pipeline = AutoMLPipeline(
            provider=provider_name,
            hyperparameters={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "lora_rank": lora_rank,
            },
        )

        # Add progress callback
        progress_callback = ProgressCallback()
        pipeline.training_orchestrator.add_callback(progress_callback)

        # Upload dataset
        click.echo("\n[1/3] Uploading dataset...")
        dataset = pipeline.upload_dataset(dataset_path)
        click.echo(f"      Uploaded {len(dataset)} samples")

        # Create template if provided
        if template_string:
            click.echo("\n[2/3] Creating prompt template...")
            template = pipeline.create_template(template_string, name="cli-template")
            click.echo(f"      Template: {template_string[:50]}...")
        else:
            template = None
            click.echo("\n[2/3] No template provided, using default format")

        # Start training
        click.echo("\n[3/3] Starting training...")
        job = pipeline.train(
            dataset=dataset,
            template=template,
            model=model_name,
        )

        click.echo("\nTraining job started!")
        click.echo(f"  Job ID: {job.job_id}")

        if output_json:
            import json
            click.echo(json.dumps(job.to_dict(), indent=2))

        if wait:
            click.echo("\nWaiting for training to complete...")
            from ...core.exceptions import AutoMLChainTimeoutError

            try:
                result = pipeline.training_orchestrator.wait_for_completion(
                    job.job_id,
                    timeout=7200,  # 2 hours
                )
                click.echo("\nTraining completed!")
                click.echo(f"  Duration: {result.duration_seconds:.1f}s")
                if result.cost:
                    click.echo(f"  Cost: ${result.cost:.4f}")

                if output_json:
                    click.echo(json.dumps(result.to_dict(), indent=2))

            except AutoMLChainTimeoutError:
                click.echo("Training timed out. Use 'automlchain status' to check progress.")
                sys.exit(1)

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
