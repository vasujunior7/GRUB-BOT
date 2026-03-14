import click
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

@click.group()
def cli():
    """Grubbot: Autonomous loop for tool-use finetuning."""
    pass

@cli.command()
@click.option('--tools', type=click.Path(exists=True), required=True, help="Path to tools.yaml")
@click.option('--goal', type=click.Path(exists=True), required=True, help="Path to goal.md")
@click.option('--model', required=True, help="HuggingFace model to fine-tune")
def run(tools, goal, model):
    """Run the complete 4-stage pipeline."""
    click.echo(f"Starting Grubbot pipeline with model {model}...")
    from grubbot.pipeline import run_full_pipeline
    run_full_pipeline(tools, goal, model)

@cli.command()
@click.option('--model', type=click.Path(exists=True), required=True, help="Path to finetuned model")
@click.option('--data', type=click.Path(exists=True), required=True, help="Path to eval.jsonl")
def eval(model, data):
    """Run evaluation only."""
    click.echo(f"Evaluating model {model} on {data}...")
    pass

@cli.command()
@click.option('--tools', type=click.Path(exists=True), required=True, help="Path to tools.yaml")
@click.option('--goal', type=click.Path(exists=True), required=True, help="Path to goal.md")
@click.option('--model', type=click.Path(exists=True), required=True, help="Path to current model checkpoint")
def loop(tools, goal, model):
    """Resume the retrain loop from a checkpoint."""
    click.echo(f"Resuming loop with model {model}...")
    pass

if __name__ == '__main__':
    cli()
