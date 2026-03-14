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

@cli.command()
@click.option('--tools', type=click.Path(exists=True), required=True, help="Path to tools.yaml")
@click.option('--goal', type=click.Path(exists=True), required=True, help="Path to goal.md")
@click.option('--provider', default="gemini", help="LLM Provider to use (gemini, groq, ollama)")
@click.option('--count', default=50, help="Number of examples per tool")
def datagen(tools, goal, provider, count):
    """Run data generation (Stage 1) only."""
    from grubbot.config import load_tools, load_goal_from_markdown
    from grubbot.providers import get_provider
    from grubbot.datagen import generate_examples, split_and_save
    import os
    
    click.echo(f"Starting Data Generation with provider {provider}...")
    tool_defs = load_tools(tools)
    goal_config = load_goal_from_markdown(goal)
    llm = get_provider(provider)
    
    click.echo(f"Loaded {len(tool_defs)} tools.")
    examples = generate_examples(tool_defs, goal_config, llm, count_per_tool=count)
    
    click.echo(f"Generated {len(examples)} total examples. Splitting into train and eval...")
    
    os.makedirs("data", exist_ok=True)
    split_and_save(examples, "data/train.jsonl", "data/eval.jsonl")
    click.echo("Done! Wrote realistic data to data/train.jsonl and data/eval.jsonl")

if __name__ == '__main__':
    cli()
