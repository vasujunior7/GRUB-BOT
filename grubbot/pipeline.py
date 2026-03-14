import os
import json
from loguru import logger
from typing import Optional

from .config import load_tools, load_goal_from_markdown, GrubbotConfig
from .providers import get_provider
from .datagen import generate_examples, split_and_save
from .finetune import load_model, prepare_dataset, train, save_checkpoint
from .eval import evaluate, EvalResult
from .cluster import embed_failures, cluster_failures
from .loop import generate_targeted_data

def run_full_pipeline(tools_path: str, goal_path: str, model_name: str, provider_name: str = "gemini"):
    # Stage 0: Config Load
    logger.info("Loading tools and goal configuration...")
    tools = load_tools(tools_path)
    goal = load_goal_from_markdown(goal_path)
    config = GrubbotConfig(tools=tools, goal=goal, model_name=model_name, provider=provider_name)
    provider = get_provider(provider_name)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/failures", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    train_path = "data/train.jsonl"
    eval_path = "data/eval.jsonl"
    
    # Stage 1: Data Generation (if train.jsonl doesn't exist to prevent full rebuilds randomly)
    if not os.path.exists(train_path):
        logger.info("Stage 1 - Generating base synthetic data...")
        examples = generate_examples(tools, goal, provider, count_per_tool=30)
        logger.info(f"Generated {len(examples)} examples.")
        split_and_save(examples, train_path, eval_path)
    else:
        logger.info("Base synthetic data exists, skipping Stage 1 initial generation.")
        
    iteration = 1
    best_accuracy = 0.0
    current_model_path = model_name
    
    clusters_resolved = []
    
    run_log = {
        "iterations": [],
        "best_accuracy": best_accuracy
    }
        
    while iteration <= config.goal.max_iterations:
        logger.info(f"--- Starting Iteration {iteration} / {config.goal.max_iterations} ---")
        out_dir = f"models/grubbot-{model_name.replace('/', '-')}-v{iteration}"
        
        # Stage 2: Finetuning
        logger.info(f"Stage 2 - Finetuning ({current_model_path})...")
        model, tokenizer = load_model(current_model_path)
        dataset = prepare_dataset(train_path, tokenizer)
        trainer = train(model, tokenizer, dataset, out_dir)
        save_checkpoint(model, tokenizer, out_dir)
        current_model_path = out_dir # Next loop (if any) starts from here
        
        # Stage 3: Evaluation
        logger.info("Stage 3 - Evaluating model...")
        eval_result = evaluate(model, tokenizer, eval_path, tools)
        
        logger.info(f"Overall Accuracy: {eval_result.overall_accuracy * 100:.1f}%")
        for t_name, acc in eval_result.per_tool_accuracy.items():
            logger.info(f"  {t_name}: {acc * 100:.1f}%")
            
        iter_log = {
            "iteration": iteration,
            "overall_accuracy": eval_result.overall_accuracy,
            "per_tool_accuracy": eval_result.per_tool_accuracy,
            "failure_count": len(eval_result.failures)
        }
        run_log["iterations"].append(iter_log)
        
        if eval_result.overall_accuracy > best_accuracy:
            best_accuracy = eval_result.overall_accuracy
            run_log["best_accuracy"] = best_accuracy

        # Check target
        if eval_result.overall_accuracy >= config.goal.target_accuracy:
            logger.info(f"Target accuracy reached: {eval_result.overall_accuracy * 100:.1f}% >= {config.goal.target_accuracy * 100:.1f}%")
            break
            
        if iteration >= config.goal.max_iterations:
            logger.info("Max iterations reached. Stopping loop.")
            break
            
        if not eval_result.failures:
            logger.info("No failures detected, stopping.")
            break
            
        # Stage 4: Custom Loops (Clustering + Re-generation)
        logger.info("Target not reached. Stage 4 - Clustering failures...")
        
        # Save fail logs
        with open(f"data/failures/iter_{iteration}.json", "w") as f:
            f.write(json.dumps([fw.model_dump() for fw in eval_result.failures], indent=2))
        
        embeddings = embed_failures(eval_result.failures)
        clusters = cluster_failures(eval_result.failures, embeddings)
        
        logger.info(f"Found {len(clusters)} distinct failure patterns.")
        
        new_examples_count = 0
        with open(train_path, "a", encoding="utf-8") as f:
            for cluster in clusters:
                logger.info(f"Generating data for cluster: {cluster.label} (Size: {cluster.size})")
                clusters_resolved.append(cluster.label)
                targeted_data = generate_targeted_data(cluster, tools, provider, target_count=min(20, cluster.size * 3))
                for idx, ex in enumerate(targeted_data):
                    ex["id"] = f"train_iter{iteration}_{cluster.cluster_id}_{idx}"
                    f.write(json.dumps(ex) + "\n")
                    new_examples_count += 1
                    
        logger.info(f"Appended {new_examples_count} targeted examples to train.jsonl")
        iteration += 1

    # End Pipeline
    logger.info("=== Pipeline Complete ===")
    logger.info(f"Final Best Accuracy: {best_accuracy * 100:.1f}%")
    logger.info(f"Total Iterations: {iteration}")
    
    with open(f"runs/run_{len(os.listdir('runs')) + 1}.json", "w") as f:
        json.dump(run_log, f, indent=2)


def run_datagen_only(tools_path: str, goal_path: str, provider_name: str = "gemini"):
    """Run only Stage 1 to generate synthetic training data."""
    logger.info("Loading tools and goal configuration for datagen...")
    tools = load_tools(tools_path)
    goal = load_goal_from_markdown(goal_path)
    provider = get_provider(provider_name)
    
    os.makedirs("data", exist_ok=True)
    train_path = "data/train.jsonl"
    eval_path = "data/eval.jsonl"
    
    logger.info(f"Stage 1 - Generating base synthetic data using {provider_name}...")
    examples = generate_examples(tools, goal, provider, count_per_tool=30)
    logger.info(f"Generated {len(examples)} examples.")
    split_and_save(examples, train_path, eval_path)
    logger.info(f"Data saved to {train_path} and {eval_path}.")


def run_eval_only(model_path: str, eval_path: str, tools_path: str) -> EvalResult:
    tools = load_tools(tools_path)
    model, tokenizer = load_model(model_path)
    return evaluate(model, tokenizer, eval_path, tools)
