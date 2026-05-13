import os
import json
from loguru import logger
from typing import Optional

from .config import load_tools, load_goal, GrubbotConfig
from .providers import get_provider
from .datagen import generate_examples, split_and_save, get_example_hash
from .finetune import load_base_model, load_from_checkpoint, prepare_dataset, train, save_checkpoint
from .eval import evaluate, EvalResult
from .cluster import embed_failures, cluster_failures
from .loop import generate_targeted_data

def run_full_pipeline(tools_path: str, goal_path: str, model_name: str, provider_name: str = "gemini"):
    # Stage 0: Config Load
    logger.info("Loading tools and goal configuration...")
    tools = load_tools(tools_path)
    goal = load_goal(goal_path)
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
    run_id = len(os.listdir('runs')) + 1
    
    # Stage 1.5: Baseline Eval
    logger.info("--- Stage 1.5: Baseline Evaluation (Iteration 0) ---")
    baseline_result = run_eval_only(model_name, eval_path, tools_path)
    logger.info(f"Baseline Accuracy: {baseline_result.overall_accuracy * 100:.1f}%")
    best_accuracy = baseline_result.overall_accuracy
    best_checkpoint = model_name
    current_model_path = model_name
    
    clusters_resolved = []
    
    run_log = {
        "run_id": run_id,
        "iterations": [{
            "iteration": 0,
            "overall_accuracy": baseline_result.overall_accuracy,
            "per_tool_accuracy": baseline_result.per_tool_accuracy,
            "failure_count": len(baseline_result.failures)
        }],
        "best_accuracy": best_accuracy,
        "best_checkpoint": best_checkpoint
    }
        
    while iteration <= config.goal.max_iterations:
        logger.info(f"--- Starting Iteration {iteration} / {config.goal.max_iterations} ---")
        out_dir = f"models/grubbot-{model_name.replace('/', '-')}-v{iteration}"
        
        # Stage 2: Finetuning
        logger.info(f"Stage 2 - Finetuning ({current_model_path})...")
        if iteration == 1:
            model, tokenizer = load_base_model(model_name)
        else:
            model, tokenizer = load_from_checkpoint(model_name, current_model_path)
            
        dataset = prepare_dataset(train_path, tokenizer)
        trainer = train(model, tokenizer, dataset, out_dir, iteration=iteration)
        
        # P1-8: Persist training loss curves
        loss_history = [log for log in trainer.state.log_history if "loss" in log]
        with open(f"runs/run_{run_id}_iter_{iteration}_loss.json", "w") as f:
            json.dump(loss_history, f, indent=2)
            
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
            best_checkpoint = out_dir
            run_log["best_accuracy"] = best_accuracy
            run_log["best_checkpoint"] = best_checkpoint

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
        
        # P1-6: Deduplication
        existing_hashes = set()
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    user_query = ex["messages"][0]["content"]
                    expected_tc = ex["expected_tool_call"]
                    existing_hashes.add(get_example_hash(user_query, expected_tc))
                except:
                    pass
        
        new_examples_count = 0
        with open(train_path, "a", encoding="utf-8") as f:
            for cluster in clusters:
                logger.info(f"Generating data for cluster: {cluster.label} (Size: {cluster.size})")
                clusters_resolved.append(cluster.label)
                targeted_data = generate_targeted_data(cluster, tools, provider, target_count=min(20, cluster.size * 3), existing_hashes=existing_hashes)
                for idx, ex in enumerate(targeted_data):
                    ex["id"] = f"train_iter{iteration}_{cluster.cluster_id}_{idx}"
                    f.write(json.dumps(ex) + "\n")
                    new_examples_count += 1
                    
        logger.info(f"Appended {new_examples_count} targeted examples to train.jsonl")
        
        # P3-4: Dataset Size Cap
        max_dataset_size = 5000
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > max_dataset_size:
            logger.info(f"Dataset exceeds cap ({len(lines)} > {max_dataset_size}). Trimming oldest examples...")
            lines = lines[-max_dataset_size:]
            with open(train_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
                
        # P0-5: VRAM cleanup
        import gc
        import torch
        del model, tokenizer, trainer, dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        iteration += 1

    # End Pipeline
    logger.info("=== Pipeline Complete ===")
    logger.info(f"Final Best Accuracy: {best_accuracy * 100:.1f}%")
    logger.info(f"Best Checkpoint Saved At: {best_checkpoint}")
    logger.info(f"Total Iterations: {iteration}")
    
    with open(f"runs/run_{run_id}.json", "w") as f:
        json.dump(run_log, f, indent=2)


def run_datagen_only(tools_path: str, goal_path: str, provider_name: str = "gemini"):
    """Run only Stage 1 to generate synthetic training data."""
    logger.info("Loading tools and goal configuration for datagen...")
    tools = load_tools(tools_path)
    goal = load_goal(goal_path)
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tools = load_tools(tools_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    load_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    return evaluate(model, tokenizer, eval_path, tools)
