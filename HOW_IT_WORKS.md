# GRUBBOT: Autonomous Finetuning Pipeline - How It Works

Grubbot is a self-improving, autonomous fine-tuning loop designed to teach local Small Language Models (SLMs) how to use complex tools accurately.

Instead of writing thousands of training examples by hand, you provide Grubbot with a goal (in a YAML configuration) and an API key for a "teacher" model (like Gemini Flash, Groq, or OpenAI). Grubbot handles the rest: generating data, training your model locally, evaluating its mistakes, and focusing the next round of training on its weaknesses.

## 🚀 The User Flow

1. **Configuration**: You define your goal, tools, and constraints in `config.yaml` (or pass it via CLI).
2. **Start the Loop**: You run `python -m grubbot.cli pipeline --config config.yaml`.
3. **Wait & Monitor**: Grubbot starts the autonomous loop. You can monitor the progress in your console. The pipeline will iteratively refine the model until it hits your target accuracy threshold or the max iterations.
4. **Done**: Once complete, the fully trained LoRA adapters are saved in `models/finetuned_model_final/`.

---

## ⚙️ Technical Architecture (Step-by-Step)

### 1. Initial Data Generation & Validation (Stage 1)
- Grubbot asks the "Teacher" model to generate synthetic training examples demonstrating how to use the tools.
- **Robustness**: Calls to the Teacher API use **Exponential Backoff Retry Logic** to handle rate limits and network errors.
- **Validation**: Generated data is strictly validated against your tool schemas. Hallucinated or malformed JSON is instantly discarded.
- **Deduplication**: Examples are cryptographically hashed. If an example is already in the dataset, it is skipped. This prevents over-representing common scenarios and ensures maximum dataset diversity.

### 2. Baseline Evaluation (Iteration 0)
- Before any training begins, Grubbot loads your base model (e.g., Llama-3-8B-Instruct) and runs a test benchmark.
- This creates a **Baseline Evaluation**, providing a starting accuracy score to measure subsequent improvements against.

### 3. Iterative Fine-Tuning (Stage 2)
- Grubbot trains the model using **Unsloth** and **TRL (SFTTrainer)** via LoRA (Low-Rank Adaptation) on your local GPU.
- **VRAM Management**: Between iterations, Grubbot aggressively clears GPU memory, deletes old model instances, and forces Python garbage collection to prevent Out-Of-Memory (OOM) crashes.
- **Learning Rate Decay**: As iterations progress, the learning rate shrinks ($LR_{base} / \sqrt{iteration}$) to prevent catastrophic forgetting and stabilize convergence.
- **Checkpointing**: Adapter weights are safely saved and reloaded for incremental training.

### 4. Semantic Evaluation (Stage 3)
- The newly trained model is tested against a fresh batch of evaluation prompts.
- **Semantic Scoring**: Instead of strict exact-string matching, Grubbot uses `difflib.SequenceMatcher` to evaluate arguments. If the model outputs `"new york city"` instead of `"New York"`, it receives partial credit rather than an immediate fail.
- The pipeline measures accuracy. If it hits your `target_accuracy`, the loop exits successfully!

### 5. Failure Clustering & Augmentation (Stage 4)
- If the model failed test cases, Grubbot extracts the failed prompts.
- Using `sentence-transformers` and `HDBSCAN`, it clusters these failures semantically to find common patterns in the model's blind spots.
- Grubbot asks the Teacher API to generate *highly targeted* synthetic data focusing specifically on these failure clusters.
- **Dataset Size Cap**: The new examples are appended to the dataset. To prevent unbounded memory growth, the dataset is hard-capped at 5,000 examples (trimming the oldest first).

### 6. The Loop Repeats
- The pipeline returns to Stage 2, fine-tuning the model again with the newly augmented dataset, heavily weighting the model's past mistakes. This continues until the target accuracy is met!
