<p align="center">
  <img src="assets/logo.png" alt="Grubbot Logo" width="600">
</p>

<h1 align="center">Grubbot</h1>

<p align="center">
  <strong>Dig until it works.</strong><br>
  <em>An autonomous pipeline that takes your custom tool definitions and produces a small, reliable, locally-finetuned model to use them flawlessly.</em>
</p>

<p align="center">
  <a href="#the-problem">The Problem</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#the-four-stages">The 4 Stages</a> 
</p>

---

## 🛑 The Problem
Every developer who wants a local AI assistant hits the same wall: **small models are unreliable with custom tools**. They hallucinate parameters, call the wrong tool, or format responses incorrectly. 

Existing solutions finetune once, evaluate once, and stop. If the model fails on 20% of your tool calls, you are stuck manually curating more data. **No clean solution exists for this specific problem. Grubbot is that solution.**

## ✨ Features
- **Starts from your tools:** Defined in YAML. Not generic function-calling benchmarks.
- **Autonomous Retrain Loop:** Tests, finds failures, clusters them by pattern, patches them, and re-trains until the target accuracy is hit.
- **Fully Free after Setup:** Uses free APIs for data generation (Gemini Flash, Groq, or local Ollama). Everything after runs locally for free.
- **Private & Local:** Your tools, your data, your GPU, your finetuned model.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Clone the repository and install in editable mode
pip install -e .
```

### 2. Configure Environment
Copy the example environment variables and add your preferred provider credentials:
```bash
cp .env.example .env
```
*(Grubbot uses **Gemini 2.0 Flash** or **Groq Llama 3.3 70b** for high-quality initial data generation)*

### 3. Define Tools & Goals
Write your tool definitions (`tools.yaml`) and what accuracy you want to reach (`goal.md`). See the `examples/` directory!

### 4. Run the Pipeline!

**Just want to test data generation? (No GPU required)**
```bash
grubbot datagen --tools examples/tools.yaml --goal examples/goal.md --provider gemini --count 50
```

**Ready to Finetune & Loop? (NVIDIA GPU Required)**
```bash
grubbot run --tools examples/tools.yaml --goal examples/goal.md --model unsloth/Qwen2.5-3B-bnb-4bit
```

---

## 🧬 The Four Stages

Grubbot operates in a linear but looping 4-stage pipeline:

1. **🟡 Stage 1: Data Generation** — Calls a free LLM to synthesize hundreds of permutations (positive, edge-case, and negative queries) from your `tools.yaml` schema.
2. **🟣 Stage 2: Finetuning** — Wraps **Unsloth** and **TRL** to efficiently train a LoRA adapter natively on your GPU.
3. **🟠 Stage 3: Evaluation** — Evaluates the finetuned model locally against the hold-out test set (`eval.jsonl`) scoring exactly on parameters and JSON integrity.
4. **🔴 Stage 4: Failure Clustering & Auto-Loop** — Uses **sentence-transformers** & **HDBSCAN** to embed failures, group them by logical pattern (e.g., "Hallucinated unit parameter"), generate highly-targeted patch data, and loop back to Stage 2.

> **Research vs Workflow:** The loop between Stage 2 ↔ Stage 4 is the core innovation, applying "auto-research" principles to dataset iteration.

---

## 🛠 Tech Stack

| Layer | Tool | Role |
|---|---|---|
| **Data Generation** | `litellm` | Single interface to Gemini, Groq, or Ollama |
| **Finetuning** | `unsloth` + `trl` | Blazing fast LoRA adapter training |
| **Failure Clustering**| `sentence-transformers` + `hdbscan` | Dense embeddings + density-based clustering |
| **Evaluation** | `python` | Strict parsing and assertion scoring |
| **CLI App** | `click` | Entrypoint orchestration |

<p align="center"><i>built for developers who want models that actually listen.</i></p>
