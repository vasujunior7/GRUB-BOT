# Grubbot

Grubbot is a four-stage autonomous pipeline that takes your custom tool definitions and produces a small, reliable, local model that can use them correctly.

## Installation

```bash
pip install -e .
```

Copy `.env.example` to `.env` and fill in your API keys.

## Usage

```bash
grubbot run --tools tools.yaml --goal goal.md --model qwen2.5-3b
```
