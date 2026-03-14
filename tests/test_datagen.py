import pytest
import os
import json
from grubbot.config import ToolDefinition, ToolParameter, GoalConfig
from grubbot.datagen import build_datagen_prompt, split_and_save

def test_build_datagen_prompt():
    tool = ToolDefinition(
        name="weather",
        description="Get weather for a location",
        parameters={
            "location": ToolParameter(type="string", description="City name", required=True)
        }
    )
    prompt = build_datagen_prompt(tool, count=5)
    assert "weather" in prompt
    assert "Get weather for a location" in prompt
    assert "location (string, required)" in prompt
    assert "City name" in prompt
    assert "5" in prompt

def test_split_and_save(tmp_path):
    examples = [{"data": i} for i in range(10)]
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    
    split_and_save(examples, str(train_path), str(eval_path), split_ratio=0.8)
    
    assert train_path.exists()
    assert eval_path.exists()
    
    train_lines = train_path.read_text().strip().split('\n')
    eval_lines = eval_path.read_text().strip().split('\n')
    
    assert len(train_lines) == 8
    assert len(eval_lines) == 2
    
    loaded_train = json.loads(train_lines[0])
    assert "id" in loaded_train
    assert loaded_train["id"].startswith("train_")
