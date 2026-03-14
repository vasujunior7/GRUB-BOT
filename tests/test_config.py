import pytest
import os
import yaml
from grubbot.config import load_tools, load_goal_from_markdown, ToolDefinition, GoalConfig

def test_load_tools(tmp_path):
    tools_yaml = """
tools:
  - name: test_tool
    description: A test tool
    parameters:
      arg1:
        type: string
        description: Description 1
        required: true
      arg2:
        type: integer
        description: Description 2
        required: false
"""
    file_path = tmp_path / "tools.yaml"
    file_path.write_text(tools_yaml)

    tools = load_tools(str(file_path))
    assert len(tools) == 1
    assert tools[0].name == "test_tool"
    assert tools[0].description == "A test tool"
    assert "arg1" in tools[0].parameters
    assert tools[0].parameters["arg1"].type == "string"
    assert tools[0].parameters["arg1"].required is True
    assert tools[0].parameters["arg2"].required is False

def test_load_goal(tmp_path):
    goal_md = """
# Goal
Target: 95%+ accuracy on tool selection.
Max iterations: 3
Priority: never hallucinate parameters.
"""
    file_path = tmp_path / "goal.md"
    file_path.write_text(goal_md)

    goal = load_goal_from_markdown(str(file_path))
    assert goal.target_accuracy == 0.95
    assert goal.max_iterations == 3
    assert len(goal.priorities) == 1
    assert goal.priorities[0] == "never hallucinate parameters."
