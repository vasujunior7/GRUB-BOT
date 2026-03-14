import yaml
from pydantic import BaseModel, Field
from typing import Dict, Any, List

class ToolParameter(BaseModel):
    type: str
    description: str
    required: bool = True

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ToolParameter]

class GoalConfig(BaseModel):
    target_accuracy: float = Field(ge=0.0, le=1.0)
    max_iterations: int = Field(gt=0)
    priorities: List[str] = []

class GrubbotConfig(BaseModel):
    tools: List[ToolDefinition]
    goal: GoalConfig
    model_name: str
    provider: str = "gemini"

def load_tools(path: str) -> List[ToolDefinition]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "tools" not in data:
        raise ValueError(f"Invalid tools file format in {path}: Missing 'tools' key.")
    
    tools = []
    for t_data in data["tools"]:
        # Handle dict formats appropriately if any parsing needs converting
        params = t_data.get("parameters", {})
        parsed_params = {}
        for k, v in params.items():
            if isinstance(v, str):
                parsed_params[k] = ToolParameter(type=v, description="", required=True)
            else:
                parsed_params[k] = ToolParameter(**v)
        
        tools.append(ToolDefinition(
            name=t_data["name"],
            description=t_data.get("description", ""),
            parameters=parsed_params
        ))
    return tools

def load_goal_from_markdown(path: str) -> GoalConfig:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Parse markdown simply
    lines = content.splitlines()
    target_accuracy = 0.9
    max_iterations = 5
    priorities = []
    
    for line in lines:
        lower_line = line.lower()
        if "target:" in lower_line:
            # Extract target percentage (e.g. 90%+)
            import re
            match = re.search(r"(\d+)%", lower_line)
            if match:
                target_accuracy = float(match.group(1)) / 100.0
        elif "max iterations:" in lower_line:
            import re
            match = re.search(r"(\d+)", lower_line)
            if match:
                max_iterations = int(match.group(1))
        elif "priority:" in lower_line:
            parts = line.split(":", 1)
            if len(parts) > 1:
                priorities.append(parts[1].strip())
                
    return GoalConfig(target_accuracy=target_accuracy, max_iterations=max_iterations, priorities=priorities)
