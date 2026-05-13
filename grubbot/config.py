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

def load_goal(path: str) -> GoalConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    if not data or "goal" not in data:
        # Fallback for old markdown style gracefully, but encourage yaml
        return GoalConfig(target_accuracy=0.9, max_iterations=5, priorities=[])
        
    g_data = data["goal"]
    return GoalConfig(
        target_accuracy=g_data.get("target_accuracy", 0.9),
        max_iterations=g_data.get("max_iterations", 5),
        priorities=g_data.get("priorities", [])
    )
