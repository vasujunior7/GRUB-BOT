import json
import random
from typing import List, Dict, Any
from pydantic import BaseModel
from loguru import logger

from .config import GrubbotConfig, ToolDefinition
from .cluster import FailureCluster
from .providers.base import BaseProvider
from .datagen import build_datagen_prompt

class LoopResult(BaseModel):
    iterations: int
    final_accuracy: float
    per_tool_accuracy: Dict[str, float]
    clusters_resolved: List[str]

def generate_targeted_data(cluster: FailureCluster, tools: List[ToolDefinition], provider: BaseProvider, target_count: int = 15) -> List[Dict[str, Any]]:
    # Format a prompt specifically highlighting the failure
    examples_str = "\n".join([f"Query: {e.user_query} | Expected: {json.dumps(e.expected)} | Model did: {e.predicted}" for e in cluster.examples[:3]])
    
    prompt = f"""
We have a local text generation model failing on specific tool use cases. 
The failure pattern is categorized as: {cluster.label}

Here are some examples of what it got wrong:
{examples_str}

Please generate {target_count} diverse NEW user queries and their correct expected tool_calls (JSON) that address this exact failure pattern. Focus on corner cases where the model might get confused.

Output ONLY a JSON array of objects, with each object structured exactly like:
{{
  "user_query": "...",
  "expected_tool_call": {{"name": "...", "arguments": {{...}}}}
}}
"""
    system_instruction = "You are an expert synthetic data generator fixing AI failure cases. Output raw JSON arrays."
    
    raw_response = provider.generate(prompt, system=system_instruction)
    
    # Simple JSON extraction
    if "```json" in raw_response:
        raw_response = raw_response.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_response:
        raw_response = raw_response.split("```")[1].split("```")[0].strip()
        
    all_examples = []
        
    # Tools schema again
    tools_schema = []
    for t in tools:
        props = {}
        for p_name, p_def in t.parameters.items():
            props[p_name] = {"type": p_def.type, "description": p_def.description}
        required = [k for k, v in t.parameters.items() if v.required]
        tools_schema.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {"type": "object", "properties": props, "required": required}
            }
        })
        
    try:
        data = json.loads(raw_response)
        for item in data:
             all_examples.append({
                "tools": tools_schema,
                "messages": [{"role": "user", "content": item["user_query"]}],
                "expected_tool_call": item["expected_tool_call"]
            })
    except Exception as e:
        logger.error(f"Failed to parse targeted examples for cluster {cluster.label}: {e}")
        
    return all_examples

def run_loop(config: GrubbotConfig) -> LoopResult:
    """Normally orchestrates the retrain loop between eval, cluster, data-aug, and finetune."""
    pass
