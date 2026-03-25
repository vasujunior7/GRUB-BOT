import json
import random
from typing import List, Dict, Any
from .config import ToolDefinition, GoalConfig
from .providers.base import BaseProvider

def build_datagen_prompt(tool: ToolDefinition, count: int) -> str:
    """Builds a prompt asking the LLM to generate tool-use examples based on the tool definition."""
    prompt = f"""
You are tasked with generating {count} diverse user queries that should trigger the following tool:

Tool Name: {tool.name}
Description: {tool.description}

Parameters:
"""
    for p_name, p_def in tool.parameters.items():
        req = "required" if p_def.required else "optional"
        prompt += f"- {p_name} ({p_def.type}, {req}): {p_def.description}\n"

    prompt += """
Please generate the output as a valid JSON array of objects.
Each object MUST have:
1. "user_query": A realistic user message.
2. "expected_tool_call": A dictionary having the "name" of the tool, and "arguments" containing the exact extracted parameters from the user's query.

Make the examples extremely diverse to ensure robust model performance:
- **Standard requests:** Clear, direct queries.
- **Conversational boilerplate:** Queries wrapped in conversational text (e.g., "Hey, can you please...", "I was wondering if...").
- **Missing optional parameters:** Queries where optional parameters are deliberately left out.
- **Negative examples:** Queries that are related but should NOT trigger this tool. For these, `expected_tool_call` should be `null`. (Generate about 10-15% of these).
- **Ambiguous queries:** Queries that might be slightly confusing or could potentially be misinterpreted, testing the model's ability to differentiate.
- **Complex phrasing:** Queries that use indirect language or require more inference to understand.
- **Typographical errors:** Queries with common spelling mistakes.

Output strictly valid JSON (an array of objects) and nothing else.
"""
    return prompt

def generate_examples(tools: List[ToolDefinition], goal: GoalConfig, provider: BaseProvider, count_per_tool: int = 200) -> List[Dict[str, Any]]:
    all_examples = []
    
    # We serialize the full tools schema to include it in every example's "tools" field
    tools_schema = []
    for t in tools:
        props = {}
        required = []
        for p_name, p_def in t.parameters.items():
            props[p_name] = {"type": p_def.type, "description": p_def.description}
            if p_def.required:
                required.append(p_name)
        
        tools_schema.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required
                }
            }
        })
    
    for tool in tools:
        prompt = build_datagen_prompt(tool, count_per_tool)
        system_instruction = "You are a synthetic training data generator. Generate JSON responses only, without formatting blocks."
        
        raw_response = provider.generate(prompt=prompt, system=system_instruction)
        
        # Clean up Markdown JSON blocks if LLM still formats it
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
            
        try:
            generated_items = json.loads(raw_response.strip())
            for item in generated_items:
                # Format to ChatML with tool calls
                formatted_example = {
                    "tools": tools_schema,
                    "messages": [
                        {"role": "user", "content": item["user_query"]}
                    ],
                    "expected_tool_call": item["expected_tool_call"]
                }
                all_examples.append(formatted_example)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON for tool {tool.name}: {e}")
            print(f"Raw response was: {raw_response[:200]}...")
            
    return all_examples

def split_and_save(examples: List[Dict[str, Any]], train_path: str, eval_path: str, split_ratio: float = 0.8):
    random.shuffle(examples)
    split_idx = int(len(examples) * split_ratio)
    
    train_data = examples[:split_idx]
    eval_data = examples[split_idx:]
    
    # Ensure dirs exist
    import os
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    
    with open(train_path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(train_data):
            ex["id"] = f"train_{idx}"
            f.write(json.dumps(ex) + "\n")
            
    with open(eval_path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(eval_data):
            ex["id"] = f"eval_{idx}"
            f.write(json.dumps(ex) + "\n")
