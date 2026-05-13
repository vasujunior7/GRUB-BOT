import json
import re
import difflib
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from .config import ToolDefinition

def is_semantically_similar(a: str, b: str, threshold: float = 0.8) -> bool:
    if a.lower() == b.lower(): 
        return True
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

class FailedExample(BaseModel):
    id: str
    user_query: str
    expected: Dict[str, Any]
    predicted: str
    error_type: str  # e.g., 'wrong_tool', 'missing_param', 'malformed_json'

class EvalResult(BaseModel):
    overall_accuracy: float
    per_tool_accuracy: Dict[str, float]
    failures: List[FailedExample]

def score_single(prediction: str, expected: Dict[str, Any]) -> Tuple[bool, str]:
    """Score a single prediction string against the expected JSON dict.
    Returns (is_correct, error_type_if_any)
    """
    try:
        # Simple extraction if model wraps in markdown
        if "```json" in prediction:
            prediction = prediction.split("```json")[1].split("```")[0].strip()
        elif "```" in prediction:
            prediction = prediction.split("```")[1].split("```")[0].strip()

        pred_json = json.loads(prediction)
    except json.JSONDecodeError:
        return False, "malformed_json"

    if "name" not in pred_json or pred_json["name"] != expected["name"]:
        return False, "wrong_tool"

    pred_args = pred_json.get("arguments", {})
    exp_args = expected.get("arguments", {})

    # Evaluate exact/semantic parameter matches
    for k, v in exp_args.items():
        if k not in pred_args:
            return False, "missing_param"
        
        val_pred = pred_args[k]
        val_exp = v
        
        if isinstance(val_exp, str) and isinstance(val_pred, str):
            if not is_semantically_similar(val_pred, val_exp):
                return False, "wrong_param_value"
        else:
            if str(val_pred).lower() != str(val_exp).lower():
                 return False, "wrong_param_value"

    # Evaluate hallucinated parameters
    for k in pred_args.keys():
        if k not in exp_args:
            return False, "hallucinated_param"

    return True, ""

def evaluate(model, tokenizer, eval_path: str, tools: List[ToolDefinition]) -> EvalResult:
    # Use standard PyTorch inference mode
    model.eval()
    
    with open(eval_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    correct = 0
    total = len(lines)
    tool_counts = {t.name: {"total": 0, "correct": 0} for t in tools}
    failures = []
    
    for line in lines:
        data = json.loads(line)
        user_query = data["messages"][0]["content"]
        expected = data["expected_tool_call"]
        tool_name = expected.get("name")
        
        if tool_name not in tool_counts:
            tool_counts[tool_name] = {"total": 0, "correct": 0}
        tool_counts[tool_name]["total"] += 1
        
        # Format the inference prompt with tool definitions
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
            
        tools_json = json.dumps(tools_schema, indent=2)
        system_prompt = f"You are a helpful assistant with access to the following tools:\n{tools_json}\n\nWhen the user's request matches a tool, respond with a JSON object containing 'name' and 'arguments'. If no tool matches, respond with 'null'."
        
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        import torch
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                use_cache=True,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1
            )
        # Decode only the new output tokens
        pred_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        is_correct, error_type = score_single(pred_text, expected)
        
        if is_correct:
            correct += 1
            tool_counts[tool_name]["correct"] += 1
        else:
            failures.append(FailedExample(
                id=data.get("id", "unknown"),
                user_query=user_query,
                expected=expected,
                predicted=pred_text,
                error_type=error_type
            ))
            
    per_tool_acc = {}
    for t_name, counts in tool_counts.items():
        if counts["total"] > 0:
            per_tool_acc[t_name] = counts["correct"] / counts["total"]
        else:
            per_tool_acc[t_name] = 0.0
            
    return EvalResult(
        overall_accuracy=correct / total if total > 0 else 0.0,
        per_tool_accuracy=per_tool_acc,
        failures=failures
    )