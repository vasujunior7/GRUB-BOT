import json
import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from .config import ToolDefinition

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

    # Evaluate exact parameter matches
    for k, v in exp_args.items():
        if k not in pred_args:
            return False, "missing_param"
        if str(pred_args[k]).lower() != str(v).lower():
             return False, "wrong_param_value"

    # Evaluate hallucinated parameters
    for k in pred_args.keys():
        if k not in exp_args:
            return False, "hallucinated_param"

    return True, ""

def evaluate(model, tokenizer, eval_path: str, tools: List[ToolDefinition]) -> EvalResult:
    # Model is passed if using UNsloth inference directly
    # For simplicity of standalone eval in this iteration, we mock inference or require generation logic
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    
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
        
        # Format the inference prompt
        conversation = [
            {"role": "user", "content": user_query}
        ]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
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
