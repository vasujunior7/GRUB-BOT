import pytest
from grubbot.eval import score_single

def test_score_single_correct():
    expected = {"name": "weather", "arguments": {"location": "Paris"}}
    pred = '{"name": "weather", "arguments": {"location": "Paris"}}'
    is_correct, error = score_single(pred, expected)
    assert is_correct is True
    assert error == ""

def test_score_single_wrong_tool():
    expected = {"name": "weather", "arguments": {"location": "Paris"}}
    pred = '{"name": "time", "arguments": {"location": "Paris"}}'
    is_correct, error = score_single(pred, expected)
    assert is_correct is False
    assert error == "wrong_tool"

def test_score_single_malformed():
    expected = {"name": "weather"}
    pred = '{"name: weather'
    is_correct, error = score_single(pred, expected)
    assert is_correct is False
    assert error == "malformed_json"

def test_score_single_missing_param():
    expected = {"name": "weather", "arguments": {"location": "Paris", "days": "3"}}
    pred = '{"name": "weather", "arguments": {"location": "Paris"}}'
    is_correct, error = score_single(pred, expected)
    assert is_correct is False
    assert error == "missing_param"
