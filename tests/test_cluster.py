import pytest
import numpy as np
from grubbot.cluster import embed_failures, cluster_failures
from grubbot.eval import FailedExample

def test_embed_and_cluster():
    failures = [
        FailedExample(id="1", user_query="What's the weather in Paris?", expected={"name": "weather"}, predicted='{"name":"time"}', error_type="wrong_tool"),
        FailedExample(id="2", user_query="Weather for London", expected={"name": "weather"}, predicted='{"name":"time"}', error_type="wrong_tool"),
        FailedExample(id="3", user_query="Is it sunny?", expected={"name": "weather"}, predicted='{"name":"time"}', error_type="wrong_tool"),
        FailedExample(id="4", user_query="Get 5 days weather", expected={"name": "weather"}, predicted='{"name":"time"}', error_type="wrong_tool"),
        FailedExample(id="5", user_query="Turn on lights", expected={"name": "lights"}, predicted='{"name":"weather"}', error_type="wrong_tool"),
        FailedExample(id="6", user_query="Lights in kitchen", expected={"name": "lights"}, predicted='{"name":"weather"}', error_type="wrong_tool"),
    ]
    
    embeddings = embed_failures(failures)
    assert embeddings.shape == (6, 384) # MiniLM embedding size
    
    clusters = cluster_failures(failures, embeddings)
    assert len(clusters) >= 1
    assert sum(c.size for c in clusters) == len(failures)
