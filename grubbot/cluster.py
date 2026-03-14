import json
import numpy as np
from typing import List, Dict
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import hdbscan

from .eval import FailedExample

class FailureCluster(BaseModel):
    cluster_id: int
    label: str
    examples: List[FailedExample]
    size: int

def embed_failures(failures: List[FailedExample]) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts_to_embed = []
    for f in failures:
        text = f"Expected tool: {f.expected.get('name', 'unknown')} | " \
               f"Expected params: {json.dumps(f.expected.get('arguments', {}))} | " \
               f"Predicted output: {f.predicted} | " \
               f"Error type: {f.error_type} | " \
               f"User query: {f.user_query}"
        texts_to_embed.append(text)
        
    embeddings = model.encode(texts_to_embed)
    return embeddings

def cluster_failures(failures: List[FailedExample], embeddings: np.ndarray) -> List[FailureCluster]:
    if len(failures) < 5:
        # Not enough data for meaningful HDBSCAN, fallback to single generic cluster
        return [FailureCluster(
            cluster_id=0,
            label="general_failures",
            examples=failures,
            size=len(failures)
        )]
        
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, min(5, len(failures) // 2)))
    cluster_labels = clusterer.fit_predict(embeddings)
    
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(failures[i])
        
    result = []
    for c_id, examples in clusters.items():
        # Ideally, we'd call an LLM here to label the cluster dynamically based on a sub-sample of examples.
        # For default implementation, we use a generic placeholder label indicating its top error type.
        dominant_error = max(set([e.error_type for e in examples]), key=[e.error_type for e in examples].count)
        label_name = f"Cluster_{c_id}_{dominant_error}" if c_id != -1 else "noise_outliers"
        
        result.append(FailureCluster(
            cluster_id=int(c_id),
            label=label_name,
            examples=examples,
            size=len(examples)
        ))
        
    return result
