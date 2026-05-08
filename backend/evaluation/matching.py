"""
Matching strategies — how to compare extracted vs. expected values.

Each strategy returns a score between 0.0 (no match) and 1.0 (perfect match).
Different strategies are appropriate for different field types:
  - Exact: IDs, enums, booleans
  - Normalized: names, labels
  - Token overlap: descriptions, free text
  - Embedding similarity: semantically equivalent but differently worded
"""

from typing import Optional
from enum import Enum


class MatchStrategy(str, Enum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    TOKEN_OVERLAP = "token_overlap"
    EMBEDDING = "embedding"


def match_exact(expected: str, extracted: str) -> float:
    """Exact string match. Returns 1.0 or 0.0."""
    return 1.0 if expected == extracted else 0.0


def match_normalized(expected: str, extracted: str) -> float:
    """Case-insensitive, whitespace-normalized match."""
    def normalize(s: str) -> str:
        return " ".join(s.lower().strip().split())
    return 1.0 if normalize(expected) == normalize(extracted) else 0.0


def match_token_overlap(expected: str, extracted: str) -> float:
    """
    Proportion of expected tokens found in extracted string.
    
    "Fronius TPS 400i" vs "MIG-Schweißgerät Fronius TPS 400i"
    → 3 of 3 expected tokens found → 1.0
    
    "Fronius TPS 400i" vs "Fronius Gerät"
    → 1 of 3 expected tokens found → 0.33
    """
    expected_tokens = set(expected.lower().split())
    extracted_tokens = set(extracted.lower().split())
    
    if not expected_tokens:
        return 1.0 if not extracted_tokens else 0.0
    
    overlap = expected_tokens & extracted_tokens
    return len(overlap) / len(expected_tokens)


def match_embedding(expected: str, extracted: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Cosine similarity between sentence embeddings.
    Requires sentence-transformers: uv add sentence-transformers
    
    Returns similarity score between 0.0 and 1.0.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer(model_name)
        embeddings = model.encode([expected, extracted])
        
        cosine_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(max(0.0, cosine_sim))
    except ImportError:
        print("⚠️ sentence-transformers not installed. Run: uv add sentence-transformers")
        return 0.0


def compare_values(
    expected, 
    extracted, 
    strategy: MatchStrategy = MatchStrategy.NORMALIZED,
    threshold: float = 0.8,
) -> dict:
    """
    Compare two values and return match details.
    
    Handles None, numeric, bool, and string comparisons.
    
    Returns:
        {
            "match": True/False (above threshold),
            "score": 0.0-1.0,
            "strategy": "normalized",
            "expected": ...,
            "extracted": ...
        }
    """
    # Both None → match
    if expected is None and extracted is None:
        return {"match": True, "score": 1.0, "strategy": strategy.value,
                "expected": None, "extracted": None}
    
    # One None → no match
    if expected is None or extracted is None:
        return {"match": False, "score": 0.0, "strategy": strategy.value,
                "expected": expected, "extracted": extracted}
    
    # Booleans
    if isinstance(expected, bool):
        score = 1.0 if expected == extracted else 0.0
        return {"match": score >= threshold, "score": score, "strategy": "exact",
                "expected": expected, "extracted": extracted}
    
    # Numbers
    if isinstance(expected, (int, float)):
        if isinstance(extracted, (int, float)):
            if expected == 0:
                score = 1.0 if extracted == 0 else 0.0
            else:
                score = max(0.0, 1.0 - abs(expected - extracted) / abs(expected))
        else:
            score = 0.0
        return {"match": score >= threshold, "score": score, "strategy": "numeric",
                "expected": expected, "extracted": extracted}
    
    # Strings
    expected_str = str(expected)
    extracted_str = str(extracted)
    
    if strategy == MatchStrategy.EXACT:
        score = match_exact(expected_str, extracted_str)
    elif strategy == MatchStrategy.NORMALIZED:
        score = match_normalized(expected_str, extracted_str)
    elif strategy == MatchStrategy.TOKEN_OVERLAP:
        score = match_token_overlap(expected_str, extracted_str)
    elif strategy == MatchStrategy.EMBEDDING:
        score = match_embedding(expected_str, extracted_str)
    else:
        score = match_exact(expected_str, extracted_str)
    
    return {"match": score >= threshold, "score": score, "strategy": strategy.value,
            "expected": expected_str, "extracted": extracted_str}