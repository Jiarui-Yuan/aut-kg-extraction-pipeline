"""
Evaluators — compare extracted vs. expected at different levels.

Level 1: Entity-level   — was the right entity found?
Level 2: Attribute-level — does the entity have the right field values?
Level 3: Relation-level  — are the right entities connected?
Level 4: Structure-level — is the nesting correct?

Each evaluator takes expected + extracted objects and returns PRF1 metrics.
"""

from typing import List, Type, Optional, Tuple, Sequence
from pydantic import BaseModel
from backend.evaluation.matching import compare_values, MatchStrategy, match_token_overlap
from backend.evaluation.metrics import PRF1, compute_prf1


# ============================================================================
# ENTITY-LEVEL EVAL
# ============================================================================

def evaluate_entity_detection(
    expected: Sequence[BaseModel],
    extracted: Sequence[BaseModel],
    name_field: str = "name",
    match_strategy: MatchStrategy = MatchStrategy.TOKEN_OVERLAP,
    match_threshold: float = 0.6,
) -> Tuple[PRF1, List[dict]]:
    """
    Entity-level evaluation: was each expected entity found in the extracted list?
    
    Matching is based on the name field using the specified strategy.
    
    Returns:
        Tuple of (PRF1 metrics, detailed match log)
    """
    matched_expected = set()
    matched_extracted = set()
    match_log = []
    
    # Try to match each expected entity to an extracted one
    for i, exp in enumerate(expected):
        exp_name = getattr(exp, name_field, "")
        best_score = 0.0
        best_match_idx = -1
        
        for j, ext in enumerate(extracted):
            if j in matched_extracted:
                continue
            ext_name = getattr(ext, name_field, "")
            
            comparison = compare_values(exp_name, ext_name, match_strategy, match_threshold)
            if comparison["score"] > best_score:
                best_score = comparison["score"]
                best_match_idx = j
        
        if best_match_idx >= 0 and best_score >= match_threshold:
            matched_expected.add(i)
            matched_extracted.add(best_match_idx)
            match_log.append({
                "status": "TP",
                "expected": exp_name,
                "extracted": getattr(extracted[best_match_idx], name_field, ""),
                "score": round(best_score, 3),
            })
        else:
            match_log.append({
                "status": "FN",
                "expected": exp_name,
                "extracted": None,
                "score": round(best_score, 3),
            })
    
    # Unmatched extracted entities are false positives
    for j, ext in enumerate(extracted):
        if j not in matched_extracted:
            match_log.append({
                "status": "FP",
                "expected": None,
                "extracted": getattr(ext, name_field, ""),
                "score": 0.0,
            })
    
    tp = len(matched_expected)
    fp = len(extracted) - len(matched_extracted)
    fn = len(expected) - len(matched_expected)
    
    entity_type = type(expected[0]).__name__ if expected else "Unknown"
    metrics = compute_prf1(entity_type, tp, fp, fn)
    
    return metrics, match_log


# ============================================================================
# ATTRIBUTE-LEVEL EVAL
# ============================================================================

def evaluate_attributes(
    expected: BaseModel,
    extracted: BaseModel,
    fields: Optional[List[str]] = None,
    match_strategy: MatchStrategy = MatchStrategy.TOKEN_OVERLAP,
    match_threshold: float = 0.6,
) -> Tuple[PRF1, List[dict]]:
    """
    Attribute-level evaluation: for a matched entity pair,
    how many attributes were correctly extracted?
    
    Args:
        expected: The gold-standard entity
        extracted: The extracted entity (already matched to expected)
        fields: Which fields to evaluate (default: all non-None expected fields)
        match_strategy: How to compare string values
        match_threshold: Minimum score to count as match
    
    Returns:
        Tuple of (PRF1 metrics, detailed field comparison log)
    """
    if fields is None:
        # Evaluate all fields that have a value in expected
        fields = [
            f for f in expected.model_fields 
            if f not in ("source_text", "description") and getattr(expected, f) is not None
        ]
    
    field_log = []
    tp = 0
    fp = 0
    fn = 0
    
    for field in fields:
        exp_val = getattr(expected, field, None)
        ext_val = getattr(extracted, field, None)
        
        comparison = compare_values(exp_val, ext_val, match_strategy, match_threshold)
        
        if exp_val is not None and comparison["match"]:
            tp += 1
            status = "TP"
        elif exp_val is not None and not comparison["match"]:
            fn += 1
            status = "FN" if ext_val is None else "WRONG"
        elif exp_val is None and ext_val is not None:
            fp += 1
            status = "FP"
        else:
            continue  # both None, skip
        
        field_log.append({
            "field": field,
            "status": status,
            "expected": exp_val,
            "extracted": ext_val,
            "score": comparison["score"],
        })
    
    entity_name = getattr(expected, "name", type(expected).__name__)
    metrics = compute_prf1(f"attrs:{entity_name}", tp, fp, fn)
    
    return metrics, field_log


# ============================================================================
# RELATION-LEVEL EVAL
# ============================================================================

def evaluate_relation_detection(
    expected: Sequence[BaseModel],
    extracted: Sequence[BaseModel],
    role_fields: List[str],
    match_strategy: MatchStrategy = MatchStrategy.TOKEN_OVERLAP,
    match_threshold: float = 0.6,
) -> Tuple[PRF1, List[dict]]:
    """
    Relation-level evaluation: are the right entities connected?
    
    A relation is considered matched if all its role players match.
    
    Args:
        expected: List of expected relations
        extracted: List of extracted relations
        role_fields: Fields that identify the role players, e.g. ["step", "tool"]
                     Each role player is matched by its "name" attribute.
        match_strategy: How to compare role player names
        match_threshold: Minimum score for role player match
    
    Returns:
        Tuple of (PRF1 metrics, detailed match log)
    """
    matched_extracted = set()
    match_log = []
    tp = 0
    fn = 0
    
    for exp in expected:
        best_match_idx = -1
        best_total_score = 0.0
        
        for j, ext in enumerate(extracted):
            if j in matched_extracted:
                continue
            
            # Check if all role players match
            role_scores = []
            for role_field in role_fields:
                exp_player = getattr(exp, role_field, None)
                ext_player = getattr(ext, role_field, None)
                
                if exp_player is None and ext_player is None:
                    role_scores.append(1.0)
                elif exp_player is None or ext_player is None:
                    role_scores.append(0.0)
                else:
                    exp_name = getattr(exp_player, "name", str(exp_player))
                    ext_name = getattr(ext_player, "name", str(ext_player))
                    comparison = compare_values(exp_name, ext_name, match_strategy, match_threshold)
                    role_scores.append(comparison["score"])
            
            total_score = sum(role_scores) / len(role_scores) if role_scores else 0.0
            if total_score > best_total_score:
                best_total_score = total_score
                best_match_idx = j
        
        if best_match_idx >= 0 and best_total_score >= match_threshold:
            tp += 1
            matched_extracted.add(best_match_idx)
            match_log.append({
                "status": "TP",
                "expected": {rf: getattr(getattr(exp, rf, None), "name", None) for rf in role_fields},
                "extracted": {rf: getattr(getattr(extracted[best_match_idx], rf, None), "name", None) for rf in role_fields},
                "score": round(best_total_score, 3),
            })
        else:
            fn += 1
            match_log.append({
                "status": "FN",
                "expected": {rf: getattr(getattr(exp, rf, None), "name", None) for rf in role_fields},
                "extracted": None,
                "score": round(best_total_score, 3),
            })
    
    fp = len(extracted) - len(matched_extracted)
    for j, ext in enumerate(extracted):
        if j not in matched_extracted:
            match_log.append({
                "status": "FP",
                "expected": None,
                "extracted": {rf: getattr(getattr(ext, rf, None), "name", None) for rf in role_fields},
                "score": 0.0,
            })
    
    rel_type = type(expected[0]).__name__ if expected else "Unknown"
    metrics = compute_prf1(rel_type, tp, fp, fn)
    
    return metrics, match_log


# ============================================================================
# COMBINED LEVEL EVAL
# ============================================================================

def evaluate_extraction_level(
    level: int,
    expected_entities: Sequence[BaseModel],
    extracted_entities: Sequence[BaseModel],
    expected_relations: Optional[Sequence[BaseModel]] = None,
    extracted_relations: Optional[Sequence[BaseModel]] = None,
    relation_role_fields: Optional[List[str]] = None,
    match_strategy: MatchStrategy = MatchStrategy.TOKEN_OVERLAP,
    match_threshold: float = 0.6,
) -> dict:
    """
    Run full evaluation for one complexity level.
    
    Returns a dict with entity metrics, attribute metrics (per matched pair),
    and optionally relation metrics.
    """
    results: dict[str, object] = {"level": level}
    
    # Entity detection
    entity_metrics, entity_log = evaluate_entity_detection(
        expected_entities, extracted_entities,
        match_strategy=match_strategy, match_threshold=match_threshold
    )
    results["entity_detection"] = entity_metrics
    results["entity_log"] = entity_log
    
    # Attribute evaluation for matched pairs
    attr_metrics_list = []
    for entry in entity_log:
        if entry["status"] == "TP":
            # Find the matched pair
            exp_match = next(
                (e for e in expected_entities if getattr(e, "name", "") == entry["expected"]), 
                None
            )
            ext_match = next(
                (e for e in extracted_entities if getattr(e, "name", "") == entry["extracted"]), 
                None
            )
            if exp_match and ext_match:
                attr_metrics, attr_log = evaluate_attributes(
                    exp_match, ext_match,
                    match_strategy=match_strategy, match_threshold=match_threshold
                )
                attr_metrics_list.append({"metrics": attr_metrics, "log": attr_log})
    
    results["attribute_evaluations"] = attr_metrics_list
    
    # Relation detection (if applicable)
    if expected_relations and extracted_relations and relation_role_fields:
        rel_metrics, rel_log = evaluate_relation_detection(
            expected_relations, extracted_relations,
            role_fields=relation_role_fields,
            match_strategy=match_strategy, match_threshold=match_threshold
        )
        results["relation_detection"] = rel_metrics
        results["relation_log"] = rel_log
    
    return results