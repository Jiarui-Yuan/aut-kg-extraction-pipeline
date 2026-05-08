"""
Metrics computation — Precision, Recall, F1 at various levels.

These are the core numbers for the paper.
"""

from typing import List, Optional
from pydantic import BaseModel


class PRF1(BaseModel):
    """Precision / Recall / F1 for one category."""
    category: str
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_expected: int
    total_extracted: int


def compute_prf1(
    category: str,
    true_positives: int,
    false_positives: int, 
    false_negatives: int,
) -> PRF1:
    """Compute P/R/F1 from TP, FP, FN counts."""
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return PRF1(
        category=category,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        total_expected=true_positives + false_negatives,
        total_extracted=true_positives + false_positives,
    )


def aggregate_prf1(results: List[PRF1], category: str = "overall") -> PRF1:
    """Aggregate multiple PRF1 results (micro-averaging)."""
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)
    
    return compute_prf1(category, total_tp, total_fp, total_fn)


def format_prf1(result: PRF1) -> str:
    """Pretty-print a PRF1 result."""
    return (
        f"{result.category:30s} | "
        f"P={result.precision:.3f} R={result.recall:.3f} F1={result.f1:.3f} | "
        f"TP={result.true_positives} FP={result.false_positives} FN={result.false_negatives}"
    )


def format_prf1_table(results: List[PRF1], title: str = "Evaluation Results") -> str:
    """Format a list of PRF1 results as a readable table."""
    lines = [
        f"\n{'=' * 80}",
        f"  {title}",
        f"{'=' * 80}",
        f"{'Category':30s} | {'P':>6s} {'R':>6s} {'F1':>6s} | {'TP':>3s} {'FP':>3s} {'FN':>3s}",
        f"{'-' * 80}",
    ]
    for r in results:
        lines.append(
            f"{r.category:30s} | "
            f"{r.precision:6.3f} {r.recall:6.3f} {r.f1:6.3f} | "
            f"{r.true_positives:3d} {r.false_positives:3d} {r.false_negatives:3d}"
        )
    lines.append(f"{'=' * 80}")
    return "\n".join(lines)