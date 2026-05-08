"""
Process Knowledge Relations — from binary to deeply nested n-ary.

Complexity levels for extraction experiments:
  Level 2: Binary relations (StepOrder, ProcedureHasStep)
  Level 3: Relations with own attributes (ToolRequirement, MaterialRequirement)
  Level 4: N-ary relations with 3-4 roles (StepExecution, QualityCheck)
  Level 5: Deeply nested structures (ProcedureExecution with StepExecutions)

In LadybugDB/Neo4j: n-ary relations are reified as intermediate nodes.
In TypeDB: these map directly to relations with roles.

Reference: Carriero et al. (2025) — PKO
"""

from typing import Optional, List, Literal
from datetime import datetime
from backend.schemas.base import KGRelation
from backend.schemas.process_knowledge.entities import (
    Tool, Material, PPE, Worker, Step, Procedure,
    ProcessParameter, QualityRequirement
)


# ============================================================================
# LEVEL 2 — Binary relations
# ============================================================================

class StepOrder(KGRelation):
    """Sequential ordering between two steps. Maps to pko:nextStep."""
    before: Step
    after: Step


class ProcedureHasStep(KGRelation):
    """A procedure contains a step. Maps to pko:hasStep."""
    procedure: Procedure
    step: Step


# ============================================================================
# LEVEL 3 — Relations with own attributes
# ============================================================================

class ToolRequirement(KGRelation):
    """A step requires a specific tool, with optional configuration and parameters."""
    step: Step
    tool: Tool
    is_mandatory: bool = True
    configuration_notes: Optional[str] = None
    parameters: Optional[List[ProcessParameter]] = None


class MaterialRequirement(KGRelation):
    """A step requires a specific material."""
    step: Step
    material: Material
    quantity: Optional[str] = None
    preparation_notes: Optional[str] = None


class PPERequirement(KGRelation):
    """A step requires specific PPE. Maps to pko:requiresPPE."""
    step: Step
    ppe: PPE
    is_mandatory: bool = True
    reason: Optional[str] = None


# ============================================================================
# LEVEL 4 — N-ary relations with 3-4 roles
# ============================================================================

class StepExecution(KGRelation):
    """
    A concrete execution of a step by a worker.
    Maps to pko:StepExecution ⊑ prov:Activity.
    This is the EXECUTION (what actually happened).
    """
    step: Step
    executed_by: Worker
    tools_used: Optional[List[Tool]] = None
    materials_used: Optional[List[Material]] = None

    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    actual_duration_seconds: Optional[int] = None

    status: Optional[Literal[
        "completed", "failed", "skipped", "partially_completed", "interrupted"
    ]] = None

    observed_parameters: Optional[List[ProcessParameter]] = None
    deviation_from_spec: Optional[str] = None


class QualityCheck(KGRelation):
    """
    An inspection performed on a step execution.
    N-ary: inspector × step_execution × requirement → result
    """
    inspector: Worker
    step_execution: StepExecution
    requirement: QualityRequirement

    result: Optional[Literal["pass", "fail", "conditional", "not_inspected"]] = None
    findings: Optional[str] = None
    corrective_action_required: bool = False
    corrective_action_description: Optional[str] = None
    inspected_at: Optional[datetime] = None


class QualificationRecord(KGRelation):
    """
    A worker holds a qualification for a specific procedure domain.
    N-ary: worker × expertise_level × domain × certifying_body × validity
    """
    worker: Worker
    expertise_level: Literal["apprentice", "skilled", "expert", "master"]
    procedure_domain: str
    certifying_body: Optional[str] = None
    certification_norm: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    certificate_id: Optional[str] = None


# ============================================================================
# LEVEL 5 — Deeply nested structures
# ============================================================================

class StepFeedback(KGRelation):
    """
    Feedback reported during a step execution.
    Maps to pko:askedQuestion / pko:reportedIssue.
    """
    step_execution: StepExecution
    reported_by: Worker
    feedback_type: Literal["question", "issue", "suggestion", "observation"]
    content: str
    severity: Optional[Literal["info", "warning", "critical"]] = None
    resolved: bool = False
    resolution: Optional[str] = None
    resolved_by: Optional[Worker] = None


class ProcedureExecution(KGRelation):
    """
    A complete execution of a procedure — the top-level execution record.
    Maps to pko:ProcedureExecution.
    
    Most complex extraction target: full procedure execution with nested
    step executions, quality checks, and feedback.
    """
    procedure: Procedure
    executed_by: List[Worker]
    supervised_by: Optional[Worker] = None

    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    step_executions: Optional[List[StepExecution]] = None
    quality_checks: Optional[List[QualityCheck]] = None
    feedback: Optional[List[StepFeedback]] = None

    overall_status: Optional[Literal[
        "completed", "completed_with_deviations", "failed", "aborted"
    ]] = None
    overall_result: Optional[Literal[
        "accepted", "accepted_with_conditions", "rejected", "pending_review"
    ]] = None
    procedure_version: Optional[str] = None