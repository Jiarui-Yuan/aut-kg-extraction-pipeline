"""
Process Knowledge Entities — oriented at PKO (Procedural Knowledge Ontology).

Complexity levels for extraction experiments:
  Level 1: Flat entities (Tool, Material, PPE, Worker)
  Level 2: Entities with enums and constrained fields (Step, Procedure)
  Level 3: Entities with nested optional structures (ProcessParameter, QualityRequirement)

Reference: Carriero et al. (2025) — Procedural Knowledge Ontology (PKO)
"""

from typing import Optional, List, Literal
from backend.schemas.base import KGEntity


# ============================================================================
# LEVEL 1 — Flat entities
# ============================================================================

class Tool(KGEntity):
    """A tool or instrument used during a process step. Maps to dcat:Resource in PKO."""
    tool_type: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None


class Material(KGEntity):
    """A material or consumable used in a process."""
    material_type: Optional[str] = None
    specification: Optional[str] = None
    dimensions: Optional[str] = None


class PPE(KGEntity):
    """Personal Protective Equipment. Maps to pko:requiresPPE."""
    ppe_type: Optional[str] = None
    standard: Optional[str] = None


# ============================================================================
# LEVEL 2 — Entities with enums and constrained fields
# ============================================================================

class Worker(KGEntity):
    """A person performing or supervising a process. Maps to prov:Agent + pro:RoleInTime."""
    role: Optional[Literal[
        "operator", "supervisor", "quality_inspector",
        "maintenance", "trainee", "welding_engineer"
    ]] = None
    expertise_level: Optional[Literal[
        "apprentice", "skilled", "expert", "master"
    ]] = None
    certifications: Optional[List[str]] = None


class Step(KGEntity):
    """
    A single step within a procedure specification. Maps to pplan:Step.
    This is the SPECIFICATION, not the execution.
    """
    step_number: Optional[int] = None
    expected_duration_seconds: Optional[int] = None
    is_critical: bool = False
    step_type: Optional[Literal[
        "preparation", "execution", "inspection",
        "cleanup", "documentation", "safety_check"
    ]] = None


class Procedure(KGEntity):
    """
    A complete procedure specification. Maps to pko:Procedure ⊑ pplan:Plan.
    Example: "MIG welding procedure for S235 T-joint"
    """
    procedure_id: Optional[str] = None
    version: Optional[str] = None
    status: Optional[Literal[
        "draft", "approved", "active", "deprecated", "archived"
    ]] = None
    domain: Optional[Literal[
        "welding", "assembly", "inspection", "maintenance",
        "surface_treatment", "bending", "laser_cutting"
    ]] = None
    applicable_norms: Optional[List[str]] = None


# ============================================================================
# LEVEL 3 — Entities with nested optional structures
# ============================================================================

class ProcessParameter(KGEntity):
    """A measurable parameter within a process step (current, voltage, speed, etc.)."""
    parameter_type: Optional[Literal[
        "temperature", "pressure", "speed", "current",
        "voltage", "flow_rate", "force", "time", "distance"
    ]] = None
    unit: Optional[str] = None
    nominal_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class QualityRequirement(KGEntity):
    """A quality criterion that must be met at a step or procedure level."""
    requirement_type: Optional[Literal[
        "dimensional", "visual", "destructive_test",
        "non_destructive_test", "documentation", "traceability"
    ]] = None
    acceptance_criteria: Optional[str] = None
    inspection_method: Optional[str] = None
    applicable_norm: Optional[str] = None