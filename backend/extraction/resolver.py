from typing import Literal, TypeAlias

from pydantic import BaseModel

from backend.schemas.process_knowledge.entities import Material, PPE, Tool


class EquivalenceGroup(BaseModel):
    """Input values that refer to the same semantic entity."""

    canonical_entity: str
    entities: list[str]


class SemanticResolution(BaseModel):
    """LLM response containing a partition of all supplied entity values."""

    groups: list[EquivalenceGroup]


ObjectEntity: TypeAlias = Tool | Material | PPE


class ClassifiedObject(BaseModel):
    """A resolved object classified as one supported knowledge entity."""

    input_label: str
    entity_type: Literal["tool", "material", "ppe"]
    tool: Tool | None = None
    material: Material | None = None
    ppe: PPE | None = None
