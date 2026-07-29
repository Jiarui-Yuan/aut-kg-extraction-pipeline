"""Create step-level requirements from resolved instruction objects."""

from backend.extraction.resolver import ObjectEntity, ObjectResolver
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.entities import Material, PPE, Step, Tool
from backend.schemas.process_knowledge.relations import (
    MaterialRequirement,
    PPERequirement,
    ToolRequirement,
)

LevelThreeRelation = ToolRequirement | MaterialRequirement | PPERequirement


class Refiner:
    """Link resolved physical entities to the steps that reference them."""

    def __init__(
        self,
        sequencer: Sequencer,
        object_resolver: ObjectResolver,
    ):
        self.sequencer = sequencer
        self.object_resolver = object_resolver
        self.tool_requirements: list[ToolRequirement] = []
        self.material_requirements: list[MaterialRequirement] = []
        self.ppe_requirements: list[PPERequirement] = []
        self.relations: list[LevelThreeRelation] = []
        self._relations_by_action_id: dict[
            str,
            list[LevelThreeRelation],
        ] = {}
        self._create_relations(object_resolver.entity_bindings)

    def _create_relations(
        self,
        bindings: list[tuple[object, ObjectEntity]],
    ) -> None:
        if not bindings and self.object_resolver.objects:
            raise ValueError(
                "ObjectResolver.create_entities() must be called before "
                "initializing Refiner"
            )

        for observed_object, entity in bindings:
            action_ids = getattr(observed_object, "action_ids")
            for action_id in action_ids:
                step = self.sequencer.steps_by_action_id.get(action_id)
                if step is None:
                    raise ValueError(
                        f"Object references unknown action {action_id!r}"
                    )

                relation = self._requirement_for(step, entity)
                self.relations.append(relation)
                self._relations_by_action_id.setdefault(
                    action_id,
                    [],
                ).append(relation)
                if isinstance(relation, ToolRequirement):
                    self.tool_requirements.append(relation)
                elif isinstance(relation, MaterialRequirement):
                    self.material_requirements.append(relation)
                else:
                    self.ppe_requirements.append(relation)

    @staticmethod
    def _requirement_for(
        step: Step,
        entity: ObjectEntity,
    ) -> LevelThreeRelation:
        if isinstance(entity, Tool):
            return ToolRequirement(step=step, tool=entity)
        if isinstance(entity, Material):
            return MaterialRequirement(step=step, material=entity)
        if isinstance(entity, PPE):
            return PPERequirement(step=step, ppe=entity)
        raise TypeError(f"Unsupported resolved object entity: {type(entity)!r}")
