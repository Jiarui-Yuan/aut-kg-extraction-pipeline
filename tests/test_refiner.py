from backend.extraction.refiner import (
    ActionLevelThreeEntities,
    Refiner,
    SegmentLevelThreeEntities,
)
from backend.extraction.resolver import ClassifiedObject, ObjectResolver
from backend.extraction.sequnecer import Sequencer
from backend.schemas.process_knowledge.entities import (
    Material,
    PPE,
    ProcessParameter,
    Tool,
)
from backend.schemas.process_knowledge.text import (
    Instruction,
    ObservedAction,
    ObservedObject,
    Segment,
)


class FakeExtractor:
    def __init__(
        self,
        classified_objects: list[ClassifiedObject] | None = None,
        segment_refinements: list[SegmentLevelThreeEntities] | None = None,
    ):
        self.classified_objects = classified_objects or []
        self.segment_refinements = iter(segment_refinements or [])

    def extract_list(self, **_kwargs: object) -> list[ClassifiedObject]:
        return self.classified_objects

    def extract(self, **_kwargs: object) -> SegmentLevelThreeEntities:
        return next(self.segment_refinements)


def test_refiner_links_entities_to_steps_by_action_reference() -> None:
    instruction = Instruction(
        id="instruction-1",
        name="Fasten the plate",
        segments=[
            Segment(
                id="segment-1",
                scene="workbench",
                actions=[
                    ObservedAction(
                        id="action-1",
                        start_time_ms=0,
                        end_time_ms=1000,
                        actor="operator",
                        action="Put on safety glasses",
                        object="safety glasses",
                    ),
                    ObservedAction(
                        id="action-2",
                        start_time_ms=1000,
                        end_time_ms=2000,
                        actor="operator",
                        action="Fasten the plate",
                        object="steel plate",
                        instrument="hammer",
                    ),
                ],
                objects=[
                    ObservedObject(name="hammer", object_type="tool"),
                    ObservedObject(
                        name="steel plate",
                        object_type="material",
                    ),
                    ObservedObject(
                        name="safety glasses",
                        object_type="ppe",
                    ),
                ],
                uncertainties=[],
            ),
        ],
    )
    sequencer = Sequencer(instruction)
    resolver = ObjectResolver(instruction)
    labels = [resolver.semantic_label(obj) for obj in resolver.objects]
    resolver.create_entities(
        extractor=FakeExtractor([
            ClassifiedObject(
                input_label=labels[0],
                entity_type="tool",
                tool=Tool(name="hammer"),
            ),
            ClassifiedObject(
                input_label=labels[1],
                entity_type="material",
                material=Material(name="steel plate"),
            ),
            ClassifiedObject(
                input_label=labels[2],
                entity_type="ppe",
                ppe=PPE(name="safety glasses"),
            ),
        ]),
    )

    process_parameter = ProcessParameter(
        name="impact force",
        parameter_type="force",
        nominal_value=20,
        unit="N",
    )
    refiner = Refiner(
        sequencer,
        resolver,
        extractor=FakeExtractor(segment_refinements=[
            SegmentLevelThreeEntities(actions=[
                ActionLevelThreeEntities(
                    action_id="action-2",
                    process_parameters=[process_parameter],
                ),
            ]),
        ]),
    )

    assert len(refiner.relations) == 3
    assert refiner.tool_requirements[0].step.step_number == 2
    assert refiner.tool_requirements[0].tool.name == "hammer"
    assert refiner.material_requirements[0].step.step_number == 2
    assert refiner.material_requirements[0].material.name == "steel plate"
    assert refiner.ppe_requirements[0].step.step_number == 1
    assert refiner.ppe_requirements[0].ppe.name == "safety glasses"
    assert refiner.process_parameters == [process_parameter]
    assert refiner.tool_requirements[0].parameters == [process_parameter]


def test_refiner_requires_created_object_entities() -> None:
    instruction = Instruction(
        id="instruction-1",
        name="Use a hammer",
        segments=[
            Segment(
                id="segment-1",
                scene="workbench",
                actions=[
                    ObservedAction(
                        id="action-1",
                        start_time_ms=0,
                        end_time_ms=1000,
                        actor="operator",
                        action="Strike",
                        instrument="hammer",
                    ),
                ],
                objects=[ObservedObject(name="hammer", object_type="tool")],
                uncertainties=[],
            ),
        ],
    )

    resolver = ObjectResolver(instruction)

    try:
        Refiner(
            Sequencer(instruction),
            resolver,
            extractor=FakeExtractor(),
        )
    except ValueError as error:
        assert "create_entities" in str(error)
    else:
        raise AssertionError("Refiner accepted unresolved object entities")
