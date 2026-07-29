import pytest

from backend.extraction.resolver import (
    ActorResolver,
    ClassifiedObject,
    ObjectResolver,
    SemanticResolution,
)
from backend.schemas.process_knowledge.entities import Material, PPE, Tool, Worker
from backend.schemas.process_knowledge.text import (
    Instruction,
    ObservedAction,
    ObservedActor,
    ObservedObject,
    Segment,
)


def make_segment(segment_id: str, actors: list[str]) -> Segment:
    return Segment(
        id=segment_id,
        scene="workbench",
        actions=[
            ObservedAction(
                id=f"{segment_id}-action-{index}",
                start_time_ms=index,
                end_time_ms=index + 1,
                actor=actor,
                action="work",
            )
            for index, actor in enumerate(actors)
        ],
        objects=[],
        uncertainties=[],
    )


def test_resolver_handles_instruction_actors() -> None:
    instruction = Instruction(
        id="instruction-1",
        name="Test instruction",
        segments=[
            make_segment("segment-1", ["worker", "supervisor"]),
            make_segment("segment-2", ["worker", "inspector"]),
        ],
    )

    resolver = ActorResolver(instruction)

    assert resolver.actors == [
        ObservedActor(
            name="worker",
            action_ids=["segment-1-action-0", "segment-2-action-0"],
        ),
        ObservedActor(
            name="supervisor",
            action_ids=["segment-1-action-1"],
        ),
        ObservedActor(
            name="inspector",
            action_ids=["segment-2-action-1"],
        ),
    ]


def test_resolve_adds_an_instruction_to_existing_actors() -> None:
    resolver = ActorResolver()
    resolver.add(ObservedActor(name="worker", action_ids=["existing-action"]))
    instruction = Instruction(
        id="instruction-1",
        name="Test instruction",
        segments=[make_segment("segment-1", ["worker", "supervisor"])],
    )

    resolved = resolver.resolve(instruction)

    assert resolved == [
        ObservedActor(
            name="worker",
            action_ids=["existing-action", "segment-1-action-0"],
        ),
        ObservedActor(
            name="supervisor",
            action_ids=["segment-1-action-1"],
        ),
    ]


class FakeExtractor:
    def __init__(
        self,
        resolution: SemanticResolution | None = None,
        workers: list[Worker] | None = None,
        classified_objects: list[ClassifiedObject] | None = None,
    ):
        self.resolution = resolution
        self.workers = workers
        self.classified_objects = classified_objects
        self.system_prompts: list[str] = []

    def extract(self, **kwargs: object) -> SemanticResolution:
        self.system_prompts.append(str(kwargs["system_prompt"]))
        assert self.resolution is not None
        return self.resolution

    def extract_list(
        self,
        **kwargs: object,
    ) -> list[Worker] | list[ClassifiedObject]:
        self.system_prompts.append(str(kwargs["system_prompt"]))
        if kwargs["item_model"] is Worker:
            assert self.workers is not None
            return self.workers
        assert self.classified_objects is not None
        return self.classified_objects


def test_semantic_deduplicate_groups_equivalent_actor_labels() -> None:
    resolver = ActorResolver()
    actors = [
        ObservedActor(name="worker", action_ids=["action-1"]),
        ObservedActor(name="operator", action_ids=["action-2"]),
        ObservedActor(name="supervisor", action_ids=["action-3"]),
    ]
    extractor = FakeExtractor(SemanticResolution.model_validate({
        "groups": [
            {
                "canonical_entity": "worker",
                "entities": ["worker", "operator"],
            },
            {
                "canonical_entity": "supervisor",
                "entities": ["supervisor"],
            },
        ],
    }))

    result = resolver.semantic_deduplicate(actors, extractor=extractor)

    assert result == [
        ObservedActor(
            name="worker",
            action_ids=["action-1", "action-2"],
        ),
        ObservedActor(name="supervisor", action_ids=["action-3"]),
    ]


def test_semantic_deduplicate_rejects_missing_actors() -> None:
    resolver = ActorResolver()
    extractor = FakeExtractor(SemanticResolution.model_validate({
        "groups": [
            {"canonical_entity": "worker", "entities": ["worker"]},
        ],
    }))

    with pytest.raises(ValueError, match="every input entity exactly once"):
        resolver.semantic_deduplicate(
            [
                ObservedActor(name="worker"),
                ObservedActor(name="supervisor"),
            ],
            extractor=extractor,
        )


def test_object_resolver_handles_instruction_objects() -> None:
    shared_object = ObservedObject(name="hammer", object_type="tool")
    instruction = Instruction(
        id="instruction-1",
        name="Test instruction",
        segments=[
            Segment(
                id="segment-1",
                scene="workbench",
                actions=[],
                objects=[
                    shared_object,
                    ObservedObject(name="nail", object_type="fastener"),
                ],
                uncertainties=[],
            ),
            Segment(
                id="segment-2",
                scene="workbench",
                actions=[],
                objects=[
                    ObservedObject(name="hammer", object_type="tool"),
                ],
                uncertainties=[],
            ),
        ],
    )

    resolver = ObjectResolver(instruction)

    assert resolver.objects == [
        shared_object,
        ObservedObject(name="nail", object_type="fastener"),
    ]


def test_object_resolver_semantically_deduplicates_objects() -> None:
    hammer = ObservedObject(name="hammer", object_type="tool")
    mallet = ObservedObject(name="mallet", object_type="tool")
    resolver = ObjectResolver()
    labels = [
        resolver.semantic_label(hammer),
        resolver.semantic_label(mallet),
    ]
    extractor = FakeExtractor(SemanticResolution.model_validate({
        "groups": [
            {
                "canonical_entity": labels[0],
                "entities": labels,
            },
        ],
    }))

    result = resolver.semantic_deduplicate(
        [hammer, mallet],
        extractor=extractor,
    )

    assert result == [hammer]


def test_actor_resolver_creates_enriched_workers_in_actor_order() -> None:
    resolver = ActorResolver()
    extractor = FakeExtractor(workers=[
        Worker(name="supervisor", role="supervisor"),
        Worker(name="operator", role="operator"),
    ])

    workers = resolver.create_workers(
        [
            ObservedActor(name="operator"),
            ObservedActor(name="supervisor"),
        ],
        extractor=extractor,
    )

    assert workers == [
        Worker(name="operator", role="operator"),
        Worker(name="supervisor", role="supervisor"),
    ]


def test_actor_resolver_rejects_missing_workers() -> None:
    resolver = ActorResolver()
    extractor = FakeExtractor(workers=[Worker(name="operator")])

    with pytest.raises(ValueError, match="every actor exactly once"):
        resolver.create_workers(
            [
                ObservedActor(name="operator"),
                ObservedActor(name="supervisor"),
            ],
            extractor=extractor,
        )


def test_object_resolver_creates_tool_material_and_ppe() -> None:
    resolver = ObjectResolver()
    objects = [
        ObservedObject(name="hammer", object_type="hand tool"),
        ObservedObject(name="steel plate", object_type="raw material"),
        ObservedObject(name="safety glasses", object_type="protective equipment"),
    ]
    labels = [resolver.semantic_label(obj) for obj in objects]
    extractor = FakeExtractor(classified_objects=[
        ClassifiedObject(
            input_label=labels[2],
            entity_type="ppe",
            ppe=PPE(name="safety glasses", ppe_type="eye protection"),
        ),
        ClassifiedObject(
            input_label=labels[0],
            entity_type="tool",
            tool=Tool(name="hammer", tool_type="hand tool"),
        ),
        ClassifiedObject(
            input_label=labels[1],
            entity_type="material",
            material=Material(
                name="steel plate",
                material_type="raw material",
            ),
        ),
    ])

    entities = resolver.create_entities(objects, extractor=extractor)

    assert entities == [
        Tool(name="hammer", tool_type="hand tool"),
        Material(name="steel plate", material_type="raw material"),
        PPE(name="safety glasses", ppe_type="eye protection"),
    ]


def test_object_resolver_rejects_mismatched_entity_field() -> None:
    resolver = ObjectResolver()
    observed_object = ObservedObject(name="hammer", object_type="tool")
    label = resolver.semantic_label(observed_object)
    extractor = FakeExtractor(classified_objects=[
        ClassifiedObject(
            input_label=label,
            entity_type="tool",
            material=Material(name="hammer"),
        ),
    ])

    with pytest.raises(ValueError, match="selected by entity_type"):
        resolver.create_entities([observed_object], extractor=extractor)


def test_instruction_metadata_is_added_to_every_prompt() -> None:
    instruction = Instruction(
        id="instruction-1",
        name="Build the frame",
        segments=[make_segment("segment-1", ["operator"])],
    )
    resolver = ActorResolver(instruction)
    extractor = FakeExtractor(workers=[Worker(name="operator")])

    resolver.create_workers(extractor=extractor)

    assert '"id": "instruction-1"' in extractor.system_prompts[0]
    assert '"name": "Build the frame"' in extractor.system_prompts[0]
