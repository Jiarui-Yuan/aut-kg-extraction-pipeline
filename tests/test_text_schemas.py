from backend.schemas.process_knowledge.text import (
    Instruction,
    ObservedAction,
    ObservedActor,
    ObservedObject,
    Segment,
    VideoSegment,
)


def make_video_segment(
    objects: list[ObservedObject],
    actors: list[ObservedActor] | None = None,
) -> VideoSegment:
    return VideoSegment(
        id="segment-1",
        start_time_ms=0,
        end_time_ms=1000,
        scene="workbench",
        actions=[],
        actors=actors or [],
        objects=objects,
        uncertainties=[],
    )


def test_video_segment_accepts_numbered_repeated_objects() -> None:
    segment = make_video_segment([
        ObservedObject(name="cup_1", object_type="cup"),
        ObservedObject(name="cup_2", object_type="cup"),
    ])

    assert [obj.name for obj in segment.objects] == ["cup_1", "cup_2"]


def test_video_segment_numbers_repeated_objects() -> None:
    segment = make_video_segment([
        ObservedObject(name="cup", object_type="cup"),
        ObservedObject(name="cup", object_type="cup"),
    ])

    assert [obj.name for obj in segment.objects] == ["cup_1", "cup_2"]


def test_video_segment_numbers_repeated_actors() -> None:
    segment = make_video_segment(
        objects=[],
        actors=[ObservedActor(name="worker"), ObservedActor(name="worker")],
    )

    assert [actor.name for actor in segment.actors] == ["worker_1", "worker_2"]


def test_instruction_adds_action_references_to_actors_and_objects() -> None:
    segment = Segment(
        id="segment-1",
        scene="workbench",
        actions=[
            ObservedAction(
                id="action-1",
                start_time_ms=0,
                end_time_ms=1,
                actor="worker",
                action="strike",
                object="nail",
                instrument="hammer",
            ),
            ObservedAction(
                id="action-2",
                start_time_ms=1,
                end_time_ms=2,
                actor="worker",
                action="inspect",
                object="nail",
            ),
        ],
        objects=[
            ObservedObject(name="hammer", object_type="tool"),
            ObservedObject(name="nail", object_type="fastener"),
        ],
        uncertainties=[],
    )

    instruction = Instruction(
        id="instruction-1",
        name="Drive a nail",
        segments=[segment],
    )

    assert instruction.actors["segment-1"] == [
        ObservedActor(name="worker", action_ids=["action-1", "action-2"]),
    ]
    assert instruction.objects["segment-1"] == [
        ObservedObject(
            name="hammer",
            object_type="tool",
            action_ids=["action-1"],
        ),
        ObservedObject(
            name="nail",
            object_type="fastener",
            action_ids=["action-1", "action-2"],
        ),
    ]
