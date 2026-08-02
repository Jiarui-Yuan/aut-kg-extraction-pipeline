"""Tests for the segmentation module."""

import pytest

from backend.preprocessing.segmenter import (
    ExtractedAction,
    Segmenter,
)


class FakeExtractor:
    """Return prepared actions without calling Ollama."""

    def __init__(self, actions: list[ExtractedAction]):
        self.actions = iter(actions)
        self.calls: list[dict[str, object]] = []

    def extract(self, **kwargs: object) -> ExtractedAction:
        self.calls.append(kwargs)
        return next(self.actions)


def make_actions() -> list[ExtractedAction]:
    """Create two actions for the assembly example."""

    return [
        ExtractedAction(
            start_time="00:01:12.400",
            end_time="00:01:14.100",
            actor="worker_1",
            action="pick up",
            object_name="screwdriver",
        ),
        ExtractedAction(
            start_time="00:01:14.200",
            end_time="00:01:18.900",
            actor="worker_1",
            action="tighten",
            object_name="screw",
            instrument_name="screwdriver",
            target_name="metal panel",
        ),
    ]


def sample_text() -> str:
    """Return one input in the expected upstream format."""

    return """
Video: indego_assembly_001
Instruction name: Indego assembly
Segment: seg_0003
Time: 00:01:12.400 - 00:01:18.900

Scene:
The worker picks up a screwdriver and tightens a screw
on the metal panel.

Actions:
1. [00:01:12.400 - 00:01:14.100]
   worker_1 picks up the screwdriver.

2. [00:01:14.200 - 00:01:18.900]
   worker_1 tightens the screw on the metal panel
   using the screwdriver.

Objects:
- screwdriver: tool
- screw: component
- metal panel: workpiece

Uncertainty:
- The exact screw type is not visible.
"""


def test_segmenter_creates_valid_instruction() -> None:
    fake_extractor = FakeExtractor(make_actions())
    segmenter = Segmenter(extractor=fake_extractor)

    instruction = segmenter.segment(sample_text())

    assert instruction.id == "indego_assembly_001"
    assert instruction.name == "Indego assembly"
    assert len(instruction.segments) == 1

    segment = instruction.segments[0]
    assert segment.id == "seg_0003"
    assert segment.start_time_ms == 72_400
    assert segment.end_time_ms == 78_900
    assert len(segment.actions) == 2
    assert len(segment.actors) == 1
    assert len(segment.objects) == 3

    assert segment.actions[0].object_id == "seg_0003-object-1"
    assert segment.actions[1].object_id == "seg_0003-object-2"
    assert (
        segment.actions[1].instrument_id
        == "seg_0003-object-1"
    )
    assert segment.actions[1].target_id == "seg_0003-object-3"

    assert segment.actors[0].action_ids == [
        "seg_0003-action-1",
        "seg_0003-action-2",
    ]
    assert segment.objects[0].action_ids == [
        "seg_0003-action-1",
        "seg_0003-action-2",
    ]

    assert len(fake_extractor.calls) == 2
    assert all(
        call["response_model"] is ExtractedAction
        for call in fake_extractor.calls
    )


def test_segmenter_rejects_empty_input() -> None:
    fake_extractor = FakeExtractor(make_actions())
    segmenter = Segmenter(extractor=fake_extractor)

    with pytest.raises(ValueError, match="must not be empty"):
        segmenter.segment("   ")

    assert fake_extractor.calls == []


def test_segmenter_rejects_missing_video_header() -> None:
    fake_extractor = FakeExtractor(make_actions())
    segmenter = Segmenter(extractor=fake_extractor)

    with pytest.raises(ValueError, match="Missing required header: Video"):
        segmenter.segment(
            "Segment: seg_0003\n"
            "Time: 00:01:12.400 - 00:01:18.900"
        )

    assert fake_extractor.calls == []