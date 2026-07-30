"""Schemas for observations extracted from text and video segments."""

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, NewType

from pydantic import BaseModel, Field, model_validator

ActionId = NewType("ActionId", str)
ActorId = NewType("ActorId", str)
ObjectId = NewType("ObjectId", str)
SegmentId = NewType("SegmentId", str)


class ObservedAction(BaseModel):
    """An action observed during a bounded time interval.

    Actor and entity references use typed observation IDs. Display names are
    never used as foreign keys.
    """

    id: ActionId
    start_time_ms: int
    end_time_ms: int

    actor_id: ActorId
    action: str
    object_id: ObjectId | None = None
    instrument_id: ObjectId | None = None
    target_id: ObjectId | None = None


class ObservedObject(BaseModel):
    """An object observed in a segment."""

    id: ObjectId
    name: str
    object_type: str
    action_ids: list[ActionId] = Field(default_factory=list)

    def matches(self, other: object) -> bool:
        """Match objects by their observed name and type."""

        return (
            isinstance(other, ObservedObject)
            and self.name == other.name
            and self.object_type == other.object_type
        )

    def merge(self, other: "ObservedObject") -> None:
        """Merge action references while preserving observation order."""

        self.action_ids = list(dict.fromkeys([*self.action_ids, *other.action_ids]))


class ObservedActor(BaseModel):
    """An actor observed in a video segment."""

    id: ActorId
    name: str
    action_ids: list[ActionId] = Field(default_factory=list)

    def matches(self, other: object) -> bool:
        """Match actors by their observed name."""

        return isinstance(other, ObservedActor) and self.name == other.name

    def merge(self, other: "ObservedActor") -> None:
        """Merge action references while preserving observation order."""

        self.action_ids = list(dict.fromkeys([*self.action_ids, *other.action_ids]))


class Segment(BaseModel):
    """A text-derived segment containing observations to be processed."""

    id: SegmentId
    scene: str
    actions: list[ObservedAction]
    actors: list[ObservedActor]
    objects: list[ObservedObject]
    uncertainties: list[str]


class VideoSegment(Segment):
    """A time-bounded segment produced by the video-processing pipeline.

    Repeated object and actor names are numbered after validation so that every
    observed entity has a unique name within the segment.
    """

    start_time_ms: int
    end_time_ms: int

    scene: str
    actions: list[ObservedAction]
    actors: list[ObservedActor]
    objects: list[ObservedObject]
    uncertainties: list[str]

    @model_validator(mode="after")
    def number_repeated_entities(self) -> "VideoSegment":
        """Number repeated object and actor names in their observation order."""

        self._number_repeated_names(self.objects)
        self._number_repeated_names(self.actors)
        return self

    @staticmethod
    def _number_repeated_names(
        entities: list[ObservedObject] | list[ObservedActor],
    ) -> None:
        """Append one-based indices to every occurrence of a repeated name."""

        name_counts = Counter(entity.name for entity in entities)
        next_number: Counter[str] = Counter()

        for entity in entities:
            original_name = entity.name
            if name_counts[original_name] > 1:
                next_number[original_name] += 1
                entity.name = f"{original_name}_{next_number[original_name]}"


@dataclass(frozen=True)
class InstructionReferenceIndex:
    """Read-only instruction-wide identity and reverse-reference indexes."""

    segments_by_id: Mapping[SegmentId, Segment]
    actions_by_id: Mapping[ActionId, ObservedAction]
    actors_by_id: Mapping[ActorId, ObservedActor]
    objects_by_id: Mapping[ObjectId, ObservedObject]
    actor_action_ids: Mapping[ActorId, tuple[ActionId, ...]]
    object_action_ids: Mapping[ObjectId, tuple[ActionId, ...]]

class Instruction:
    """A collection of segments with observations indexed by segment ID.

    Objects are taken directly from each segment. Actors are derived from the
    actor references in its actions and de-duplicated while retaining their
    first-observed order.
    """

    def __init__(
        self,
        id: str,
        name: str,
        segments: list[Segment],
    ):
        """Initialize an instruction and build its per-segment lookup maps."""

        self._validate_reference_ids(segments)
        self.id = id
        self.name = name
        self.segments = segments
        self.objects: dict[str, list[ObservedObject]] = {}
        self.actors: dict[str, list[ObservedActor]] = {}
        for segment in segments:
            self._add_object_references(segment)
            self.objects[segment.id] = segment.objects
            self.actors[segment.id] = self._actors_from_segment(segment)

    @staticmethod
    def _validate_reference_ids(segments: list[Segment]) -> None:
        """Ensure every ID used as an instruction-wide lookup key is unique."""

        segment_counts = Counter(segment.id for segment in segments)
        duplicate_segment_ids = sorted(
            segment_id for segment_id, count in segment_counts.items() if count > 1
        )
        if duplicate_segment_ids:
            raise ValueError(
                "Instruction segment IDs must be unique; duplicates: "
                f"{duplicate_segment_ids!r}"
            )

        action_counts = Counter(
            action.id for segment in segments for action in segment.actions
        )
        duplicate_action_ids = sorted(
            action_id for action_id, count in action_counts.items() if count > 1
        )
        if duplicate_action_ids:
            raise ValueError(
                "Instruction action IDs must be globally unique; duplicates: "
                f"{duplicate_action_ids!r}"
            )

    def extract_actors(self) -> list[ObservedActor]:
        """Extract a de-duplicated list of actors across all segments."""

        actors_by_name: dict[str, ObservedActor] = {}
        for segment_actors in self.actors.values():
            for actor in segment_actors:
                existing = actors_by_name.get(actor.name)
                if existing is None:
                    actors_by_name[actor.name] = actor.model_copy(deep=True)
                else:
                    existing.merge(actor)
        return list(actors_by_name.values())

    @staticmethod
    def _actors_from_segment(segment: Segment) -> list[ObservedActor]:
        actors_by_name: dict[str, ObservedActor] = {}
        for action in segment.actions:
            actor = actors_by_name.setdefault(
                action.actor,
                ObservedActor(name=action.actor),
            )
            if action.id not in actor.action_ids:
                actor.action_ids.append(action.id)
        return list(actors_by_name.values())

    @staticmethod
    def _add_object_references(segment: Segment) -> None:
        for observed_object in segment.objects:
            referenced_ids = [
                action.id
                for action in segment.actions
                if observed_object.name
                in (action.object, action.instrument, action.target)
            ]
            observed_object.action_ids = list(
                dict.fromkeys([
                    *observed_object.action_ids,
                    *referenced_ids,
                ])
            )
