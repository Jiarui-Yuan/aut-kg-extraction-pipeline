"""Schemas for observations extracted from text and video segments."""

from collections import Counter

from pydantic import BaseModel, Field, model_validator


class ObservedAction(BaseModel):
    """An action observed during a bounded time interval.

    Actor and entity references are stored as names matching the corresponding
    observed actors and objects.
    """

    id: str
    start_time_ms: int
    end_time_ms: int

    actor: str
    action: str
    object: str | None = None
    instrument: str | None = None
    target: str | None = None


class ObservedObject(BaseModel):
    """An object observed in a segment."""

    name: str
    object_type: str
    action_ids: list[str] = Field(default_factory=list)

    def matches(self, other: object) -> bool:
        """Match objects by their observed name and type."""

        return (
            isinstance(other, ObservedObject)
            and self.name == other.name
            and self.object_type == other.object_type
        )

    def merge(self, other: "ObservedObject") -> None:
        """Merge action references while preserving observation order."""

        self.action_ids = list(
            dict.fromkeys([*self.action_ids, *other.action_ids])
        )


class ObservedActor(BaseModel):
    """An actor observed in a video segment."""

    name: str
    action_ids: list[str] = Field(default_factory=list)

    def matches(self, other: object) -> bool:
        """Match actors by their observed name."""

        return isinstance(other, ObservedActor) and self.name == other.name

    def merge(self, other: "ObservedActor") -> None:
        """Merge action references while preserving observation order."""

        self.action_ids = list(
            dict.fromkeys([*self.action_ids, *other.action_ids])
        )


class Segment(BaseModel):
    """A text-derived segment containing observations to be processed."""

    id: str
    scene: str
    actions: list[ObservedAction]
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

        self.id = id
        self.name = name
        self.segments = segments
        self.objects: dict[str, list[ObservedObject]] = {}
        self.actors: dict[str, list[ObservedActor]] = {}
        for segment in segments:
            self._add_object_references(segment)
            self.objects[segment.id] = segment.objects
            self.actors[segment.id] = self._actors_from_segment(segment)

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
