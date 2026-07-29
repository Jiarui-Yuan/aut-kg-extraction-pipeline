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
