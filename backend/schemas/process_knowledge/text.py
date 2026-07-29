"""Schemas for observations extracted from text and video segments."""

from pydantic import BaseModel, Field


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
