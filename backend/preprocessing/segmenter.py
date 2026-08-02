"""Convert structured scene text into validated video segments."""

import re

from pydantic import BaseModel, Field

from backend.extraction.extractor import Extractor
from backend.schemas.process_knowledge.text import (
    Instruction,
    ObservedAction,
    ObservedActor,
    ObservedObject,
    VideoSegment,
)


class ExtractedAction(BaseModel):
    """One action extracted from an input segment."""

    start_time: str
    end_time: str
    actor: str
    action: str
    object_name: str | None = None
    instrument_name: str | None = None
    target_name: str | None = None


class ExtractedObject(BaseModel):
    """One object listed in an input segment."""

    name: str
    object_type: str


class SegmentDraft(BaseModel):
    """Intermediate information used to build a video segment."""

    scene: str
    actions: list[ExtractedAction] = Field(default_factory=list)
    objects: list[ExtractedObject] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class Segmenter:
    """Convert one external segment description into an Instruction."""

    SECTION_NAMES = (
        "Video",
        "Instruction name",
        "Segment",
        "Time",
        "Scene",
        "Actions",
        "Objects",
        "Uncertainty",
    )

    def __init__(self, *, extractor: Extractor | None = None):
        self.extractor = extractor or Extractor()

    def segment(self, text: str) -> Instruction:
        """Parse and convert one external video segment."""

        if not text.strip():
            raise ValueError("Input text must not be empty")

        video_id = self._read_header(text, "Video")
        segment_id = self._read_header(text, "Segment")
        instruction_name = self._read_header(
            text,
            "Instruction name",
            required=False,
        )
        if instruction_name is None:
            instruction_name = video_id

        time_range = self._read_header(text, "Time")
        start_time, end_time = self._split_time_range(time_range)

        draft = SegmentDraft(
            scene=self._read_section(text, "Scene"),
            actions=self._extract_actions(
                self._read_section(text, "Actions")
            ),
            objects=self._parse_objects(
                self._read_section(text, "Objects")
            ),
            uncertainties=self._parse_bullets(
                self._read_section(
                    text,
                    "Uncertainty",
                    required=False,
                )
                or ""
            ),
        )

        video_segment = self._build_video_segment(
            segment_id=segment_id,
            start_time=start_time,
            end_time=end_time,
            draft=draft,
        )

        return Instruction(
            id=video_id,
            name=instruction_name,
            segments=[video_segment],
        )

    @staticmethod
    def _read_header(
        text: str,
        label: str,
        *,
        required: bool = True,
    ) -> str | None:
        pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$"
        match = re.search(pattern, text)

        if match is not None:
            return match.group(1).strip()

        if required:
            raise ValueError(f"Missing required header: {label}")

        return None

    @classmethod
    def _read_section(
        cls,
        text: str,
        label: str,
        *,
        required: bool = True,
    ) -> str | None:
        section_names = "|".join(
            re.escape(name) for name in cls.SECTION_NAMES
        )
        pattern = (
            rf"(?ims)^\s*{re.escape(label)}\s*:\s*"
            rf"(.*?)(?=^\s*(?:{section_names})\s*:|\Z)"
        )
        match = re.search(pattern, text)

        if match is not None:
            value = match.group(1).strip()
            if value:
                return value

        if required:
            raise ValueError(f"Missing required section: {label}")

        return None

    @staticmethod
    def _split_time_range(value: str) -> tuple[str, str]:
        parts = re.split(r"\s+-\s+", value.strip(), maxsplit=1)

        if len(parts) != 2:
            raise ValueError(f"Invalid time range: {value!r}")

        return parts[0], parts[1]

    @staticmethod
    def _split_actions(value: str) -> list[str]:
        blocks = re.findall(
            r"(?ms)^\s*\d+\.\s*(.*?)(?=^\s*\d+\.\s*|\Z)",
            value,
        )

        if blocks:
            return [block.strip() for block in blocks]

        if value.strip():
            return [value.strip()]

        return []

    def _extract_actions(self, value: str) -> list[ExtractedAction]:
        return [
            self.extractor.extract(
                text=block,
                response_model=ExtractedAction,
                system_prompt=self._action_prompt(),
            )
            for block in self._split_actions(value)
        ]

    @staticmethod
    def _parse_objects(value: str) -> list[ExtractedObject]:
        objects: list[ExtractedObject] = []

        for line in value.splitlines():
            item = line.strip().lstrip("-•").strip()
            if not item:
                continue

            if ":" not in item:
                raise ValueError(f"Invalid object entry: {line!r}")

            name, object_type = item.split(":", maxsplit=1)
            objects.append(
                ExtractedObject(
                    name=name.strip(),
                    object_type=object_type.strip(),
                )
            )

        return objects

    @staticmethod
    def _parse_bullets(value: str) -> list[str]:
        return [
            line.strip().lstrip("-•").strip()
            for line in value.splitlines()
            if line.strip().lstrip("-•").strip()
        ]

    @staticmethod
    def _timestamp_to_ms(value: str) -> int:
        pattern = (
            r"^(?:(?P<hours>\d{1,2}):)?"
            r"(?P<minutes>\d{1,2}):"
            r"(?P<seconds>\d{1,2})"
            r"(?:[.,](?P<milliseconds>\d{1,3}))?$"
        )
        match = re.fullmatch(pattern, value.strip())

        if match is None:
            raise ValueError(f"Invalid timestamp: {value!r}")

        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        milliseconds = int(
            (match.group("milliseconds") or "0").ljust(3, "0")
        )

        if minutes >= 60 or seconds >= 60:
            raise ValueError(f"Invalid timestamp: {value!r}")

        return (
            hours * 3_600_000
            + minutes * 60_000
            + seconds * 1_000
            + milliseconds
        )

    @staticmethod
    def _name_key(value: str) -> str:
        normalized = " ".join(value.casefold().split())
        return re.sub(r"^(the|a|an)\s+", "", normalized)

    def _build_video_segment(
        self,
        *,
        segment_id: str,
        start_time: str,
        end_time: str,
        draft: SegmentDraft,
    ) -> VideoSegment:
        actor_names = list(
            dict.fromkeys(action.actor for action in draft.actions)
        )

        actors = [
            ObservedActor(
                id=f"{segment_id}-actor-{index}",
                name=name,
            )
            for index, name in enumerate(actor_names, start=1)
        ]
        actor_ids = {
            self._name_key(actor.name): actor.id
            for actor in actors
        }

        objects = [
            ObservedObject(
                id=f"{segment_id}-object-{index}",
                name=item.name,
                object_type=item.object_type,
            )
            for index, item in enumerate(draft.objects, start=1)
        ]
        object_ids = {
            self._name_key(item.name): item.id
            for item in objects
        }

        actions = [
            ObservedAction(
                id=f"{segment_id}-action-{index}",
                start_time_ms=self._timestamp_to_ms(
                    item.start_time
                ),
                end_time_ms=self._timestamp_to_ms(item.end_time),
                actor_id=self._lookup(
                    actor_ids,
                    item.actor,
                    "actor",
                ),
                action=item.action,
                object_id=self._optional_lookup(
                    object_ids,
                    item.object_name,
                    "object",
                ),
                instrument_id=self._optional_lookup(
                    object_ids,
                    item.instrument_name,
                    "instrument",
                ),
                target_id=self._optional_lookup(
                    object_ids,
                    item.target_name,
                    "target",
                ),
            )
            for index, item in enumerate(draft.actions, start=1)
        ]

        return VideoSegment(
            id=segment_id,
            start_time_ms=self._timestamp_to_ms(start_time),
            end_time_ms=self._timestamp_to_ms(end_time),
            scene=draft.scene,
            actions=actions,
            actors=actors,
            objects=objects,
            uncertainties=draft.uncertainties,
        )

    def _lookup(
        self,
        values: dict[str, str],
        name: str,
        kind: str,
    ) -> str:
        identifier = values.get(self._name_key(name))

        if identifier is None:
            raise ValueError(f"Unknown {kind}: {name!r}")

        return identifier

    def _optional_lookup(
        self,
        values: dict[str, str],
        name: str | None,
        kind: str,
    ) -> str | None:
        if name is None:
            return None

        return self._lookup(values, name, kind)

    @staticmethod
    def _action_prompt() -> str:
        return (
            "Extract exactly one observed action. Copy the start and end "
            "timestamps exactly. Return the actor and a short action verb. "
            "object_name is the item directly acted on. instrument_name is "
            "a tool used to perform the action. target_name is the destination "
            "or surface. The role depends on the action: when a worker picks "
            "up a screwdriver, the screwdriver is object_name, not the "
            "instrument. When a worker tightens a screw using a screwdriver, "
            "the screw is object_name and the screwdriver is instrument_name. "
            "Use null when a role is absent. Return data only."
        )