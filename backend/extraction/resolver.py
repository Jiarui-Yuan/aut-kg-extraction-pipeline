import json
from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel

from backend.extraction.extractor import Extractor
from backend.schemas.process_knowledge.entities import Material, PPE, Tool

EntityT = TypeVar("EntityT")
SourceT = TypeVar("SourceT")


class EquivalenceGroup(BaseModel):
    """Input values that refer to the same semantic entity."""

    canonical_entity: str
    entities: list[str]


class SemanticResolution(BaseModel):
    """LLM response containing a partition of all supplied entity values."""

    groups: list[EquivalenceGroup]


ObjectEntity: TypeAlias = Tool | Material | PPE


class ClassifiedObject(BaseModel):
    """A resolved object classified as one supported knowledge entity."""

    input_label: str
    entity_type: Literal["tool", "material", "ppe"]
    tool: Tool | None = None
    material: Material | None = None
    ppe: PPE | None = None


class Resolver(ABC, Generic[EntityT, SourceT]):
    """Base resolver for logical and LLM-assisted semantic deduplication."""

    def __init__(self, source: SourceT | None = None):
        self._entities: list[EntityT] = []
        self._source: SourceT | None = None
        if source is not None:
            self.add_source(source)

    def add(self, entity: EntityT) -> EntityT:
        for existing in self._entities:
            if self._matches(existing, entity):
                merge = getattr(existing, "merge", None)
                if callable(merge):
                    merge(entity)
                return existing

        self._entities.append(entity)
        return entity

    def add_source(self, source: SourceT) -> list[EntityT]:
        """Add all entities exposed by a resolver-specific source."""

        self._source = source
        for entity in self.entities_from_source(source):
            self.add(entity)
        return self.entities

    def resolve(self, source: SourceT) -> list[EntityT]:
        """Logically resolve entities from a source into this resolver."""

        return self.add_source(source)

    @abstractmethod
    def entities_from_source(self, source: SourceT) -> list[EntityT]:
        """Return entities from the source in observation order."""

    @abstractmethod
    def semantic_label(self, entity: EntityT) -> str:
        """Return the entity representation supplied to the LLM."""

    @property
    @abstractmethod
    def entity_kind(self) -> str:
        """Human-readable entity type used in the semantic prompt."""

    def semantic_deduplicate(
        self,
        entities: list[EntityT] | None = None,
        *,
        extractor: Extractor | None = None,
    ) -> list[EntityT]:
        """Use an LLM to merge values that refer to the same semantic entity.

        The logical entity list is used by default and is not mutated.
        """

        candidates = entities if entities is not None else self._entities
        labels = [self.semantic_label(entity) for entity in candidates]
        if len(labels) != len(set(labels)):
            raise ValueError(
                "Semantic labels must be unique after logical deduplication"
            )
        if not candidates:
            return []

        semantic_extractor = extractor or Extractor()
        resolution = semantic_extractor.extract(
            text=json.dumps(labels, ensure_ascii=False),
            response_model=SemanticResolution,
            system_prompt=self._semantic_deduplication_prompt(),
        )
        canonical_labels = self._validate_semantic_resolution(
            labels,
            resolution,
        )
        entities_by_label = dict(zip(labels, candidates, strict=True))
        for group in resolution.groups:
            canonical = entities_by_label[group.canonical_entity]
            merge = getattr(canonical, "merge", None)
            if callable(merge):
                for label in group.entities:
                    if label != group.canonical_entity:
                        merge(entities_by_label[label])
        return [entities_by_label[label] for label in canonical_labels]

    def _semantic_deduplication_prompt(self) -> str:
        return (
            f"You resolve {self.entity_kind} identity in industrial process "
            "instructions. The user provides a JSON list of logically unique "
            f"{self.entity_kind} labels. Group labels only when they "
            f"semantically refer to the same {self.entity_kind}. Every input "
            "label must occur in exactly one group. For each group, choose "
            "canonical_entity verbatim from that group's input labels. Do not "
            "add, remove, translate, or rewrite labels. A label with no "
            "equivalent must be returned as a one-item group. "
            f"{self.source_context()}"
        )

    def source_context(self) -> str:
        """Format source metadata for inclusion in every LLM prompt."""

        if self._source is None:
            return "No source instruction metadata is available."

        metadata = {
            "id": getattr(self._source, "id", None),
            "name": getattr(self._source, "name", None),
        }
        return (
            "Source instruction metadata: "
            f"{json.dumps(metadata, ensure_ascii=False)}."
        )

    @staticmethod
    def _validate_semantic_resolution(
        labels: list[str],
        resolution: SemanticResolution,
    ) -> list[str]:
        input_labels = set(labels)
        resolved_labels = [
            entity
            for group in resolution.groups
            for entity in group.entities
        ]

        if (
            len(resolved_labels) != len(labels)
            or set(resolved_labels) != input_labels
        ):
            raise ValueError(
                "The semantic resolution must contain every input entity "
                "exactly once"
            )

        canonical_by_entity: dict[str, str] = {}
        for group in resolution.groups:
            if (
                not group.entities
                or group.canonical_entity not in group.entities
            ):
                raise ValueError(
                    "Each semantic group must choose one of its entities as "
                    "the canonical entity"
                )
            for entity in group.entities:
                canonical_by_entity[entity] = group.canonical_entity

        # Preserve the first-observed order of the semantic groups.
        return list(
            dict.fromkeys(canonical_by_entity[label] for label in labels)
        )

    @staticmethod
    def _matches(existing: Any, entity: Any) -> bool:
        matches = getattr(existing, "matches", None)
        if callable(matches):
            return bool(matches(entity))
        return bool(existing == entity)

    @property
    def entities(self) -> list[EntityT]:
        return self._entities
