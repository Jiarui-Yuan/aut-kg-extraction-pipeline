"""
Base classes for all schema definitions.

These provide common fields and patterns used across domains.
Every entity and relation in the knowledge graph inherits from these.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel


class KGEntity(BaseModel):
    """Base class for all knowledge graph entities."""
    name: str
    description: Optional[str] = None
    source_text: Optional[str] = None


class KGRelation(BaseModel):
    """
    Base class for all knowledge graph relations.
    
    In a property graph (LadybugDB/Neo4j), n-ary relations are represented
    through reification: this relation becomes an intermediate node with 
    binary edges to each participant.
    
    In TypeDB, these map directly to n-ary relations with roles.
    """
    source_text: Optional[str] = None


class ExtractionResult(BaseModel):
    """
    Container for all extracted information from a single text chunk.
    
    The extractor fills this with whatever it finds in the text.
    Assembly and deduplication happen downstream.
    """
    entities: List[KGEntity] = []
    relations: List[KGRelation] = []
    source_chunk: Optional[str] = None
    extraction_model: Optional[str] = None
    extraction_timestamp: Optional[datetime] = None