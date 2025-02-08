"""Data models for the memory graph."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from uuid7 import uuid7


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid7())


class MemoryTier(str, Enum):
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    LONG_TERM = "long_term"


class MemoryNode(BaseModel):
    """A node in the memory graph."""
    id: str = Field(default_factory=_uuid)
    label: str
    type: str = "entity"  # entity, concept, fact, decision, tool, error
    properties: dict[str, Any] = Field(default_factory=dict)
    tier: MemoryTier = MemoryTier.SHORT_TERM
    session_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    embedding: list[float] | None = None
    confidence: float = 1.0

    # Sync metadata
    origin_machine: str | None = None
    version: int = 1


class MemoryEdge(BaseModel):
    """A typed relationship between two nodes."""
    id: str = Field(default_factory=_uuid)
    source: str  # node ID
    target: str  # node ID
    relation: str  # USES, CAUSED_BY, CONTRADICTS, REFINES, DEPENDS_ON, etc.
    properties: dict[str, Any] = Field(default_factory=dict)
    tier: MemoryTier = MemoryTier.SHORT_TERM
    session_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    weight: float = 1.0


class Episode(BaseModel):
    """A timestamped session event for episodic memory."""
    id: str = Field(default_factory=_uuid)
    session_id: str
    summary: str
    detail: str | None = None
    tags: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)  # linked graph nodes
    edge_ids: list[str] = Field(default_factory=list)  # linked graph edges
    created_at: datetime = Field(default_factory=_now)
    origin_machine: str | None = None


class Session(BaseModel):
    """A named working session."""
    id: str = Field(default_factory=_uuid)
    name: str
    description: str | None = None
    created_at: datetime = Field(default_factory=_now)
    last_active: datetime = Field(default_factory=_now)
    is_active: bool = True


class RetrievalResult(BaseModel):
    """A ranked memory hit from retrieval."""
    node: MemoryNode
    tier: MemoryTier
    score: float  # 0.0 - 1.0 relevance
    edges: list[MemoryEdge] = Field(default_factory=list)
    episode: Episode | None = None
