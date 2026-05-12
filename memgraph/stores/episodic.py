"""Episodic memory — timestamped session events in SQLite."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import sqlite_utils

from memgraph.models import Episode, MemoryNode, MemoryEdge, MemoryTier


class EpisodicStore:
    """Append-only event log backed by SQLite. Each episode links to graph nodes."""

    def __init__(self, data_dir: Path):
        self.db_path = data_dir / "episodic" / "episodes.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite_utils.Database(str(self.db_path))
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        if "episodes" not in self.db.table_names():
            self.db["episodes"].create({
                "id": str,
                "session_id": str,
                "summary": str,
                "detail": str,
                "tags": str,  # JSON array
                "node_ids": str,  # JSON array
                "edge_ids": str,  # JSON array
                "created_at": str,
                "origin_machine": str,
            }, pk="id")
            self.db["episodes"].create_index(["session_id"])
            self.db["episodes"].create_index(["created_at"])

        if "episodic_nodes" not in self.db.table_names():
            self.db["episodic_nodes"].create({
                "id": str,
                "label": str,
                "type": str,
                "properties": str,  # JSON
                "session_id": str,
                "created_at": str,
                "updated_at": str,
                "confidence": float,
                "origin_machine": str,
                "version": int,
            }, pk="id")

        if "episodic_edges" not in self.db.table_names():
            self.db["episodic_edges"].create({
                "id": str,
                "source": str,
                "target": str,
                "relation": str,
                "properties": str,
                "session_id": str,
                "created_at": str,
                "weight": float,
            }, pk="id")

    def save_episode(
        self,
        summary: str,
        session_id: str,
        detail: str | None = None,
        tags: list[str] | None = None,
        nodes: list[MemoryNode] | None = None,
        edges: list[MemoryEdge] | None = None,
        origin_machine: str | None = None,
    ) -> Episode:
        """Save a developer-prompted episode with linked graph nodes."""
        nodes = nodes or []
        edges = edges or []
        tags = tags or []

        # Persist linked nodes/edges into episodic tables
        for node in nodes:
            node.tier = MemoryTier.EPISODIC
            self.db["episodic_nodes"].upsert(
                {
                    "id": node.id,
                    "label": node.label,
                    "type": node.type,
                    "properties": json.dumps(node.properties),
                    "session_id": session_id,
                    "created_at": node.created_at.isoformat(),
                    "updated_at": node.updated_at.isoformat(),
                    "confidence": node.confidence,
                    "origin_machine": origin_machine,
                    "version": node.version,
                },
                pk="id",
            )

        for edge in edges:
            edge.tier = MemoryTier.EPISODIC
            self.db["episodic_edges"].upsert(
                {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "properties": json.dumps(edge.properties),
                    "session_id": session_id,
                    "created_at": edge.created_at.isoformat(),
                    "weight": edge.weight,
                },
                pk="id",
            )

        episode = Episode(
            session_id=session_id,
            summary=summary,
            detail=detail,
            tags=tags,
            node_ids=[n.id for n in nodes],
            edge_ids=[e.id for e in edges],
            origin_machine=origin_machine,
        )

        self.db["episodes"].insert({
            "id": episode.id,
            "session_id": episode.session_id,
            "summary": episode.summary,
            "detail": episode.detail,
            "tags": json.dumps(episode.tags),
            "node_ids": json.dumps(episode.node_ids),
            "edge_ids": json.dumps(episode.edge_ids),
            "created_at": episode.created_at.isoformat(),
            "origin_machine": episode.origin_machine,
        })

        return episode

    def search(self, query: str, limit: int = 10) -> list[Episode]:
        """Search episodes by summary text."""
        rows = self.db.execute(
            "SELECT * FROM episodes WHERE summary LIKE ? ORDER BY created_at DESC LIMIT ?",
            [f"%{query}%", limit],
        ).fetchall()
        columns = [d[0] for d in self.db.execute("SELECT * FROM episodes LIMIT 0").description]
        results = []
        for row in rows:
            data = dict(zip(columns, row))
            data["tags"] = json.loads(data["tags"])
            data["node_ids"] = json.loads(data["node_ids"])
            data["edge_ids"] = json.loads(data["edge_ids"])
            results.append(Episode(**data))
        return results

    def get_by_session(self, session_id: str, limit: int = 50) -> list[Episode]:
        """Get all episodes for a session."""
        rows = list(self.db["episodes"].rows_where(
            "session_id = ?", [session_id],
            order_by="created_at desc", limit=limit,
        ))
        for r in rows:
            r["tags"] = json.loads(r["tags"])
            r["node_ids"] = json.loads(r["node_ids"])
            r["edge_ids"] = json.loads(r["edge_ids"])
        return [Episode(**r) for r in rows]

    def get_recent(self, limit: int = 20) -> list[Episode]:
        """Get most recent episodes across all sessions."""
        rows = list(self.db["episodes"].rows_where(
            order_by="created_at desc", limit=limit,
        ))
        for r in rows:
            r["tags"] = json.loads(r["tags"])
            r["node_ids"] = json.loads(r["node_ids"])
            r["edge_ids"] = json.loads(r["edge_ids"])
        return [Episode(**r) for r in rows]

    def stats(self) -> dict:
        return {
            "episodes": self.db["episodes"].count,
            "nodes": self.db["episodic_nodes"].count,
            "edges": self.db["episodic_edges"].count,
            "tier": "episodic",
        }

    def export_all(self) -> list[dict]:
        """Export all episodes for sync."""
        return list(self.db["episodes"].rows)
