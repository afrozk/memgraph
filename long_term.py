"""Long-term memory — persistent graph in Kuzu embedded database."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memgraph.models import MemoryNode, MemoryEdge, MemoryTier


class LongTermStore:
    """Persistent knowledge graph backed by Kuzu (embedded columnar graph DB).

    Schema:
      Node table: memory_node (id, label, type, properties, confidence, ...)
      Rel table:  memory_edge (source → target, relation, properties, weight, ...)
    """

    def __init__(self, data_dir: Path):
        import kuzu

        self.db_path = data_dir / "longterm" / "kuzu_db"
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create node/rel tables if they don't exist."""
        try:
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS memory_node (
                    id STRING,
                    label STRING,
                    type STRING,
                    properties STRING,
                    session_id STRING,
                    created_at STRING,
                    updated_at STRING,
                    confidence DOUBLE,
                    origin_machine STRING,
                    version INT64,
                    PRIMARY KEY (id)
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS memory_edge (
                    FROM memory_node TO memory_node,
                    id STRING,
                    relation STRING,
                    properties STRING,
                    session_id STRING,
                    created_at STRING,
                    weight DOUBLE
                )
            """)
        except Exception:
            # Tables already exist
            pass

    def add_node(self, node: MemoryNode) -> MemoryNode:
        """Upsert a node into long-term memory."""
        node.tier = MemoryTier.LONG_TERM
        now = datetime.now(timezone.utc).isoformat()
        node.updated_at = datetime.now(timezone.utc)

        # Try merge (upsert)
        self.conn.execute(
            """
            MERGE (n:memory_node {id: $id})
            ON CREATE SET
                n.label = $label, n.type = $type, n.properties = $props,
                n.session_id = $sid, n.created_at = $cat, n.updated_at = $uat,
                n.confidence = $conf, n.origin_machine = $om, n.version = $ver
            ON MATCH SET
                n.label = $label, n.properties = $props,
                n.updated_at = $uat, n.confidence = $conf,
                n.version = n.version + 1
            """,
            parameters={
                "id": node.id, "label": node.label, "type": node.type,
                "props": json.dumps(node.properties), "sid": node.session_id or "",
                "cat": node.created_at.isoformat(), "uat": now,
                "conf": node.confidence, "om": node.origin_machine or "",
                "ver": node.version,
            },
        )
        return node

    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        """Add a relationship. Both source and target nodes must exist."""
        edge.tier = MemoryTier.LONG_TERM
        self.conn.execute(
            """
            MATCH (a:memory_node {id: $src}), (b:memory_node {id: $tgt})
            CREATE (a)-[:memory_edge {
                id: $id, relation: $rel, properties: $props,
                session_id: $sid, created_at: $cat, weight: $w
            }]->(b)
            """,
            parameters={
                "src": edge.source, "tgt": edge.target,
                "id": edge.id, "rel": edge.relation,
                "props": json.dumps(edge.properties),
                "sid": edge.session_id or "",
                "cat": edge.created_at.isoformat(), "w": edge.weight,
            },
        )
        return edge

    def search_nodes(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Search nodes by label (contains, case-insensitive)."""
        result = self.conn.execute(
            """
            MATCH (n:memory_node)
            WHERE n.label CONTAINS $q
            RETURN n.*
            ORDER BY n.updated_at DESC
            LIMIT $lim
            """,
            parameters={"q": query, "lim": limit},
        )
        nodes = []
        while result.has_next():
            row = result.get_next()
            nodes.append(self._row_to_node(row, result.get_column_names()))
        return nodes

    def get_subgraph(self, node_id: str, depth: int = 2) -> tuple[list[MemoryNode], list[MemoryEdge]]:
        """Get a node and its neighborhood up to N hops."""
        result = self.conn.execute(
            f"""
            MATCH (start:memory_node {{id: $nid}})
            OPTIONAL MATCH (start)-[e:memory_edge*1..{depth}]-(neighbor:memory_node)
            RETURN DISTINCT neighbor.*
            """,
            parameters={"nid": node_id},
        )
        nodes = []
        while result.has_next():
            row = result.get_next()
            if row[0] is not None:
                nodes.append(self._row_to_node(row, result.get_column_names()))
        # Also get the start node
        start_result = self.conn.execute(
            "MATCH (n:memory_node {id: $nid}) RETURN n.*",
            parameters={"nid": node_id},
        )
        if start_result.has_next():
            row = start_result.get_next()
            nodes.insert(0, self._row_to_node(row, start_result.get_column_names()))

        return nodes, []  # edges retrieval can be added

    def get_all_nodes(self, limit: int = 200) -> list[MemoryNode]:
        result = self.conn.execute(
            "MATCH (n:memory_node) RETURN n.* ORDER BY n.updated_at DESC LIMIT $lim",
            parameters={"lim": limit},
        )
        nodes = []
        while result.has_next():
            row = result.get_next()
            nodes.append(self._row_to_node(row, result.get_column_names()))
        return nodes

    def stats(self) -> dict:
        node_count = self.conn.execute(
            "MATCH (n:memory_node) RETURN count(n)"
        ).get_next()[0]
        edge_count = self.conn.execute(
            "MATCH ()-[e:memory_edge]->() RETURN count(e)"
        ).get_next()[0]
        return {
            "nodes": node_count,
            "edges": edge_count,
            "tier": "long_term",
            "db_path": str(self.db_path),
        }

    def _row_to_node(self, row: list, columns: list[str]) -> MemoryNode:
        """Convert a Kuzu result row into a MemoryNode."""
        data = {}
        for col, val in zip(columns, row):
            # Strip the 'n.' prefix from column names
            key = col.split(".")[-1] if "." in col else col
            data[key] = val

        data["properties"] = json.loads(data.get("properties", "{}"))
        data["tier"] = MemoryTier.LONG_TERM
        return MemoryNode(**data)

    def export_all(self) -> list[dict]:
        """Export all nodes and edges for sync."""
        nodes = [n.model_dump(mode="json") for n in self.get_all_nodes(limit=10000)]
        return nodes
