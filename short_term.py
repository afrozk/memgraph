"""Short-term memory — in-memory graph per session using NetworkX."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx

from memgraph.models import MemoryNode, MemoryEdge, MemoryTier


class ShortTermStore:
    """In-memory graph for the active session. Serialises to disk on save."""

    def __init__(self, data_dir: Path, session_name: str):
        self.data_dir = data_dir / "short_term"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session_name = session_name
        self._db_path = self.data_dir / f"{session_name}.pkl"
        self.graph: nx.DiGraph = self._load_or_create()

    def _load_or_create(self) -> nx.DiGraph:
        if self._db_path.exists():
            with open(self._db_path, "rb") as f:
                return pickle.load(f)
        return nx.DiGraph()

    def add_node(self, node: MemoryNode) -> MemoryNode:
        """Add or update a node in the short-term graph."""
        node.tier = MemoryTier.SHORT_TERM
        node.session_id = self.session_name
        self.graph.add_node(
            node.id,
            data=node.model_dump(mode="json"),
        )
        return node

    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        """Add a relationship edge."""
        edge.tier = MemoryTier.SHORT_TERM
        edge.session_id = self.session_name
        self.graph.add_edge(
            edge.source,
            edge.target,
            key=edge.id,
            data=edge.model_dump(mode="json"),
        )
        return edge

    def get_node(self, node_id: str) -> MemoryNode | None:
        if node_id in self.graph.nodes:
            return MemoryNode(**self.graph.nodes[node_id]["data"])
        return None

    def find_by_label(self, label: str) -> list[MemoryNode]:
        """Fuzzy find nodes by label (case-insensitive contains)."""
        label_lower = label.lower()
        results = []
        for nid, ndata in self.graph.nodes(data=True):
            node = MemoryNode(**ndata["data"])
            if label_lower in node.label.lower():
                results.append(node)
        return results

    def get_neighbors(self, node_id: str, depth: int = 1) -> list[MemoryNode]:
        """Get nodes within N hops."""
        if node_id not in self.graph:
            return []
        visited = set()
        frontier = {node_id}
        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                for neighbor in list(self.graph.successors(nid)) + list(self.graph.predecessors(nid)):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            visited |= frontier
            frontier = next_frontier
        visited |= frontier
        visited.discard(node_id)
        return [MemoryNode(**self.graph.nodes[nid]["data"]) for nid in visited if nid in self.graph.nodes]

    def get_all_nodes(self) -> list[MemoryNode]:
        return [MemoryNode(**d["data"]) for _, d in self.graph.nodes(data=True)]

    def get_all_edges(self) -> list[MemoryEdge]:
        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append(MemoryEdge(**d["data"]))
        return edges

    def stats(self) -> dict:
        return {
            "session": self.session_name,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "tier": "short_term",
        }

    def persist(self) -> None:
        """Save graph to disk."""
        with open(self._db_path, "wb") as f:
            pickle.dump(self.graph, f)

    def clear(self) -> None:
        """Wipe short-term memory for this session."""
        self.graph.clear()
        if self._db_path.exists():
            self._db_path.unlink()
