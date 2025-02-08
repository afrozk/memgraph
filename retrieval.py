"""Retrieval engine — queries all three tiers and ranks results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from memgraph.models import (
    MemoryNode, MemoryEdge, MemoryTier, RetrievalResult, Episode,
)
from memgraph.stores.short_term import ShortTermStore
from memgraph.stores.episodic import EpisodicStore
from memgraph.stores.long_term import LongTermStore


class RetrievalEngine:
    """Fan-out search across all memory tiers, score and rank results."""

    def __init__(
        self,
        short_term: ShortTermStore,
        episodic: EpisodicStore,
        long_term: LongTermStore,
    ):
        self.short_term = short_term
        self.episodic = episodic
        self.long_term = long_term

    def retrieve(self, query: str, limit: int = 15) -> list[RetrievalResult]:
        """Search all tiers and return ranked results."""
        results: list[RetrievalResult] = []

        # --- Short-term: label search ---
        for node in self.short_term.find_by_label(query):
            neighbors = self.short_term.get_neighbors(node.id, depth=1)
            edges = []  # could extract from graph
            score = self._score_short_term(node, query)
            results.append(RetrievalResult(
                node=node, tier=MemoryTier.SHORT_TERM,
                score=score, edges=edges,
            ))

        # --- Episodic: text search on summaries ---
        for ep in self.episodic.search(query, limit=limit):
            # Create a synthetic node from the episode summary
            ep_node = MemoryNode(
                id=ep.id,
                label=ep.summary,
                type="episode",
                tier=MemoryTier.EPISODIC,
                session_id=ep.session_id,
                created_at=ep.created_at,
            )
            score = self._score_episodic(ep, query)
            results.append(RetrievalResult(
                node=ep_node, tier=MemoryTier.EPISODIC,
                score=score, episode=ep,
            ))

        # --- Long-term: Kuzu label search ---
        for node in self.long_term.search_nodes(query, limit=limit):
            score = self._score_long_term(node, query)
            results.append(RetrievalResult(
                node=node, tier=MemoryTier.LONG_TERM,
                score=score,
            ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def retrieve_formatted(self, query: str, limit: int = 10) -> list[str]:
        """Return human-readable context strings for LLM injection."""
        results = self.retrieve(query, limit=limit)
        lines = []
        for r in results:
            tier_tag = r.tier.value.replace("_", "-")
            stars = "★" * max(1, int(r.score * 3)) + "☆" * (3 - max(1, int(r.score * 3)))
            line = f"[{tier_tag}] {stars} {r.node.label}"
            if r.node.properties:
                props = ", ".join(f"{k}={v}" for k, v in r.node.properties.items())
                line += f" ({props})"
            if r.episode:
                line += f" | session: {r.episode.session_id}"
            lines.append(line)
        return lines

    def detect_conflicts(self, query: str) -> list[tuple[RetrievalResult, RetrievalResult, str]]:
        """Find conflicting facts across tiers."""
        results = self.retrieve(query, limit=20)
        conflicts = []

        # Simple heuristic: same entity label appearing in different tiers
        # with different properties
        by_label: dict[str, list[RetrievalResult]] = {}
        for r in results:
            key = r.node.label.lower().strip()
            by_label.setdefault(key, []).append(r)

        for label, hits in by_label.items():
            if len(hits) < 2:
                continue
            for i, a in enumerate(hits):
                for b in hits[i + 1:]:
                    if a.tier != b.tier and a.node.properties != b.node.properties:
                        conflicts.append((
                            a, b,
                            f"'{label}' differs between {a.tier.value} and {b.tier.value}",
                        ))

        return conflicts

    # --- Scoring heuristics ---

    def _score_short_term(self, node: MemoryNode, query: str) -> float:
        """Short-term gets recency boost but lower base."""
        base = 0.5
        label_match = 0.3 if query.lower() in node.label.lower() else 0.0
        recency = self._recency_boost(node.created_at, hours=2)
        return min(1.0, base + label_match + recency * 0.2)

    def _score_episodic(self, episode: Episode, query: str) -> float:
        """Episodic scored by text match and age."""
        base = 0.4
        match = 0.3 if query.lower() in episode.summary.lower() else 0.0
        recency = self._recency_boost(episode.created_at, hours=168)  # 1 week
        return min(1.0, base + match + recency * 0.2)

    def _score_long_term(self, node: MemoryNode, query: str) -> float:
        """Long-term gets high base (trusted knowledge)."""
        base = 0.6
        match = 0.3 if query.lower() in node.label.lower() else 0.0
        confidence = node.confidence * 0.1
        return min(1.0, base + match + confidence)

    @staticmethod
    def _recency_boost(created_at: datetime, hours: float) -> float:
        """1.0 if just created, decays to 0.0 over `hours`."""
        now = datetime.now(timezone.utc)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_hours = (now - created_at).total_seconds() / 3600
        return max(0.0, 1.0 - (age_hours / hours))
