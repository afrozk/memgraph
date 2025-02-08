"""LLM client — talks to LM Studio via OpenAI-compatible API."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from memgraph.config import LLMConfig
from memgraph.models import MemoryNode, MemoryEdge


class LMStudioClient:
    """Wraps LM Studio's local OpenAI-compatible endpoint."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,  # LM Studio ignores this
        )

    def extract_entities(self, text: str) -> tuple[list[MemoryNode], list[MemoryEdge]]:
        """Ask the loaded LM Studio model to extract entities and relations."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": self.config.extract_system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return [], []

        nodes = []
        for ent in data.get("entities", []):
            nodes.append(MemoryNode(
                label=ent.get("label", ent.get("id", "unknown")),
                type=ent.get("type", "entity"),
                properties=ent.get("properties", {}),
            ))

        # Build an ID lookup so edges can reference by label
        label_to_id = {n.label.lower(): n.id for n in nodes}

        edges = []
        for rel in data.get("edges", []):
            src = label_to_id.get(rel.get("source", "").lower())
            tgt = label_to_id.get(rel.get("target", "").lower())
            if src and tgt:
                edges.append(MemoryEdge(
                    source=src,
                    target=tgt,
                    relation=rel.get("relation", "RELATED_TO"),
                    properties=rel.get("properties", {}),
                ))

        return nodes, edges

    def query_with_context(
        self,
        query: str,
        context_facts: list[str],
        system_prompt: str | None = None,
    ) -> str:
        """Answer a question using retrieved memory context."""
        ctx_block = "\n".join(f"- {fact}" for fact in context_facts)
        sys = system_prompt or (
            "You are a developer assistant with access to a memory graph. "
            "Use the following retrieved facts to answer the question. "
            "If facts conflict, note the conflict and use the most recent one.\n\n"
            f"Retrieved memory:\n{ctx_block}"
        )
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content

    def summarise_for_promotion(self, text: str) -> str:
        """Generate a concise summary for episodic/long-term storage."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=0.0,
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarise the following developer context into a single concise "
                        "statement suitable for long-term memory storage. Focus on facts, "
                        "decisions, and technical details. No filler."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content

    def health_check(self) -> dict:
        """Check if LM Studio is running and a model is loaded."""
        try:
            models = self.client.models.list()
            model_list = [m.id for m in models.data]
            return {"status": "ok", "models": model_list, "base_url": self.config.base_url}
        except Exception as e:
            return {"status": "error", "error": str(e), "base_url": self.config.base_url}
