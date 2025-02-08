"""Configuration management for MemGraph."""

from __future__ import annotations

import tomli
from pathlib import Path
from pydantic import BaseModel, Field


DEFAULT_DATA_DIR = Path.home() / ".memgraph"


class LLMConfig(BaseModel):
    """LM Studio connection settings."""
    base_url: str = "http://localhost:1234/v1"
    model: str = "default"  # LM Studio serves whatever is loaded
    api_key: str = "lm-studio"  # LM Studio doesn't check this
    temperature: float = 0.1
    max_tokens: int = 2048

    # Entity extraction prompt
    extract_system_prompt: str = (
        "You are a knowledge graph builder. Extract entities and relationships "
        "from the user's message. Return JSON with 'entities' (list of "
        "{id, label, type, properties}) and 'edges' (list of "
        "{source, target, relation, properties}). Be precise and concise."
    )


class EmbeddingConfig(BaseModel):
    """Local embedding model settings."""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # or "mps" for Apple Silicon


class SyncConfig(BaseModel):
    """LAN sync settings."""
    enabled: bool = True
    service_name: str = "_memgraph._tcp.local."
    port: int = 50051
    auto_sync: bool = False
    sync_interval_seconds: int = 300  # 5 minutes


class MemGraphConfig(BaseModel):
    """Root configuration."""
    data_dir: Path = DEFAULT_DATA_DIR
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    default_session: str = "default"

    @classmethod
    def load(cls, config_path: Path | None = None) -> MemGraphConfig:
        """Load config from TOML file, falling back to defaults."""
        path = config_path or (DEFAULT_DATA_DIR / "config.toml")
        if path.exists():
            with open(path, "rb") as f:
                data = tomli.load(f)
            return cls(**data)
        return cls()

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "short_term").mkdir(exist_ok=True)
        (self.data_dir / "episodic").mkdir(exist_ok=True)
        (self.data_dir / "longterm").mkdir(exist_ok=True)
        (self.data_dir / "embeddings").mkdir(exist_ok=True)
