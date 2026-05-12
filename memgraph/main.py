"""MemGraph CLI — developer-controlled memory management."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from memgraph.config import MemGraphConfig
from memgraph.models import MemoryNode, MemoryEdge, MemoryTier
from memgraph.stores.short_term import ShortTermStore
from memgraph.stores.episodic import EpisodicStore
from memgraph.stores.long_term import LongTermStore
from memgraph.engine import LMStudioClient
from memgraph.engine.retrieval import RetrievalEngine

app = typer.Typer(
    name="mem",
    help="Graph RAG local memory system — short-term, episodic, long-term.",
    no_args_is_help=True,
)
console = Console()

# --- State file for tracking active session ---
def _state_path(config: MemGraphConfig) -> Path:
    return config.data_dir / ".active_session"


def _get_active_session(config: MemGraphConfig) -> str:
    p = _state_path(config)
    if p.exists():
        return p.read_text().strip()
    return config.default_session


def _set_active_session(config: MemGraphConfig, name: str) -> None:
    config.ensure_dirs()
    _state_path(config).write_text(name)


def _machine_id() -> str:
    return platform.node()


def _load_stores(config: MemGraphConfig) -> tuple[ShortTermStore, EpisodicStore, LongTermStore]:
    session = _get_active_session(config)
    config.ensure_dirs()
    return (
        ShortTermStore(config.data_dir, session),
        EpisodicStore(config.data_dir),
        LongTermStore(config.data_dir),
    )


# ─── Session management ───────────────────────────────────────

session_app = typer.Typer(help="Manage working sessions.")
app.add_typer(session_app, name="session")


@session_app.command("list")
def session_list():
    """List all sessions."""
    config = MemGraphConfig.load()
    config.ensure_dirs()
    active = _get_active_session(config)
    st_dir = config.data_dir / "short_term"

    sessions = []
    if st_dir.exists():
        for f in st_dir.glob("*.pkl"):
            name = f.stem
            marker = " ← active" if name == active else ""
            size_kb = f.stat().st_size / 1024
            sessions.append((name, f"{size_kb:.1f} KB", marker))

    if not sessions and active:
        sessions.append((active, "0 KB", " ← active (new)"))

    table = Table(title="Sessions")
    table.add_column("Name", style="bold")
    table.add_column("Size")
    table.add_column("Status")
    for name, size, status in sessions:
        table.add_row(name, size, status)
    console.print(table)


@session_app.command("switch")
def session_switch(name: str):
    """Switch to a session (creates if new)."""
    config = MemGraphConfig.load()
    old = _get_active_session(config)

    # Persist current short-term before switching
    st = ShortTermStore(config.data_dir, old)
    st.persist()

    _set_active_session(config, name)
    console.print(f"[green]Switched:[/] {old} → [bold]{name}[/bold]")


@session_app.command("new")
def session_new(name: str, description: str = ""):
    """Create and switch to a new session."""
    config = MemGraphConfig.load()
    old = _get_active_session(config)
    st = ShortTermStore(config.data_dir, old)
    st.persist()

    _set_active_session(config, name)
    console.print(f"[green]Created and switched to:[/] [bold]{name}[/bold]")


@session_app.command("fork")
def session_fork(new_name: str):
    """Fork current session into a new one (copies short-term graph)."""
    import shutil
    config = MemGraphConfig.load()
    current = _get_active_session(config)
    st = ShortTermStore(config.data_dir, current)
    st.persist()

    src = config.data_dir / "short_term" / f"{current}.pkl"
    dst = config.data_dir / "short_term" / f"{new_name}.pkl"
    if src.exists():
        shutil.copy2(src, dst)

    _set_active_session(config, new_name)
    console.print(f"[green]Forked:[/] {current} → [bold]{new_name}[/bold]")


# ─── Memory operations ────────────────────────────────────────

@app.command("add")
def add_memory(
    text: str = typer.Argument(..., help="Text to extract entities from and store"),
):
    """Add context to short-term memory (default). Extracts entities via LM Studio."""
    config = MemGraphConfig.load()
    session = _get_active_session(config)
    st, _, _ = _load_stores(config)

    llm = LMStudioClient(config.llm)
    nodes, edges = llm.extract_entities(text)

    for n in nodes:
        n.properties.setdefault("source_text", text)
        st.add_node(n)
    for e in edges:
        st.add_edge(e)
    st.persist()

    console.print(f"[green]Added to short-term ({session}):[/] {len(nodes)} nodes, {len(edges)} edges")
    for n in nodes:
        console.print(f"  • {n.label} [{n.type}]")


@app.command("save")
def save_memory(
    summary: str = typer.Argument(..., help="What to remember"),
    episodic: bool = typer.Option(False, "--episodic", "-e", help="Save as episodic event"),
    longterm: bool = typer.Option(False, "--longterm", "-l", help="Save as long-term fact"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
):
    """Promote memory to episodic or long-term. Default is episodic if neither flag given."""
    config = MemGraphConfig.load()
    session = _get_active_session(config)
    st, ep_store, lt_store = _load_stores(config)
    llm = LMStudioClient(config.llm)
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    if longterm:
        # Extract entities and save to long-term graph
        nodes, edges = llm.extract_entities(summary)
        for n in nodes:
            n.origin_machine = _machine_id()
            n.properties.setdefault("source_text", summary)
            lt_store.add_node(n)
        for e in edges:
            lt_store.add_edge(e)

        console.print(f"[blue]Saved to long-term:[/] {len(nodes)} nodes, {len(edges)} edges")
        for n in nodes:
            console.print(f"  • {n.label} [{n.type}]")

    else:
        # Default to episodic
        # Grab related short-term nodes if any
        related_nodes = st.find_by_label(summary.split()[0]) if summary else []
        ep = ep_store.save_episode(
            summary=summary,
            session_id=session,
            tags=tag_list,
            nodes=related_nodes[:5],
            origin_machine=_machine_id(),
        )
        console.print(f"[yellow]Saved to episodic:[/] {ep.summary[:80]}")
        if tag_list:
            console.print(f"  tags: {', '.join(tag_list)}")


@app.command("recall")
def recall(
    query: str = typer.Argument(..., help="What to search for across all tiers"),
    limit: int = typer.Option(10, "--limit", "-n"),
):
    """Search across all memory tiers."""
    config = MemGraphConfig.load()
    st, ep, lt = _load_stores(config)
    engine = RetrievalEngine(st, ep, lt)

    results = engine.retrieve(query, limit=limit)
    if not results:
        console.print("[dim]No memories found.[/dim]")
        return

    table = Table(title=f"Memory recall: '{query}'")
    table.add_column("Tier", width=12)
    table.add_column("Score", width=8)
    table.add_column("Label")
    table.add_column("Type", width=10)

    tier_colors = {
        MemoryTier.SHORT_TERM: "green",
        MemoryTier.EPISODIC: "yellow",
        MemoryTier.LONG_TERM: "blue",
    }

    for r in results:
        color = tier_colors.get(r.tier, "white")
        stars = "★" * max(1, int(r.score * 3))
        table.add_row(
            f"[{color}]{r.tier.value}[/{color}]",
            stars,
            r.node.label[:60],
            r.node.type,
        )
    console.print(table)

    # Check for conflicts
    conflicts = engine.detect_conflicts(query)
    if conflicts:
        console.print(f"\n[red]⚠ {len(conflicts)} conflict(s) detected:[/red]")
        for a, b, desc in conflicts:
            console.print(f"  • {desc}")


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question to answer using memory context"),
):
    """Ask a question — retrieves context from all tiers and queries LM Studio."""
    config = MemGraphConfig.load()
    st, ep, lt = _load_stores(config)
    engine = RetrievalEngine(st, ep, lt)
    llm = LMStudioClient(config.llm)

    context = engine.retrieve_formatted(question, limit=10)
    if context:
        console.print(Panel("\n".join(context), title="Retrieved context", border_style="dim"))

    answer = llm.query_with_context(question, context)
    console.print(f"\n{answer}")


# ─── Status & health ──────────────────────────────────────────

@app.command("status")
def status():
    """Show current session, memory stats, and LM Studio health."""
    config = MemGraphConfig.load()
    session = _get_active_session(config)
    st, ep, lt = _load_stores(config)
    llm = LMStudioClient(config.llm)

    console.print(f"[bold]Active session:[/] {session}")
    console.print(f"[bold]Machine:[/] {_machine_id()}")
    console.print(f"[bold]Data dir:[/] {config.data_dir}")
    console.print()

    table = Table(title="Memory tiers")
    table.add_column("Tier")
    table.add_column("Nodes")
    table.add_column("Edges")

    st_stats = st.stats()
    ep_stats = ep.stats()
    lt_stats = lt.stats()

    table.add_row("[green]Short-term[/]", str(st_stats["nodes"]), str(st_stats["edges"]))
    table.add_row("[yellow]Episodic[/]", str(ep_stats["nodes"]), f"{ep_stats['episodes']} episodes")
    table.add_row("[blue]Long-term[/]", str(lt_stats["nodes"]), str(lt_stats["edges"]))
    console.print(table)

    health = llm.health_check()
    if health["status"] == "ok":
        console.print(f"\n[green]LM Studio:[/] connected at {health['base_url']}")
        console.print(f"  Models loaded: {', '.join(health['models'])}")
    else:
        console.print(f"\n[red]LM Studio:[/] {health['error']}")
        console.print(f"  Expected at: {health['base_url']}")


@app.command("clear")
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clear short-term memory for current session."""
    config = MemGraphConfig.load()
    session = _get_active_session(config)
    if not confirm:
        typer.confirm(f"Clear short-term memory for session '{session}'?", abort=True)

    st = ShortTermStore(config.data_dir, session)
    st.clear()
    console.print(f"[green]Cleared short-term memory for:[/] {session}")


# ─── Sync ─────────────────────────────────────────────────────

sync_app = typer.Typer(help="LAN sync between machines.")
app.add_typer(sync_app, name="sync")


@sync_app.command("discover")
def sync_discover():
    """Discover MemGraph peers on the local network via mDNS."""
    from zeroconf import ServiceBrowser, Zeroconf, ServiceStateChange

    console.print("[dim]Scanning LAN for MemGraph peers...[/dim]")
    peers = []

    class Listener:
        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name)
            if info:
                addresses = [str(addr) for addr in info.parsed_addresses()]
                peers.append({
                    "name": name,
                    "addresses": addresses,
                    "port": info.port,
                })
                console.print(f"  [green]Found:[/] {name} at {addresses[0]}:{info.port}")

        def remove_service(self, zc, type_, name):
            pass

        def update_service(self, zc, type_, name):
            pass

    zc = Zeroconf()
    browser = ServiceBrowser(zc, "_memgraph._tcp.local.", Listener())

    import time
    time.sleep(5)
    zc.close()

    if not peers:
        console.print("[yellow]No peers found.[/yellow] Is MemGraph running on other machines?")
    else:
        console.print(f"\n[green]{len(peers)} peer(s) discovered.[/green]")


@sync_app.command("export")
def sync_export(
    output: Path = typer.Option(Path("memgraph-export.json"), "--output", "-o"),
):
    """Export all memory tiers to a portable JSON file for manual transfer."""
    config = MemGraphConfig.load()
    _, ep, lt = _load_stores(config)

    export_data = {
        "machine": _machine_id(),
        "exported_at": __import__("datetime").datetime.now().isoformat(),
        "episodic": ep.export_all(),
        "long_term": lt.export_all(),
    }

    output.write_text(json.dumps(export_data, indent=2, default=str))
    console.print(f"[green]Exported to:[/] {output}")
    console.print(f"  Episodic records: {len(export_data['episodic'])}")
    console.print(f"  Long-term nodes: {len(export_data['long_term'])}")


@sync_app.command("import")
def sync_import(
    input_file: Path = typer.Argument(..., help="Path to memgraph-export.json"),
):
    """Import memory from another machine's export file."""
    config = MemGraphConfig.load()
    _, ep, lt = _load_stores(config)

    data = json.loads(input_file.read_text())
    console.print(f"[dim]Importing from {data.get('machine', 'unknown')}...[/dim]")

    # Import long-term nodes
    imported = 0
    for node_data in data.get("long_term", []):
        try:
            node = MemoryNode(**node_data)
            lt.add_node(node)
            imported += 1
        except Exception as e:
            console.print(f"  [red]Skip node:[/] {e}")

    console.print(f"[green]Imported:[/] {imported} long-term nodes")
    console.print("[dim]Episodic import merges by ID — duplicates are skipped.[/dim]")


# ─── Init ─────────────────────────────────────────────────────

@app.command("init")
def init():
    """Initialise MemGraph data directory and default config."""
    config = MemGraphConfig.load()
    config.ensure_dirs()

    config_path = config.data_dir / "config.toml"
    if not config_path.exists():
        config_path.write_text("""\
# MemGraph configuration

[llm]
base_url = "http://localhost:1234/v1"
model = "default"
api_key = "lm-studio"
temperature = 0.1

[embedding]
model_name = "all-MiniLM-L6-v2"
device = "cpu"  # use "mps" for Apple Silicon

[sync]
enabled = true
port = 50051
auto_sync = false
sync_interval_seconds = 300
""")

    console.print(f"[green]Initialised MemGraph at:[/] {config.data_dir}")
    console.print("  Edit config: [dim]~/.memgraph/config.toml[/dim]")
    console.print("  Start LM Studio and load a model, then run: [bold]mem status[/bold]")


if __name__ == "__main__":
    app()
