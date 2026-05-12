# MemGraph

Developers context-switch constantly — between projects, machines, and days — and each switch costs time re-learning what was true. Where are the notes? Which Slack thread had that architecture decision? What payment provider does that app use, again?

MemGraph is a local memory system for developers. It captures working context as a graph of entities and relationships, lets you decide what sticks across sessions, and answers natural-language questions against everything you've stored — all without leaving the terminal and without sending anything to the cloud.

## The problem

When you're deep in a project you hold a lot in your head: API contracts, integration quirks, environment decisions, half-debugged bugs. That context evaporates the moment you switch to something else. Standard approaches — Notion pages, README notes, Slack threads — are fine for documentation but terrible for retrieval. You can't ask "what broke the Foodics webhook last week?" and get a useful answer.

MemGraph treats developer context the same way a graph database treats data: entities, relationships, timestamps, and confidence. Short-lived context (current session) is stored cheaply. Things worth keeping are promoted explicitly. Everything is queryable.

## How it works

Three tiers, each with a different lifespan:

| Tier | Lifespan | Storage | What belongs here |
| ---- | -------- | ------- | ----------------- |
| Short-term | Current session | NetworkX (pickle per session) | Working context, auto-captured per `mem add` |
| Episodic | Explicit save | SQLite | "This happened" — debug sessions, decisions, events |
| Long-term | Permanent | Kuzu embedded graph | "This is true" — architecture facts, integration details |

When you run `mem add "..."`, an LM Studio model extracts entities and relationships and stores them in the short-term graph for the active session. You decide what outlasts the session by running `mem save --episodic` or `mem save --longterm`. `mem recall` and `mem ask` query all three tiers simultaneously and surface conflicts when the same entity appears with different properties across tiers.

Everything lives in `~/.memgraph/`. No API keys, no subscriptions, no data leaving your machine.

## Quick start

```bash
# Install
pip install -e .

# Initialise
mem init

# Make sure LM Studio is running on localhost:1234 with a model loaded
mem status

# Start a session
mem session new myapp-api

# Add context (entities auto-extracted via LM Studio)
mem add "BrewMate uses Afterpay for BNPL payments with Tyro as EFTPOS fallback"
mem add "The Lightspeed Kounta API handles menu sync and order management"

# Everything above is short-term. Now promote what matters:
mem save --longterm "BrewMate uses Afterpay for BNPL payments, Tyro as EFTPOS fallback"
mem save --episodic "Tested Lightspeed webhook — needs retry with exponential backoff" -t "debug,lightspeed"

# Switch to another project — short-term is persisted automatically
mem session switch myanotherapp
mem add "Migrated RDS from ap-southeast-1 to ap-southeast-2, snapshot copy completed"

# Search across everything
mem recall "payment provider"
mem recall "RDS migration"

# Ask a question with full context
mem ask "What payment provider does myanotherapp app use?"

# Check status
mem status
```

## Session management

Sessions isolate short-term memory but share episodic and long-term stores. Switching sessions persists the current short-term graph to disk so nothing is lost.

```bash
mem session list                    # show all sessions
mem session new eks-migration       # create + switch
mem session switch myapp-api        # switch (persists short-term first)
mem session fork myapp-api-v2       # clone current session's short-term
```

## Syncing between machines

The entire `~/.memgraph/` directory is portable — copy it and it works.

```bash
# Export to a single JSON file
mem sync export -o ~/memgraph-backup.json

# Transfer (scp, USB, shared drive) then import
mem sync import ~/memgraph-backup.json

# Or discover peers automatically via mDNS
mem sync discover
```

## LM Studio configuration

Edit `~/.memgraph/config.toml`:

```toml
[llm]
base_url = "http://localhost:1234/v1"   # default LM Studio endpoint
model = "default"                        # uses whatever model is loaded
temperature = 0.1
```

If LM Studio is on another machine on your LAN:
```toml
base_url = "http://192.168.1.50:1234/v1"
```

## Retrieval scoring

`mem recall` and `mem ask` query all three tiers simultaneously:

- **Short-term** hits get a recency boost but lower base score (assumed transient)
- **Episodic** hits are scored by text match and temporal decay over ~1 week
- **Long-term** hits get the highest base score (trusted, stable knowledge)
- **Conflicts** are surfaced when the same entity appears in multiple tiers with different properties

## Data directory

```text
~/.memgraph/
├── config.toml
├── .active_session
├── short_term/
│   ├── myapp-api.pkl
│   └── infra-debug.pkl
├── episodic/
│   └── episodes.sqlite
├── longterm/
│   └── kuzu_db/
└── embeddings/
    └── (LanceDB files)
```

## Requirements

- Python 3.11+
- LM Studio running locally (or on LAN)
- ~500MB for sentence-transformers model (first run)

## License

MIT
