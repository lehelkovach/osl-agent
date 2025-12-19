# osl-agent

Prototype knowledge/working-memory primitives for a lightweight agent graph. The library models immutable prototypes, instances, and associations; applies a small NLP heuristic to turn natural-language instructions into `task` or `event` concepts; and maintains a NetworkX-backed working-memory layer that reinforces edges as they reactivate. An async replicator can stream those reinforcements to a long-term store (Arango by default, but pluggable).

## Why this exists
- Build an easily inspectable “small brain” for an agent without needing a full KG stack.
- Keep the schema bootstrapping minimal: immutable prototypes + instances + associations.
- Experiment with short-term working memory that can later be persisted.
- Provide a thin ArangoDB adapter while remaining runnable with no external services.

## Core pieces
- `src/knowshowgo/models.py` — immutable prototypes, nodes, associations, and core prototype seeds.
- `src/knowshowgo/graph.py` — in-memory prototype + instance registry (UUID-based).
- `src/knowshowgo/inference.py` — classifies NL instructions into `task` vs `event`, extracts light event fields, and links them into the graph and working memory.
- `src/knowshowgo/working_memory.py` — NetworkX working-memory graph with reinforcement and caps.
- `src/knowshowgo/replication.py` — async queue that forwards edge-weight updates to a `GraphClient`.
- `src/knowshowgo/arangodb_client.py` — minimal Arango wrapper; safe to import without Arango installed.
- `src/knowshowgo/api.py` — ORM-like façade over Arango + in-memory models (prototype/node/association helpers).
- `tests/` — coverage for inference, working-memory reinforcement, and replication flushing.

## Quick start (Python 3.11+)
Using Poetry:
```bash
poetry install
poetry run pytest
```

Using vanilla venv + pip (PowerShell helpers kept for Windows users):
```powershell
.\setup-venv.ps1
. .\activate-and-install.ps1    # dot-source to stay in the venv
pytest
```

## Usage examples
Parse and store a simple reminder with working-memory reinforcement:
```python
from knowshowgo.graph import KnowledgeGraph
from knowshowgo.inference import process_instruction
from knowshowgo.working_memory import WorkingMemoryGraph

g = KnowledgeGraph()
wm = WorkingMemoryGraph()

proto, inst = process_instruction(
    g,
    "At midnight, remind me to take out the trash.",
    working_memory=wm,
    embedding_similarity=0.8,
)

print(proto.kind)                 # "event"
print(inst.attributes["time"])    # "00:00"
print(wm.get_weight(proto.id, inst.id))
```

Stream working-memory updates to a long-term store:
```python
import asyncio
from knowshowgo.replication import AsyncReplicator, EdgeUpdate
from knowshowgo.working_memory import WorkingMemoryGraph
from knowshowgo.graph import KnowledgeGraph
from knowshowgo.arangodb_client import ArangoGraphStore

store = ArangoGraphStore()  # no-op if python-arango is not installed
replicator = AsyncReplicator(store)
wm = WorkingMemoryGraph()
g = KnowledgeGraph()

async def main():
    await replicator.start()
    proto, inst = process_instruction(
        g,
        "Remind me at noon to ping the team.",
        working_memory=wm,
        replicator=replicator,
    )
    await asyncio.wait_for(replicator.queue.join(), timeout=1.0)
    await replicator.stop()

asyncio.run(main())
```

## ArangoDB notes
- Default collection names: `nodes`, `prototypes`, `associations`, `embeddings`.
- If `python-arango` is missing, the wrapper becomes a no-op so the rest of the code and tests still run.
- Real persistence logic (e.g., AQL updates for weights) can be added inside `ArangoGraphStore.increment_edge_weight`.

## Motivation and use cases
- Rapidly prototype agent “memory” experiments without heavy infra.
- Capture NL instructions as structured nodes for later planning or retrieval.
- Keep a short-term reinforcement signal that can be streamed or discarded.
- Provide a pluggable backend so teams can slot in their own `GraphClient`.

## Roadmap / TODO
- Flesh out Arango weight update with proper AQL and indexes.
- Add decay/forgetting to the working-memory graph.
- Expand inference beyond the simple `task`/`event` heuristic and add entity extraction.
- Provide more retrieval helpers (top-K from working memory, context-conditioned activation).
- Add docs and examples for swapping in a different `GraphClient` implementation.

## Development
- Run `pytest` (async-ready via `pytest-asyncio`).
- Code is pure Python 3.11; uses `networkx` and standard library only by default.
- No external services are required to run tests or examples.
