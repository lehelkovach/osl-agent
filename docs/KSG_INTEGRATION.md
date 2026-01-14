# Integration Guide: NeuroSym x KnowShowGo

## 1. Context & Objective

We have integrated **NeuroSym** reasoning capabilities into the **KnowShowGo (KSG)** service.

* **Previous State:** KSG was a graph knowledge base storing static nodes.
* **Current State:** KSG is now a **Live Inference Engine**.
    * Nodes represent **Fuzzy Concepts** (Truth: 0.0-1.0).
    * Edges represent **Logic Rules** (Implications/Attacks) with stored logic metadata.
    * The system can "solve" queries by propagating truth values across the graph.

---

## 2. Data Model Changes

### 2.1 Node Schema Updates

The `Node` class now includes neurosymbolic fields:

```python
@dataclass
class Node:
    # ... existing fields ...
    
    # Neurosymbolic fields
    truth_value: float = 0.5  # Current fuzzy state [0.0 - 1.0]
    prior: float = 0.5        # Base probability before evidence
    is_locked: bool = False   # If TRUE, value is evidence (immutable during inference)
```

### 2.2 Association Schema Updates

The `Association` class now supports logic metadata:

```python
class LogicType:
    IMPLIES = "IMPLIES"    # A -> B: if A then B
    ATTACKS = "ATTACKS"    # A defeats B (argumentation)
    SUPPORTS = "SUPPORTS"  # A reinforces B
    DEPENDS = "DEPENDS"    # A depends on B
    MUTEX = "MUTEX"        # Mutual exclusion

@dataclass
class LogicMeta:
    type: str = LogicType.IMPLIES
    weight: float = 1.0         # Confidence [0.0 - 1.0]
    op: str = LogicOp.IDENTITY  # AND, OR, IDENTITY
    learnable: bool = True

@dataclass
class Association:
    # ... existing fields ...
    logic_meta: Optional[LogicMeta] = None
```

Helper methods for common patterns:

```python
# Create implication: Penguin -> Bird (All Penguins are Birds)
assoc = Association.create_implies(
    source_id=penguin_id,
    target_id=bird_id,
    weight=1.0,
)

# Create attack: Penguin attacks Fly (Penguins inhibit Flying)
assoc = Association.create_attacks(
    source_id=penguin_id,
    target_id=fly_id,
    weight=1.0,
)
```

---

## 3. Python NeuroSym Module

A Python port of NeuroSym.js is now available at `src/knowshowgo/neuro/`:

### 3.1 Logic Core

```python
from knowshowgo.neuro import fuzzy_and, fuzzy_or, fuzzy_not, implies, inhibit, support

# Lukasiewicz logic
fuzzy_and(0.8, 0.9)  # 0.7 (max(0, sum - n + 1))
fuzzy_or(0.3, 0.4)   # 0.7 (min(1, sum))
fuzzy_not(0.3)       # 0.7 (1 - input)
implies(0.8, 0.5)    # 0.7 (min(1, 1 - a + b))

# Argumentation
inhibit(0.8, 1.0, 0.5)  # 0.4 (target reduced by attack)
support(0.5, 1.0, 0.5)  # 0.75 (target increased)
```

### 3.2 NeuroEngine

```python
from knowshowgo.neuro import NeuroEngine

schema = {
    "version": "1.0",
    "variables": {
        "raining": {"type": "bool", "prior": 0.3},
        "wet_ground": {"type": "bool", "prior": 0.1},
    },
    "rules": [{
        "id": "rain_wets",
        "type": "IMPLICATION",
        "inputs": ["raining"],
        "output": "wet_ground",
        "op": "IDENTITY",
        "weight": 0.95,
    }],
    "constraints": [],
}

engine = NeuroEngine(schema)
result = engine.run({"raining": 1.0})
print(result["wet_ground"])  # ~0.95
```

---

## 4. NeuroService

The `NeuroService` bridges KSG nodes/associations with NeuroSym reasoning.

### 4.1 Basic Usage

```python
from knowshowgo.neuro_service import NeuroService, run_local_inference

# With database adapter
service = NeuroService(db)
results = await service.solve_context(center_node_id, depth=2)

# Without database (local inference)
results = run_local_inference(
    nodes=[node_a, node_b],
    associations=[implies_assoc],
    evidence={node_a.id: 1.0},
)
```

### 4.2 The `solve_context` Routine

```python
async def solve_context(
    center_node_id: str,
    depth: int = 2,
    evidence: Optional[Dict[str, float]] = None,
    write_back: bool = True,
) -> Dict[str, float]:
    """
    1. Fetches subgraph neighborhood (context window)
    2. Converts to NeuroJSON format
    3. Runs inference (diffusion)
    4. Optionally writes back updated truth_values to DB
    """
```

### 4.3 API Endpoint (Example)

```python
# Example FastAPI endpoint
@app.post("/api/reason/{node_id}")
async def reason(node_id: str, evidence: Dict[str, float] = None):
    service = NeuroService(db)
    results = await service.solve_context(node_id, depth=2, evidence=evidence)
    return {"results": results}
```

---

## 5. Example: Penguin Classification

```python
from knowshowgo.models import Node, Association
from knowshowgo.neuro_service import run_local_inference

# Create nodes
penguin = Node.create(prototype_id="animal", prior=0.5)
bird = Node.create(prototype_id="concept", prior=0.3)
fly = Node.create(prototype_id="ability", prior=0.5)

# Penguin -> Bird (implication)
penguin_is_bird = Association.create_implies(
    source_id=penguin.id,
    target_id=bird.id,
    weight=1.0,
)

# Penguin attacks Fly (penguins don't fly)
penguin_no_fly = Association.create_attacks(
    source_id=penguin.id,
    target_id=fly.id,
    weight=0.9,
)

# Run inference
results = run_local_inference(
    nodes=[penguin, bird, fly],
    associations=[penguin_is_bird, penguin_no_fly],
    evidence={penguin.id: 1.0},  # It IS a penguin
)

print(f"Bird: {results[bird.id]:.2f}")  # High (penguin implies bird)
print(f"Fly: {results[fly.id]:.2f}")    # Low (penguin attacks fly)
```

---

## 6. Migration Checklist

- [x] Add `truth_value`, `prior`, `is_locked` fields to Node schema
- [x] Add `logic_meta` field to Association schema
- [x] Create Python NeuroSym module (`src/knowshowgo/neuro/`)
- [x] Create NeuroService (`src/knowshowgo/neuro_service.py`)
- [x] Add tests for integration (`tests/test_neuro_service.py`)

### Next Steps for Production

- [ ] Create DB migration script for new columns
- [ ] Add REST API endpoint for reasoning
- [ ] Update frontend to visualize truth_value (e.g., node opacity)
- [ ] Add caching for frequently-accessed context windows
- [ ] Implement grounding for first-order rules

---

## 7. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     KnowShowGo + NeuroSym                        │
├──────────────────┬───────────────────┬──────────────────────────┤
│   KSG Models     │   NeuroSym Core   │   NeuroService           │
│   (Data Layer)   │   (Logic Layer)   │   (Integration Layer)    │
├──────────────────┼───────────────────┼──────────────────────────┤
│ • Node           │ • fuzzy_and/or    │ • solve_context()        │
│   - truth_value  │ • implies         │ • to_neuro_json()        │
│   - prior        │ • inhibit/support │ • run_inference()        │
│   - is_locked    │ • NeuroEngine     │ • ground_abstract_rules()│
│ • Association    │   - run()         │ • query_node()           │
│   - logic_meta   │   - train()       │                          │
│                  │   - export()      │                          │
└──────────────────┴───────────────────┴──────────────────────────┘
```
