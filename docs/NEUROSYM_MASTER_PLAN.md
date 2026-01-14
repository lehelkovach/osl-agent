# NeuroSym.js: Design & Development Master Plan

**Version:** 2.0.0 (Integration Ready)
**Target Stack:** Node.js (Universal) | ArangoDB | OpenAI (LLM)
**Project Goal:** To transform KnowShowGo from a static knowledge graph into a computable, differentiable logic engine using a parsimonious neurosymbolic architecture.

---

## 1. Overview & Philosophy

**NeuroSym.js** solves the "Engineering Gap" between rigid symbolic logic (Datalog) and opaque neural networks. It treats **Code as Data**:

* **Nodes** are Variables (State).
* **Edges** are Functions (Rules).
* **The Engine** is a Processor that runs "Fuzzy Diffusion" on the graph.

### Key Capabilities

1. **Hybrid Logic:** Supports both **Strict** (Boolean/Digital) and **Fuzzy** (Probabilistic) logic in the same graph.
2. **Argumentation:** Handles contradictions natively via "Attack" edges (Defeasible Reasoning).
3. **Parsimony:** Zero-dependency JavaScript engine. No heavy Python/PyTorch requirements.
4. **Transpilation:** Can be instantly converted to Natural Language, Python, or Logic Proofs via LLMs.

---

## 2. Data Protocol: The NeuroDAG Schema

This JSON structure is the "Machine Language" of the system. All data stored in ArangoDB must be mappable to this format at runtime.

```json
{
  "graph": {
    "id": "context_123",
    "name": "Server Risk Model",
    "mode": "hybrid"
  },
  
  "nodes": [
    {
      "id": "fact:server_down",
      "type": "DIGITAL",
      "content": { "text": "Server is offline" },
      "state": { 
        "truth": 1.0,
        "prior": 0.5,
        "is_locked": true
      }
    },
    {
      "id": "risk:churn",
      "type": "FUZZY",
      "content": { "text": "Churn Risk" },
      "state": { "truth": 0.5, "prior": 0.2 }
    }
  ],

  "edges": [
    {
      "id": "rule:1",
      "source": "fact:server_down",
      "target": "risk:churn",
      "type": "IMPLICATION",
      "weight": 0.9,
      "op": "IDENTITY"
    },
    {
      "id": "rule:2",
      "source": "fact:has_backup",
      "target": "risk:churn",
      "type": "ATTACK",
      "weight": 1.0
    }
  ]
}
```

### Node Types

| Type | Behavior | Use Case |
|------|----------|----------|
| `DIGITAL` | Snaps to 0 or 1 (threshold 0.5) | Binary facts, flags |
| `FUZZY` | Continuous [0.0, 1.0] | Probabilities, confidence scores |

### Edge Types

| Type | Formula | Description |
|------|---------|-------------|
| `IMPLICATION` | `target += src * weight` | Standard logical flow |
| `ATTACK` | `target *= (1 - attacker * weight)` | Inhibition/defeat |
| `SUPPORT` | `target += (1 - target) * src * weight` | Reinforcement |

---

## 3. Module 1: The Engine (`neurosym.js`)

This core library implements the mathematical solver. It is stateless and pure.

### Implementation Status: ✅ Complete

**JavaScript Location:** `neurosym-js/src/engine.ts`
**Python Location:** `src/knowshowgo/neuro/engine.py`

```javascript
/**
 * NeuroSym Engine v2.0
 * Implements Lukasiewicz Logic & Argumentation Frames
 */
class NeuroEngine {
  constructor(schema) {
    this.nodes = new Map();
    this.edges = schema.edges;
    this.init(schema.nodes);
  }

  init(nodes) {
    nodes.forEach(n => {
      this.nodes.set(n.id, { ...n, incoming: [] });
    });
    this.edges.forEach(e => {
      if (this.nodes.has(e.target)) {
        this.nodes.get(e.target).incoming.push(e);
      }
    });
  }

  /**
   * Core Logic Gates (Lukasiewicz T-Norms)
   */
  ops = {
    // Fuzzy OR (Max-Sum): Strongest evidence wins
    IDENTITY: (vals, w) => Math.max(...vals) * w, 
    // Fuzzy AND (Lukasiewicz T-Norm): Strict requirement
    AND: (vals, w) => Math.max(0, vals.reduce((a, b) => a + b, 0) - (vals.length - 1)) * w,
    // Negation/Attack: Inhibits truth
    INHIBIT: (target, attacker, w) => target * (1.0 - (attacker * w))
  };

  /**
   * The Solver Loop (Forward Propagation)
   */
  run(evidence = {}, iterations = 5) {
    // 1. Inject Evidence (Override Priors)
    Object.entries(evidence).forEach(([id, val]) => {
      if (this.nodes.has(id)) {
        const n = this.nodes.get(id);
        n.state.truth = val;
        n.state.is_locked = true;
      }
    });

    // 2. Diffusion Cycles
    for (let i = 0; i < iterations; i++) {
      this.step();
    }

    return this.exportState();
  }

  step() {
    this.nodes.forEach(node => {
      if (node.state.is_locked) return;

      const supporters = node.incoming.filter(e => e.type === 'IMPLICATION');
      const attackers = node.incoming.filter(e => e.type === 'ATTACK');

      // Calculate Support
      let newTruth = node.state.prior;
      if (supporters.length > 0) {
        const supportVals = supporters.map(e => {
           const srcVal = this.nodes.get(e.source).state.truth;
           return srcVal * e.weight;
        });
        newTruth = Math.max(newTruth, ...supportVals);
      }

      // Calculate Attack (Inhibition)
      let inhibition = 0.0;
      if (attackers.length > 0) {
        attackers.forEach(e => {
          const atkVal = this.nodes.get(e.source).state.truth;
          inhibition = Math.max(inhibition, atkVal * e.weight);
        });
      }

      // Apply Inhibition
      newTruth = newTruth * (1.0 - inhibition);

      // Digital Collapse (Strict Mode)
      if (node.type === 'DIGITAL') {
        newTruth = newTruth > 0.5 ? 1.0 : 0.0;
      }

      node.state.truth = newTruth;
    });
  }

  exportState() {
    const res = {};
    this.nodes.forEach((n, id) => res[id] = n.state.truth);
    return res;
  }
}

module.exports = NeuroEngine;
```

---

## 4. Module 2: The Transpiler (LLM Integration)

This module translates the NeuroDAG JSON into Natural Language explanations or Executable Code.

### Implementation Status: 🔲 Planned

### System Prompt Template

**File:** `assets/prompts/neuro_transpile.md`

```markdown
# SYSTEM INSTRUCTION: NeuroSym Transpiler

You are the compiler for NeuroSym. Accept a JSON NeuroDAG and transpile it.

## INPUT: NeuroDAG JSON Schema v2.0
- Nodes: { id, type ("DIGITAL"|"FUZZY"), content }
- Edges: { source, target, type ("IMPLICATION"|"ATTACK"), weight }

## TARGET FORMATS
1. **NATURAL_LANGUAGE**: Summarize the logic. "X implies Y with 90% confidence, but Z prevents it."
2. **PREDICATE_LOGIC**: Output LaTeX. Use \to for implication, \neg for attacks.
3. **CODE_PYTHON**: Generate a standalone function `def solve(state):` implementing the math logic.

## MATH RULES FOR CODE GENERATION
1. Implication: `val = src * weight`
2. Aggregation: `val = max(val1, val2)`
3. Attack: `final = val * (1.0 - (attacker * weight))`
4. Digital: `final = 1.0 if final > 0.5 else 0.0`
```

### Transpiler Service

**File:** `src/libs/neurosym/transpiler.js`

```javascript
const fs = require('fs');
const { OpenAI } = require('openai');

class NeuroTranspiler {
  constructor(apiKey) {
    this.llm = new OpenAI({ apiKey });
    this.systemPrompt = fs.readFileSync('./assets/prompts/neuro_transpile.md', 'utf-8');
  }

  async transpile(neuroGraph, targetMode) {
    const userMessage = `TARGET: ${targetMode}\nJSON:\n${JSON.stringify(neuroGraph, null, 2)}`;
    
    const response = await this.llm.chat.completions.create({
      model: "gpt-4-turbo",
      messages: [
        { role: "system", content: this.systemPrompt },
        { role: "user", content: userMessage }
      ],
      temperature: 0.0
    });
    return response.choices[0].message.content;
  }
}

module.exports = NeuroTranspiler;
```

### Target Formats

| Format | Output | Use Case |
|--------|--------|----------|
| `NATURAL_LANGUAGE` | Human-readable explanation | UI tooltips, reports |
| `PREDICATE_LOGIC` | LaTeX formulas | Academic papers, documentation |
| `CODE_PYTHON` | Executable Python function | Integration with ML pipelines |

---

## 5. Module 3: KnowShowGo (ArangoDB) Integration

### Implementation Status: ✅ Complete (Python), 🔲 Planned (ArangoDB AQL)

### A. Data Model Updates (Arango Collections)

**Concepts Collection:**
```json
{
  "_key": "server_down",
  "label": "Server is offline",
  "neuro": {
    "mode": "DIGITAL",
    "truth": 0.5,
    "prior": 0.5,
    "is_locked": false
  }
}
```

**Associations Collection:**
```json
{
  "_from": "concepts/server_down",
  "_to": "concepts/churn_risk",
  "neuro": {
    "type": "IMPLICATION",
    "weight": 0.9,
    "op": "IDENTITY"
  }
}
```

### B. The Service Layer

**File:** `src/services/NeuroService.js`

```javascript
import { db } from '../database'; 
import NeuroEngine from '../libs/neurosym/engine';

class NeuroService {
  
  /**
   * 1. AQL Traversal: Fetch Subgraph
   */
  async fetchContext(contextId) {
    const query = `
      FOR v, e, p IN 1..2 OUTBOUND @startNode GRAPH 'knowshowgo'
      RETURN { node: v, edge: e }
    `;
    const cursor = await db.query(query, { startNode: `concepts/${contextId}` });
    return this.mapToSchema(await cursor.all());
  }

  /**
   * 2. Mapper: ArangoDB -> NeuroDAG JSON
   */
  mapToSchema(rows) {
    const nodes = [], edges = [], seen = new Set();
    
    rows.forEach(({ node, edge }) => {
      if (!seen.has(node._key)) {
        nodes.push({
          id: node._key,
          type: node.neuro?.mode || 'FUZZY',
          state: { 
            truth: node.neuro?.truth ?? 0.5, 
            prior: node.neuro?.prior ?? 0.5 
          },
          content: { text: node.label }
        });
        seen.add(node._key);
      }
      if (edge) {
        edges.push({
          source: edge._from.split('/')[1],
          target: edge._to.split('/')[1],
          type: edge.neuro?.type || 'IMPLICATION',
          weight: edge.neuro?.weight ?? 1.0
        });
      }
    });
    return { nodes, edges };
  }

  /**
   * 3. Execution Endpoint
   */
  async solve(contextId, evidence) {
    const schema = await this.fetchContext(contextId);
    const engine = new NeuroEngine(schema);
    return engine.run(evidence);
  }
}

module.exports = new NeuroService();
```

### C. Python Integration (Current Implementation)

**File:** `src/knowshowgo/neuro_service.py`

```python
from knowshowgo.neuro import NeuroEngine
from knowshowgo.models import Node, Association

class NeuroService:
    async def solve_context(self, center_node_id, depth=2, evidence=None):
        # 1. Fetch Subgraph
        context = await self.fetch_context(center_node_id, depth)
        
        # 2. Convert to NeuroJSON
        schema = self.to_neuro_json(context)
        
        # 3. Run Engine
        engine = NeuroEngine(schema)
        results = engine.run(evidence)
        
        # 4. Write Back to DB
        await self.db.bulk_update_nodes(results)
        
        return results
```

---

## 6. Development Roadmap

### Phase 1: Core Library ✅
- [x] Implement `neurosym-js/src/engine.ts` (TypeScript)
- [x] Implement `src/knowshowgo/neuro/engine.py` (Python)
- [x] Unit tests for Lukasiewicz math
- [x] 163 JS tests, 22 Python tests

### Phase 2: Data Model ✅
- [x] Update KSG Node schema with `truth_value`, `prior`, `is_locked`
- [x] Update KSG Association schema with `logic_meta`
- [x] Create LogicType and LogicMeta classes

### Phase 3: Service Layer ✅
- [x] Implement NeuroService (Python)
- [x] Context extraction and NeuroJSON conversion
- [x] Local inference without DB

### Phase 4: Database Integration 🔲
- [ ] Update ArangoDB seed/schema with `neuro` metadata fields
- [ ] Implement AQL queries for subgraph traversal
- [ ] Implement bulk update for truth values

### Phase 5: API Layer 🔲
- [ ] Expose `POST /api/neuro/solve` endpoint
- [ ] Add evidence injection via request body
- [ ] Return explained results

### Phase 6: Transpiler 🔲
- [ ] Create LLM prompt templates
- [ ] Implement NeuroTranspiler service
- [ ] Support NATURAL_LANGUAGE, PREDICATE_LOGIC, CODE_PYTHON

### Phase 7: UI Integration 🔲
- [ ] Add "Truth Sliders" for evidence injection
- [ ] Add "Logic Explanations" panel
- [ ] Visualize node opacity based on truth value

---

## 7. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NeuroSym.js Ecosystem                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Frontend   │    │   REST API   │    │   ArangoDB   │                   │
│  │  (React/Vue) │◄──►│  (Express)   │◄──►│   (Graph)    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                    │                           │
│         │                   ▼                    │                           │
│         │         ┌──────────────────┐           │                           │
│         │         │   NeuroService   │◄──────────┘                           │
│         │         │  (Orchestrator)  │                                       │
│         │         └────────┬─────────┘                                       │
│         │                  │                                                 │
│         │    ┌─────────────┼─────────────┐                                   │
│         │    ▼             ▼             ▼                                   │
│         │ ┌──────┐   ┌──────────┐   ┌──────────┐                            │
│         │ │Mapper│   │NeuroEngine│   │Transpiler│                           │
│         │ │      │   │ (Solver) │   │  (LLM)   │                            │
│         │ └──────┘   └──────────┘   └──────────┘                            │
│         │                │                │                                  │
│         │                ▼                ▼                                  │
│         │         ┌────────────┐   ┌────────────┐                           │
│         └────────►│   State    │   │ Explanation│                           │
│                   │  Updates   │   │   Output   │                           │
│                   └────────────┘   └────────────┘                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Quick Reference

### JavaScript API

```javascript
import { NeuroEngine } from 'neurosym';

const ai = new NeuroEngine(schema);
const result = ai.run({ fact_a: 1.0 }, 5);  // evidence, iterations
const trained = ai.train(examples, 100);     // data, epochs
const exported = ai.export();                // save learned weights
```

### Python API

```python
from knowshowgo.neuro import NeuroEngine

engine = NeuroEngine(schema)
result = engine.run({"fact_a": 1.0})
engine.train(examples, epochs=100)
exported = engine.export()
```

### NeuroService API

```python
from knowshowgo.neuro_service import NeuroService

service = NeuroService(db)
results = await service.solve_context(node_id, depth=2, evidence={"x": 1.0})
```

---

## 9. Files & Locations

| Component | JavaScript | Python |
|-----------|------------|--------|
| Engine | `neurosym-js/src/engine.ts` | `src/knowshowgo/neuro/engine.py` |
| Logic Core | `neurosym-js/src/logic-core.ts` | `src/knowshowgo/neuro/logic.py` |
| Types | `neurosym-js/src/types.ts` | `src/knowshowgo/neuro/types.py` |
| Service | `neurosym-js/src/engine.ts` | `src/knowshowgo/neuro_service.py` |
| Tests | `neurosym-js/tests/` | `tests/test_neuro_service.py` |
| Docs | `neurosym-js/README.md` | `docs/KSG_INTEGRATION.md` |

---

## 10. References

- [Logical Neural Networks (IBM)](https://arxiv.org/abs/2006.13155)
- [Lukasiewicz Logic](https://en.wikipedia.org/wiki/Łukasiewicz_logic)
- [Abstract Argumentation](https://en.wikipedia.org/wiki/Argumentation_framework)
- [ArangoDB Graph Queries](https://www.arangodb.com/docs/stable/aql/graphs.html)
