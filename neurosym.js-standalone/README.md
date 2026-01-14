# NeuroSym.js

<p align="center">
  <strong>Lightweight Neurosymbolic AI for JavaScript</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#api">API</a> •
  <a href="#examples">Examples</a> •
  <a href="docs/DESIGN.md">Design Doc</a>
</p>

---

**NeuroSym.js** is a **zero-dependency** JavaScript/TypeScript library for **Neurosymbolic AI**. It bridges the gap between neural networks (fuzziness, learning) and symbolic logic (guarantees, explainability).

Implements **Logical Neural Networks (LNN)** using a **Parsimonious Data Protocol (NeuroJSON)**. Runs anywhere: Node.js, Browser, Edge Workers.

**Core Philosophy:** *"Code as Data"* — Logic is defined in serializable JSON, not hardcoded functions.

## ✨ Features

- **Pure Logic Core**: Lukasiewicz T-Norms for fuzzy/continuous logic
- **Graph-Based Reasoning**: In-memory graph for variables, rules, constraints
- **Inference Engine**: Forward chaining, belief propagation, constraint handling
- **Argumentation**: Attack/support relations for defeasible reasoning
- **Learnable Weights**: Train rule weights from examples
- **Serializable**: All logic in JSON (NeuroJSON protocol)
- **Zero Dependencies**: No external runtime dependencies
- **TypeScript First**: Full type safety

## 📦 Installation

```bash
npm install neurosym
```

## 🚀 Quick Start

```typescript
import { NeuroEngine } from 'neurosym';

// Define logic as JSON
const schema = {
  version: '1.0',
  variables: {
    rain: { type: 'bool', prior: 0.3 },
    wet_ground: { type: 'bool', prior: 0.1 },
    slippery: { type: 'bool', prior: 0.05 }
  },
  rules: [
    {
      id: 'rain_wets',
      type: 'IMPLICATION',
      inputs: ['rain'],
      output: 'wet_ground',
      op: 'IDENTITY',
      weight: 0.95
    },
    {
      id: 'wet_slippery',
      type: 'IMPLICATION',
      inputs: ['wet_ground'],
      output: 'slippery',
      op: 'IDENTITY',
      weight: 0.8
    }
  ],
  constraints: []
};

// Create engine
const ai = new NeuroEngine(schema);

// Run inference with evidence
const result = ai.run({ rain: 1.0 });

console.log(result.wet_ground); // ~0.95
console.log(result.slippery);   // ~0.76
```

## 📐 NeuroJSON Schema

Logic graphs are defined as JSON:

```json
{
  "version": "1.0",
  "variables": {
    "concept_a": { "type": "bool", "prior": 0.5 },
    "concept_b": { "type": "bool", "prior": 0.1 }
  },
  "rules": [
    {
      "id": "rule_1",
      "type": "IMPLICATION",
      "inputs": ["concept_a"],
      "output": "concept_b",
      "op": "IDENTITY",
      "weight": 0.9
    }
  ],
  "constraints": [
    {
      "id": "attack_1",
      "type": "ATTACK",
      "source": "concept_c",
      "target": "concept_b",
      "weight": 1.0
    }
  ]
}
```

### Rule Types
- `IMPLICATION`: A → B
- `CONJUNCTION`: A ∧ B
- `DISJUNCTION`: A ∨ B
- `EQUIVALENCE`: A ↔ B

### Constraint Types
- `ATTACK`: Source defeats target
- `SUPPORT`: Source reinforces target
- `MUTEX`: Mutual exclusion

## 🎯 Examples

### Argumentation (Pro/Con Debate)

```typescript
const debate = new NeuroEngine({
  version: '1.0',
  variables: {
    pro_argument: { type: 'bool', prior: 0.8 },
    con_argument: { type: 'bool', prior: 0.6 },
    conclusion: { type: 'bool', prior: 0.5 }
  },
  rules: [{
    id: 'pro_supports',
    type: 'IMPLICATION',
    inputs: ['pro_argument'],
    output: 'conclusion',
    op: 'IDENTITY',
    weight: 0.9
  }],
  constraints: [{
    id: 'con_attacks',
    type: 'ATTACK',
    source: 'con_argument',
    target: 'conclusion',
    weight: 0.8
  }]
});

const result = debate.run();
console.log(result.conclusion); // Balanced by attack
```

### Training (Learning Weights)

```typescript
const ai = new NeuroEngine(schema);

ai.train([
  { inputs: { feature_a: 1.0 }, targets: { outcome: 0.9 } },
  { inputs: { feature_a: 0.0 }, targets: { outcome: 0.1 } }
], 100);

const trained = ai.export();
```

### Direct Logic Operations

```typescript
import { and, or, not, implies, inhibit } from 'neurosym';

// Lukasiewicz fuzzy logic
and(0.8, 0.9);        // 0.7
or(0.3, 0.4);         // 0.7
not(0.3);             // 0.7
implies(0.8, 0.5);    // 0.7

// Argumentation
inhibit(0.8, 1.0, 0.5); // 0.4 (target reduced by attack)
```

## 📚 API

### NeuroEngine

| Method | Description |
|--------|-------------|
| `run(evidence?, iterations?)` | Run inference |
| `query(variable, evidence?)` | Query single variable |
| `train(data, epochs?)` | Train weights |
| `export()` | Export schema with weights |
| `getVariables()` | Get variable names |
| `getRules()` | Get rule IDs |

### Logic Functions

| Function | Formula |
|----------|---------|
| `and(...values)` | max(0, sum - n + 1) |
| `or(...values)` | min(1, sum) |
| `not(a)` | 1 - a |
| `implies(a, b)` | min(1, 1 - a + b) |
| `inhibit(target, attacker, w)` | target × (1 - attacker × w) |

[Full API Reference →](docs/API.md)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     NeuroSym.js                          │
├─────────────────┬──────────────────┬───────────────────┤
│   Logic Core    │   NeuroGraph     │   NeuroEngine     │
│   (Stateless)   │   (State Mgr)    │   (Main API)      │
├─────────────────┼──────────────────┼───────────────────┤
│ • T-Norms       │ • Variables      │ • run(evidence)   │
│ • NOT/AND/OR    │ • Rules          │ • train(data)     │
│ • Implication   │ • Constraints    │ • export()        │
│ • Inhibition    │ • State Tracking │ • query(var)      │
└─────────────────┴──────────────────┴───────────────────┘
```

## 🧪 Testing

```bash
npm test              # Run tests
npm run test:coverage # With coverage
```

## 🔧 Development

```bash
npm install     # Install dependencies
npm run build   # Build distribution
npm run lint    # Lint code
```

## 📖 Documentation

- [Design Document](docs/DESIGN.md) - Architecture & philosophy
- [API Reference](docs/API.md) - Complete API documentation

## 📄 License

MIT

## 🙏 Acknowledgments

Inspired by:
- [Logical Neural Networks (IBM)](https://arxiv.org/abs/2006.13155)
- [Lukasiewicz Logic](https://en.wikipedia.org/wiki/Łukasiewicz_logic)
- [Abstract Argumentation Frameworks](https://en.wikipedia.org/wiki/Argumentation_framework)
