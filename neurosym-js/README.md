# NeuroSym.js

A lightweight, **zero-dependency** JavaScript/TypeScript library for **Neurosymbolic AI**. It bridges the gap between neural networks (fuzziness, learning) and symbolic logic (guarantees, explainability).

NeuroSym.js implements **Logical Neural Networks (LNN)** concepts using a **Parsimonious Data Protocol (NeuroJSON)**. It runs in any JS environment: Node.js, Browser, or Edge Workers.

**Core Philosophy:** "Code as Data" — Logic is defined in serializable JSON, not hardcoded functions.

## ✨ Features

- **Pure Logic Core**: Lukasiewicz T-Norms for fuzzy/continuous logic (truth values 0.0 to 1.0)
- **Graph-Based Reasoning**: In-memory graph structure for variables, rules, and constraints
- **Inference Engine**: Forward chaining, belief propagation, and constraint application
- **Argumentation Support**: Attack/support relations for defeasible reasoning
- **Learnable Weights**: Train rule weights from examples via heuristic gradient descent
- **Serializable**: All logic defined in JSON (NeuroJSON protocol)
- **Zero Dependencies**: No external runtime dependencies
- **TypeScript First**: Full type safety with comprehensive type definitions

## 📦 Installation

```bash
npm install neurosym
```

## 🚀 Quick Start

### Basic Usage (The Main API)

```typescript
import { NeuroEngine } from 'neurosym';
import schema from './bird-logic.json';

// Initialize
const ai = new NeuroEngine(schema);

// 1. Inference (Reasoning)
const result = ai.run({
  has_wings: 1.0,
  flies: 0.0  // It's a penguin!
});
console.log(result.is_bird); // Should handle the fuzzy conflict

// 2. Training (Learning)
ai.train([
  { inputs: { has_wings: 1.0, has_feathers: 1.0 }, targets: { is_bird: 0.95 } },
  { inputs: { has_wings: 0.0, has_feathers: 0.0 }, targets: { is_bird: 0.1 } }
]);

// 3. Export (Save learned weights)
const trainedSchema = ai.export();
fs.writeFileSync('trained-model.json', JSON.stringify(trainedSchema, null, 2));
```

### Causal Reasoning Example

```typescript
import { NeuroEngine } from 'neurosym';

const weatherModel = new NeuroEngine({
  version: '1.0',
  variables: {
    cloudy: { type: 'bool', prior: 0.4 },
    raining: { type: 'bool', prior: 0.2 },
    wet_ground: { type: 'bool', prior: 0.1 },
    slippery: { type: 'bool', prior: 0.05 }
  },
  rules: [
    {
      id: 'clouds_cause_rain',
      type: 'IMPLICATION',
      inputs: ['cloudy'],
      output: 'raining',
      op: 'IDENTITY',
      weight: 0.8
    },
    {
      id: 'rain_wets_ground',
      type: 'IMPLICATION',
      inputs: ['raining'],
      output: 'wet_ground',
      op: 'IDENTITY',
      weight: 0.95
    },
    {
      id: 'wet_makes_slippery',
      type: 'IMPLICATION',
      inputs: ['wet_ground'],
      output: 'slippery',
      op: 'IDENTITY',
      weight: 0.7
    }
  ],
  constraints: []
});

// Query: Given it's cloudy, what's the chance of slippery ground?
const result = weatherModel.run({ cloudy: 1.0 });
console.log('P(slippery | cloudy) =', result.slippery);
```

### Argumentation with Attacks

```typescript
import { NeuroEngine } from 'neurosym';

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
console.log('Conclusion (with conflict):', result.conclusion);
```

### Direct Logic Operations

```typescript
import { and, or, not, implies, inhibit, support } from 'neurosym';

// Lukasiewicz fuzzy logic operations
console.log(and(0.8, 0.9));        // 0.7 (max(0, 0.8 + 0.9 - 1))
console.log(or(0.3, 0.4));         // 0.7 (min(1, 0.3 + 0.4))
console.log(not(0.3));             // 0.7 (1 - 0.3)
console.log(implies(0.8, 0.5));    // 0.7 (min(1, 1 - 0.8 + 0.5))

// Argumentation operations
console.log(inhibit(0.8, 1.0, 0.5)); // 0.4 (target reduced by attack)
console.log(support(0.5, 1.0, 0.5)); // 0.75 (target increased by support)
```

## 📐 NeuroJSON Schema

All logic is defined in a serializable JSON format:

```json
{
  "version": "1.0",
  "name": "my-knowledge-graph",
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
      "weight": 0.9,
      "learnable": true
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

### Supported Operations

| Operation | Formula | Description |
|-----------|---------|-------------|
| `AND` | `max(0, sum(inputs) - (n-1))` | Strict requirement of all inputs |
| `OR` | `min(1, sum(inputs))` | Any input contributes |
| `NOT` | `1 - input` | Negation |
| `IDENTITY` | `input` | Direct pass-through |
| `INHIBIT` | `Target * (1 - Attacker * Weight)` | Attack relation |

### Rule Types
- `IMPLICATION`: A → B (if A then B)
- `CONJUNCTION`: A ∧ B (and)
- `DISJUNCTION`: A ∨ B (or)
- `EQUIVALENCE`: A ↔ B (if and only if)

### Constraint Types
- `ATTACK`: Source defeats/inhibits target
- `SUPPORT`: Source reinforces target
- `MUTEX`: Mutual exclusion (at most one true)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NeuroSym.js                               │
├─────────────────┬──────────────────┬───────────────────────────┤
│   Logic Core    │   NeuroGraph     │   NeuroEngine             │
│   (Stateless)   │   (State Mgr)    │   (Main API)              │
├─────────────────┼──────────────────┼───────────────────────────┤
│ • T-Norms       │ • Variables      │ • run(evidence)           │
│ • NOT/AND/OR    │ • Rules          │ • train(data)             │
│ • Implication   │ • Constraints    │ • export()                │
│ • Inhibition    │ • State Tracking │ • query(variable)         │
│ • Support       │ • Evidence Lock  │ • getVariables/Rules      │
└─────────────────┴──────────────────┴───────────────────────────┘
```

## 📚 API Reference

### `NeuroEngine` (Main Class)

```typescript
import { NeuroEngine } from 'neurosym';

const ai = new NeuroEngine(schema, config?);
```

#### Methods

| Method | Description |
|--------|-------------|
| `run(evidence?, iterations?)` | Run inference with optional evidence (hard constraints) |
| `query(variable, evidence?)` | Query a single variable |
| `train(data, epochs?)` | Train weights from examples |
| `export()` | Export schema with learned weights |
| `exportJSON()` | Export as JSON string |
| `getVariables()` | Get all variable names |
| `getRules()` | Get all rule IDs |
| `getRuleWeight(id)` | Get a rule's weight |
| `setRuleWeight(id, weight)` | Set a rule's weight |
| `getConfig()` / `setConfig()` | Get/set configuration |

#### Training Formula

Weights are updated using heuristic gradient descent:

```
Weight_New = Weight_Old + (Error × LearningRate × Input_Strength)
```

### Logic Functions

```typescript
import { and, or, not, implies, equivalent, inhibit, support } from 'neurosym';
```

| Function | Description |
|----------|-------------|
| `and(...values)` | Lukasiewicz conjunction |
| `or(...values)` | Lukasiewicz disjunction |
| `not(a)` | Negation |
| `implies(a, b)` | Implication |
| `equivalent(a, b)` | Equivalence |
| `inhibit(target, attacker, weight)` | Attack relation |
| `support(target, supporter, weight)` | Support relation |

## 🧪 Testing

```bash
npm test              # Run tests
npm run test:coverage # Run with coverage
```

## 🔧 Development

```bash
npm install           # Install dependencies
npm run build         # Build distribution
npm run typecheck     # Type checking
npm run lint          # Linting
```

## 📄 License

MIT

## 🙏 Acknowledgments

Inspired by:
- [Logical Neural Networks (IBM)](https://arxiv.org/abs/2006.13155)
- Lukasiewicz Logic / Fuzzy Logic
- Abstract Argumentation Frameworks
