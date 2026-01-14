# NeuroSym.js

A lightweight, **zero-dependency** JavaScript/TypeScript library for **Neurosymbolic AI**. It bridges the gap between neural networks (fuzziness, learning) and symbolic logic (guarantees, explainability).

NeuroSym.js implements **Logical Neural Networks (LNN)** concepts using a **Parsimonious Data Protocol (NeuroJSON)**. It runs in any JS environment: Node.js, Browser, or Edge Workers.

## ✨ Features

- **Pure Logic Core**: Lukasiewicz T-Norms for fuzzy/continuous logic (truth values 0.0 to 1.0)
- **Graph-Based Reasoning**: In-memory graph structure for variables, rules, and constraints
- **Inference Engine**: Forward chaining, belief propagation, and constraint application
- **Argumentation Support**: Attack/support relations for defeasible reasoning
- **Learnable Weights**: Train rule weights from examples via gradient-free optimization
- **Serializable**: All logic defined in JSON (NeuroJSON protocol), not hardcoded functions
- **Zero Dependencies**: No external runtime dependencies
- **TypeScript First**: Full type safety with comprehensive type definitions

## 📦 Installation

```bash
npm install neurosym
```

## 🚀 Quick Start

### Basic Inference

```typescript
import { NeuroGraph, InferenceEngine } from 'neurosym';

// Define a simple causal model
const graph = new NeuroGraph({
  version: '1.0',
  variables: {
    raining: { type: 'bool', prior: 0.3 },
    wet_ground: { type: 'bool', prior: 0.1 },
    slippery: { type: 'bool', prior: 0.05 }
  },
  rules: [
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
      weight: 0.8
    }
  ],
  constraints: []
});

// Create inference engine
const engine = new InferenceEngine();

// Query: What's the probability of slippery ground given it's raining?
const result = engine.inferWithEvidence(graph, { raining: 1.0 });

console.log('Wet ground:', result.states['wet_ground'].value); // ~0.95
console.log('Slippery:', result.states['slippery'].value);     // ~0.76
```

### Argumentation with Attacks

```typescript
import { NeuroGraph, InferenceEngine } from 'neurosym';

// Model an argument with pro/con reasoning
const graph = new NeuroGraph({
  version: '1.0',
  variables: {
    evidence_for: { type: 'bool', prior: 0.8 },
    evidence_against: { type: 'bool', prior: 0.6 },
    conclusion: { type: 'bool', prior: 0.5 }
  },
  rules: [
    {
      id: 'support_conclusion',
      type: 'IMPLICATION',
      inputs: ['evidence_for'],
      output: 'conclusion',
      op: 'IDENTITY',
      weight: 0.9
    }
  ],
  constraints: [
    {
      id: 'attack_conclusion',
      type: 'ATTACK',
      source: 'evidence_against',
      target: 'conclusion',
      weight: 0.8
    }
  ]
});

const engine = new InferenceEngine();
const result = engine.infer(graph);

// Conclusion is affected by both support and attack
console.log('Conclusion:', result.states['conclusion'].value);
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

### Training/Learning Weights

```typescript
import { NeuroGraph, InferenceEngine, TrainingExample } from 'neurosym';

const graph = new NeuroGraph({
  version: '1.0',
  variables: {
    input: { type: 'bool', prior: 0.5 },
    output: { type: 'bool', prior: 0.5 }
  },
  rules: [{
    id: 'learnable_rule',
    type: 'IMPLICATION',
    inputs: ['input'],
    output: 'output',
    op: 'IDENTITY',
    weight: 0.5,  // Initial weight
    learnable: true
  }],
  constraints: []
});

const engine = new InferenceEngine({ learningRate: 0.1 });

// Training examples
const examples: TrainingExample[] = [
  { inputs: { input: 1.0 }, outputs: { output: 0.9 } },
  { inputs: { input: 0.0 }, outputs: { output: 0.1 } }
];

const result = engine.train(graph, examples, 100);

console.log('Final loss:', result.loss);
console.log('Learned weight:', result.weights['learnable_rule']);
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

### Variable Types
- `bool`: Boolean/propositional variable
- `continuous`: Continuous-valued variable

### Rule Types
- `IMPLICATION`: A → B (if A then B)
- `CONJUNCTION`: A ∧ B (and)
- `DISJUNCTION`: A ∨ B (or)
- `EQUIVALENCE`: A ↔ B (if and only if)

### Operations
- `IDENTITY`: Pass-through
- `AND`: Lukasiewicz conjunction
- `OR`: Lukasiewicz disjunction
- `NOT`: Negation
- `WEIGHTED`: Weighted average

### Constraint Types
- `ATTACK`: Source defeats/inhibits target
- `SUPPORT`: Source reinforces target
- `MUTEX`: Mutual exclusion (at most one true)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NeuroSym.js                               │
├─────────────────┬──────────────────┬───────────────────────────┤
│   Logic Core    │   NeuroGraph     │   Inference Engine        │
│   (Stateless)   │   (State Mgr)    │   (Solver)                │
├─────────────────┼──────────────────┼───────────────────────────┤
│ • T-Norms       │ • Variables      │ • Forward Chaining        │
│ • NOT/AND/OR    │ • Rules          │ • Belief Propagation      │
│ • Implication   │ • Constraints    │ • Constraint Application  │
│ • Inhibition    │ • State Tracking │ • Training Loop           │
│ • Support       │ • Evidence Lock  │ • Query/Explain           │
└─────────────────┴──────────────────┴───────────────────────────┘
```

## 📚 API Reference

### Classes

#### `NeuroGraph`
- `constructor(doc?: NeuroJSON)` - Create graph from NeuroJSON
- `load(doc: NeuroJSON)` - Load NeuroJSON document
- `export()` - Export to NeuroJSON
- `getValue(name)` / `setValue(name, value)` - Get/set variable values
- `lockVariable(name, value)` - Lock variable as evidence
- `unlockVariable(name)` - Unlock variable
- `getRule(id)` / `getRules()` - Get rules
- `getConstraint(id)` / `getConstraints()` - Get constraints

#### `InferenceEngine`
- `constructor(config?: Partial<InferenceConfig>)` - Create engine
- `infer(graph, recordHistory?)` - Run inference
- `inferWithEvidence(graph, evidence)` - Run with locked evidence
- `query(graph, variable, evidence)` - Query single variable
- `explain(graph, evidence)` - Get MAP explanation
- `train(graph, examples, epochs)` - Train weights

### Functions

#### Logic Operations
- `not(a)` - Negation
- `and(...values)` - Conjunction
- `or(...values)` - Disjunction
- `implies(a, b)` - Implication
- `equivalent(a, b)` - Equivalence
- `inhibit(target, attacker, weight)` - Attack relation
- `support(target, supporter, weight)` - Support relation

## 🧪 Testing

```bash
npm test           # Run tests
npm run test:coverage  # Run with coverage
```

## 🔧 Development

```bash
npm install        # Install dependencies
npm run build      # Build distribution
npm run typecheck  # Type checking
npm run lint       # Linting
```

## 📄 License

MIT

## 🙏 Acknowledgments

Inspired by:
- [Logical Neural Networks (IBM)](https://arxiv.org/abs/2006.13155)
- Lukasiewicz Logic / Fuzzy Logic
- Abstract Argumentation Frameworks
