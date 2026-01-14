# NeuroSym.js API Reference

## Table of Contents

- [NeuroEngine](#neuroengine)
- [Logic Functions](#logic-functions)
- [Types](#types)
- [Low-Level Classes](#low-level-classes)

---

## NeuroEngine

The main entry point for NeuroSym.js.

### Constructor

```typescript
new NeuroEngine(schema: NeuroJSON, config?: Partial<EngineConfig>)
```

**Parameters:**
- `schema` - NeuroJSON document defining the logic graph
- `config` - Optional configuration overrides

**Example:**
```typescript
import { NeuroEngine } from 'neurosym';

const ai = new NeuroEngine({
  version: '1.0',
  variables: {
    rain: { type: 'bool', prior: 0.3 },
    wet: { type: 'bool', prior: 0.1 }
  },
  rules: [{
    id: 'rain_wets',
    type: 'IMPLICATION',
    inputs: ['rain'],
    output: 'wet',
    op: 'IDENTITY',
    weight: 0.95
  }],
  constraints: []
});
```

---

### Methods

#### `run(evidence?, iterations?)`

Runs inference with optional evidence.

```typescript
run(evidence?: Evidence, iterations?: number): InferenceOutput
```

**Parameters:**
- `evidence` - Optional object mapping variable names to truth values (locked as hard constraints)
- `iterations` - Optional max iterations (defaults to config.maxIterations)

**Returns:** Object with all variable values after inference

**Example:**
```typescript
const result = ai.run({ rain: 1.0 });
console.log(result.wet); // ~0.95
```

---

#### `query(variable, evidence?)`

Queries a single variable given evidence.

```typescript
query(variable: string, evidence?: Evidence): TruthValue
```

**Example:**
```typescript
const wetProb = ai.query('wet', { rain: 0.8 });
```

---

#### `train(data, epochs?)`

Trains rule weights from examples.

```typescript
train(data: TrainingData[], epochs?: number): number
```

**Parameters:**
- `data` - Array of training examples
- `epochs` - Number of training epochs (default: 100)

**Returns:** Final average loss

**Training Formula:**
```
Weight_New = Weight_Old + (Error × LearningRate × Input_Strength)
```

**Example:**
```typescript
ai.train([
  { inputs: { rain: 1.0 }, targets: { wet: 0.95 } },
  { inputs: { rain: 0.0 }, targets: { wet: 0.1 } }
], 100);
```

---

#### `export()`

Exports the schema with learned weights.

```typescript
export(): NeuroJSON
```

**Example:**
```typescript
const trained = ai.export();
fs.writeFileSync('model.json', JSON.stringify(trained, null, 2));
```

---

#### `exportJSON()`

Exports schema as a JSON string.

```typescript
exportJSON(): string
```

---

#### `getVariables()`

Gets all variable names.

```typescript
getVariables(): string[]
```

---

#### `getRules()`

Gets all rule IDs.

```typescript
getRules(): string[]
```

---

#### `getRuleWeight(ruleId)`

Gets a rule's current weight.

```typescript
getRuleWeight(ruleId: string): TruthValue | undefined
```

---

#### `setRuleWeight(ruleId, weight)`

Sets a rule's weight.

```typescript
setRuleWeight(ruleId: string, weight: TruthValue): boolean
```

---

#### `getConfig()` / `setConfig(config)`

Gets or updates the engine configuration.

```typescript
getConfig(): EngineConfig
setConfig(config: Partial<EngineConfig>): void
```

---

## Logic Functions

Pure functions for fuzzy logic operations.

### Import

```typescript
import { 
  and, or, not, implies, equivalent,
  inhibit, support, mutexNormalize,
  clamp, isValidTruthValue
} from 'neurosym';
```

---

### `and(...values)`

Lukasiewicz conjunction (T-norm).

```typescript
and(0.8, 0.9)  // 0.7 = max(0, 0.8 + 0.9 - 1)
and(1, 1)      // 1.0
and(0.5, 0.3)  // 0.0 = max(0, 0.8 - 1)
```

---

### `or(...values)`

Lukasiewicz disjunction (T-conorm).

```typescript
or(0.3, 0.4)   // 0.7 = min(1, 0.3 + 0.4)
or(0.8, 0.5)   // 1.0 = min(1, 1.3)
```

---

### `not(a)`

Negation.

```typescript
not(0.3)  // 0.7
not(1)    // 0
```

---

### `implies(antecedent, consequent)`

Lukasiewicz implication.

```typescript
implies(1, 0.5)  // 0.5 (if true, then 0.5)
implies(0, 0.5)  // 1.0 (false implies anything)
implies(0.8, 0.5) // 0.7
```

---

### `equivalent(a, b)`

Equivalence (similarity).

```typescript
equivalent(0.5, 0.5)  // 1.0 (same)
equivalent(0, 1)      // 0.0 (opposite)
equivalent(0.3, 0.7)  // 0.6
```

---

### `inhibit(target, attacker, weight?)`

Attack relation (argumentation).

```typescript
inhibit(0.8, 1.0, 0.5)  // 0.4 (50% attack)
inhibit(0.8, 1.0, 1.0)  // 0.0 (full attack)
inhibit(0.8, 0.0, 1.0)  // 0.8 (no attack)
```

---

### `support(target, supporter, weight?)`

Support relation (reinforcement).

```typescript
support(0.5, 1.0, 0.5)  // 0.75
support(0.5, 1.0, 1.0)  // 1.0
```

---

### `mutexNormalize(values)`

Normalizes values for mutual exclusion.

```typescript
mutexNormalize([0.6, 0.6])  // [0.5, 0.5] (sum ≤ 1)
mutexNormalize([0.3, 0.4])  // [0.3, 0.4] (unchanged)
```

---

## Types

### NeuroJSON

```typescript
interface NeuroJSON {
  version: string;
  name?: string;
  variables: Record<string, Variable>;
  rules: Rule[];
  constraints: Constraint[];
}
```

### Variable

```typescript
interface Variable {
  type: 'bool' | 'continuous';
  prior: number;  // [0, 1]
  locked?: boolean;
}
```

### Rule

```typescript
interface Rule {
  id: string;
  type: 'IMPLICATION' | 'CONJUNCTION' | 'DISJUNCTION' | 'EQUIVALENCE';
  inputs: string[];
  output: string;
  op: 'IDENTITY' | 'AND' | 'OR' | 'NOT' | 'WEIGHTED';
  weight: number;  // [0, 1]
  learnable?: boolean;
}
```

### Constraint

```typescript
interface Constraint {
  id: string;
  type: 'ATTACK' | 'SUPPORT' | 'MUTEX';
  source: string;
  target: string | string[];
  weight: number;
}
```

### EngineConfig

```typescript
interface EngineConfig {
  maxIterations: number;      // default: 100
  convergenceThreshold: number; // default: 0.001
  learningRate: number;       // default: 0.1
  dampingFactor: number;      // default: 0.5
}
```

### TrainingData

```typescript
interface TrainingData {
  inputs: Record<string, number>;
  targets: Record<string, number>;
}
```

---

## Low-Level Classes

For advanced use cases, you can access the underlying components.

### NeuroGraph

```typescript
import { NeuroGraph } from 'neurosym';

const graph = new NeuroGraph(neuroJSON);

graph.getValue('variableName');
graph.setValue('variableName', 0.8);
graph.lockVariable('evidence', 1.0);
graph.resetToPriors();
```

### InferenceEngine

```typescript
import { InferenceEngine } from 'neurosym';

const engine = new InferenceEngine({ maxIterations: 50 });
const result = engine.infer(graph);
const explanation = engine.explain(graph, evidence);
```
