/**
 * Basic NeuroSym.js Example
 * 
 * Demonstrates the main NeuroEngine API for:
 * - Inference (reasoning)
 * - Training (learning)
 * - Export (serialization)
 */

import { NeuroEngine, and, or, not, implies, NeuroJSON } from '../src/index';

// ============================================================================
// Example 1: Direct Logic Operations
// ============================================================================

console.log('=== Lukasiewicz Logic Operations ===\n');

console.log('and(0.8, 0.9) =', and(0.8, 0.9).toFixed(2));        // Strict AND
console.log('or(0.3, 0.4) =', or(0.3, 0.4).toFixed(2));          // Permissive OR
console.log('not(0.3) =', not(0.3).toFixed(2));                   // Negation
console.log('implies(1.0, 0.5) =', implies(1.0, 0.5).toFixed(2)); // Implication

// ============================================================================
// Example 2: Bird Classification (from design doc)
// ============================================================================

console.log('\n=== Bird Classification ===\n');

const birdSchema: NeuroJSON = {
  version: '1.0',
  name: 'bird-classifier',
  variables: {
    has_wings: { type: 'bool', prior: 0.5 },
    has_feathers: { type: 'bool', prior: 0.5 },
    flies: { type: 'bool', prior: 0.5 },
    is_bird: { type: 'bool', prior: 0.3 }
  },
  rules: [
    {
      id: 'wings_imply_bird',
      type: 'IMPLICATION',
      inputs: ['has_wings'],
      output: 'is_bird',
      op: 'IDENTITY',
      weight: 0.8,
      learnable: true
    },
    {
      id: 'feathers_imply_bird',
      type: 'IMPLICATION',
      inputs: ['has_feathers'],
      output: 'is_bird',
      op: 'IDENTITY',
      weight: 0.9,
      learnable: true
    },
    {
      id: 'flies_imply_bird',
      type: 'IMPLICATION',
      inputs: ['flies'],
      output: 'is_bird',
      op: 'IDENTITY',
      weight: 0.7,
      learnable: true
    }
  ],
  constraints: []
};

const birdAI = new NeuroEngine(birdSchema);

// Test case: Penguin (has wings, but doesn't fly)
console.log('Penguin test (has_wings=1, flies=0):');
const penguinResult = birdAI.run({ has_wings: 1.0, flies: 0.0 });
console.log('  is_bird =', penguinResult.is_bird?.toFixed(3));

// Test case: Sparrow (has wings, flies)
console.log('\nSparrow test (has_wings=1, has_feathers=1, flies=1):');
const sparrowResult = birdAI.run({ has_wings: 1.0, has_feathers: 1.0, flies: 1.0 });
console.log('  is_bird =', sparrowResult.is_bird?.toFixed(3));

// ============================================================================
// Example 3: Training
// ============================================================================

console.log('\n=== Training Demo ===\n');

console.log('Before training:');
console.log('  wings_imply_bird weight:', birdAI.getRuleWeight('wings_imply_bird'));

// Train with examples
birdAI.train([
  { inputs: { has_wings: 1.0, has_feathers: 1.0 }, targets: { is_bird: 0.95 } },
  { inputs: { has_wings: 1.0, has_feathers: 0.0, flies: 0.0 }, targets: { is_bird: 0.7 } },
  { inputs: { has_wings: 0.0, has_feathers: 0.0 }, targets: { is_bird: 0.1 } }
], 50);

console.log('\nAfter training:');
console.log('  wings_imply_bird weight:', birdAI.getRuleWeight('wings_imply_bird')?.toFixed(3));

// Test again
const trainedResult = birdAI.run({ has_wings: 1.0, has_feathers: 1.0 });
console.log('  is_bird (with feathers+wings):', trainedResult.is_bird?.toFixed(3));

// ============================================================================
// Example 4: Weather Causal Model
// ============================================================================

console.log('\n=== Weather Causal Model ===\n');

const weatherSchema: NeuroJSON = {
  version: '1.0',
  name: 'weather-model',
  variables: {
    cloudy: { type: 'bool', prior: 0.4 },
    raining: { type: 'bool', prior: 0.2 },
    wet_ground: { type: 'bool', prior: 0.1 },
    sprinkler_on: { type: 'bool', prior: 0.1 },
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
      id: 'sprinkler_wets_ground',
      type: 'IMPLICATION',
      inputs: ['sprinkler_on'],
      output: 'wet_ground',
      op: 'IDENTITY',
      weight: 0.9
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
};

const weatherAI = new NeuroEngine(weatherSchema);

console.log('Scenario: It is cloudy');
const cloudyResult = weatherAI.run({ cloudy: 1.0 });
console.log('  P(raining) =', cloudyResult.raining?.toFixed(3));
console.log('  P(wet_ground) =', cloudyResult.wet_ground?.toFixed(3));
console.log('  P(slippery) =', cloudyResult.slippery?.toFixed(3));

console.log('\nScenario: Ground is wet, not raining');
const wetNoRainResult = weatherAI.run({ wet_ground: 1.0, raining: 0.0 });
console.log('  P(cloudy) =', wetNoRainResult.cloudy?.toFixed(3));
console.log('  P(sprinkler_on) =', wetNoRainResult.sprinkler_on?.toFixed(3));

// ============================================================================
// Example 5: Argumentation (Pro/Con Debate)
// ============================================================================

console.log('\n=== Argumentation Model ===\n');

const debateSchema: NeuroJSON = {
  version: '1.0',
  name: 'debate-model',
  variables: {
    pro_argument: { type: 'bool', prior: 0.7 },
    con_argument: { type: 'bool', prior: 0.6 },
    rebuttal: { type: 'bool', prior: 0.4 },
    conclusion: { type: 'bool', prior: 0.5 }
  },
  rules: [
    {
      id: 'pro_supports_conclusion',
      type: 'IMPLICATION',
      inputs: ['pro_argument'],
      output: 'conclusion',
      op: 'IDENTITY',
      weight: 0.85
    }
  ],
  constraints: [
    {
      id: 'con_attacks_conclusion',
      type: 'ATTACK',
      source: 'con_argument',
      target: 'conclusion',
      weight: 0.7
    },
    {
      id: 'rebuttal_attacks_con',
      type: 'ATTACK',
      source: 'rebuttal',
      target: 'con_argument',
      weight: 0.8
    }
  ]
};

const debateAI = new NeuroEngine(debateSchema);

console.log('Debate with pro, con, and rebuttal:');
const debateResult = debateAI.run();
console.log('  pro_argument =', debateResult.pro_argument?.toFixed(3));
console.log('  con_argument (attacked by rebuttal) =', debateResult.con_argument?.toFixed(3));
console.log('  conclusion =', debateResult.conclusion?.toFixed(3));

console.log('\nStrong rebuttal scenario:');
const strongRebuttalResult = debateAI.run({ rebuttal: 1.0 });
console.log('  con_argument =', strongRebuttalResult.con_argument?.toFixed(3));
console.log('  conclusion =', strongRebuttalResult.conclusion?.toFixed(3));

// ============================================================================
// Example 6: Export (Serialization)
// ============================================================================

console.log('\n=== Export Demo ===\n');

const exportedSchema = birdAI.export();
console.log('Exported schema name:', exportedSchema.name);
console.log('Number of variables:', Object.keys(exportedSchema.variables).length);
console.log('Number of rules:', exportedSchema.rules.length);
console.log('Learned weight for wings_imply_bird:', 
  exportedSchema.rules.find(r => r.id === 'wings_imply_bird')?.weight.toFixed(3));

console.log('\n✅ Done!');
