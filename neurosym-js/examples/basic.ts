/**
 * Basic NeuroSym.js Example
 * 
 * Demonstrates causal reasoning with the inference engine.
 */

import { NeuroGraph, InferenceEngine, and, or, not, implies } from '../src/index';

// Example 1: Direct logic operations
console.log('=== Logic Operations ===');
console.log('and(0.8, 0.9) =', and(0.8, 0.9));        // Lukasiewicz AND
console.log('or(0.3, 0.4) =', or(0.3, 0.4));          // Lukasiewicz OR
console.log('not(0.3) =', not(0.3));                   // Negation
console.log('implies(1.0, 0.5) =', implies(1.0, 0.5)); // Implication

// Example 2: Causal reasoning - Weather model
console.log('\n=== Weather Causal Model ===');

const weatherGraph = new NeuroGraph({
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
    // Clouds increase chance of rain
    {
      id: 'clouds_cause_rain',
      type: 'IMPLICATION',
      inputs: ['cloudy'],
      output: 'raining',
      op: 'IDENTITY',
      weight: 0.8
    },
    // Rain wets the ground
    {
      id: 'rain_wets_ground',
      type: 'IMPLICATION',
      inputs: ['raining'],
      output: 'wet_ground',
      op: 'IDENTITY',
      weight: 0.95
    },
    // Sprinkler also wets the ground
    {
      id: 'sprinkler_wets_ground',
      type: 'IMPLICATION',
      inputs: ['sprinkler_on'],
      output: 'wet_ground',
      op: 'IDENTITY',
      weight: 0.9
    },
    // Wet ground makes it slippery
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

const engine = new InferenceEngine({
  maxIterations: 50,
  convergenceThreshold: 0.001
});

// Query 1: Given it's cloudy, what's the chance of slippery ground?
console.log('\nScenario: It is cloudy');
const result1 = engine.inferWithEvidence(weatherGraph, { cloudy: 1.0 });
console.log('  P(raining) =', result1.states['raining']?.value.toFixed(3));
console.log('  P(wet_ground) =', result1.states['wet_ground']?.value.toFixed(3));
console.log('  P(slippery) =', result1.states['slippery']?.value.toFixed(3));

// Query 2: Given the ground is wet but it's not raining, is sprinkler on?
console.log('\nScenario: Wet ground, no rain');
const result2 = engine.inferWithEvidence(weatherGraph, { 
  wet_ground: 1.0, 
  raining: 0.0 
});
console.log('  P(sprinkler_on) =', result2.states['sprinkler_on']?.value.toFixed(3));
console.log('  P(cloudy) =', result2.states['cloudy']?.value.toFixed(3));

// Example 3: Argumentation - Debate model
console.log('\n=== Argumentation Model ===');

const debateGraph = new NeuroGraph({
  version: '1.0',
  name: 'debate-model',
  variables: {
    claim: { type: 'bool', prior: 0.5 },
    argument_pro: { type: 'bool', prior: 0.7 },
    argument_con: { type: 'bool', prior: 0.6 },
    rebuttal: { type: 'bool', prior: 0.4 }
  },
  rules: [
    // Pro argument supports the claim
    {
      id: 'pro_supports_claim',
      type: 'IMPLICATION',
      inputs: ['argument_pro'],
      output: 'claim',
      op: 'IDENTITY',
      weight: 0.85
    }
  ],
  constraints: [
    // Con argument attacks the claim
    {
      id: 'con_attacks_claim',
      type: 'ATTACK',
      source: 'argument_con',
      target: 'claim',
      weight: 0.7
    },
    // Rebuttal attacks the con argument
    {
      id: 'rebuttal_attacks_con',
      type: 'ATTACK',
      source: 'rebuttal',
      target: 'argument_con',
      weight: 0.8
    }
  ]
});

console.log('\nInitial beliefs (before inference):');
console.log('  claim =', debateGraph.getValue('claim')?.toFixed(3));

const debateResult = engine.infer(debateGraph);
console.log('\nAfter inference:');
console.log('  claim =', debateResult.states['claim']?.value.toFixed(3));
console.log('  argument_con (attacked by rebuttal) =', 
  debateResult.states['argument_con']?.value.toFixed(3));

console.log('\nDone!');
