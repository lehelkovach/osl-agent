/**
 * NeuroSym.js
 * 
 * A lightweight, zero-dependency JavaScript library for Neurosymbolic AI.
 * Implements Logical Neural Networks (LNN) with a NeuroJSON protocol.
 * 
 * @example
 * ```typescript
 * import { NeuroGraph, InferenceEngine } from 'neurosym';
 * 
 * const graph = new NeuroGraph({
 *   version: '1.0',
 *   variables: {
 *     raining: { type: 'bool', prior: 0.3 },
 *     wet_ground: { type: 'bool', prior: 0.1 }
 *   },
 *   rules: [{
 *     id: 'rain_causes_wet',
 *     type: 'IMPLICATION',
 *     inputs: ['raining'],
 *     output: 'wet_ground',
 *     op: 'IDENTITY',
 *     weight: 0.95
 *   }],
 *   constraints: []
 * });
 * 
 * const engine = new InferenceEngine();
 * const result = engine.inferWithEvidence(graph, { raining: 1.0 });
 * console.log(result.states['wet_ground'].value); // ~0.95
 * ```
 * 
 * @packageDocumentation
 */

// =============================================================================
// Core Classes
// =============================================================================

export { NeuroGraph, createGraph, parseNeuroJSON } from './neuro-graph';
export { InferenceEngine, createEngine, infer, query } from './inference';

// =============================================================================
// Logic Functions
// =============================================================================

export {
  // Utilities
  clamp,
  isValidTruthValue,
  
  // Basic Logic Operations
  not,
  and,
  or,
  implies,
  equivalent,
  
  // Weighted Operations
  weightedAverage,
  weightedAnd,
  weightedOr,
  
  // Argumentation
  inhibit,
  support,
  mutexNormalize,
  
  // Operation Dispatcher
  applyOperation,
  
  // Gradients
  andGradient,
  orGradient,
  impliesGradient,
  inhibitGradient
} from './logic-core';

// =============================================================================
// Types
// =============================================================================

export type {
  // Core Types
  TruthValue,
  VariableType,
  Operation,
  RuleType,
  ConstraintType,
  
  // Schema Components
  Variable,
  Rule,
  Constraint,
  NeuroJSON,
  
  // Runtime Types
  VariableState,
  InferenceConfig,
  InferenceResult,
  TrainingExample,
  TrainingResult,
  
  // Validation Types
  ValidationError,
  ValidationResult
} from './types';

// =============================================================================
// Type Guards and Validators
// =============================================================================

export {
  validateNeuroJSON,
  createVariable,
  createDefaultConfig
} from './types';

// =============================================================================
// Version
// =============================================================================

/** Library version */
export const VERSION = '0.1.0';

/** NeuroJSON schema version */
export const SCHEMA_VERSION = '1.0';
