/**
 * NeuroSym.js
 * 
 * A lightweight, zero-dependency JavaScript library for Neurosymbolic AI.
 * Implements Logical Neural Networks (LNN) with a NeuroJSON protocol.
 * 
 * Core Philosophy: "Code as Data" - Logic is defined in serializable JSON, not hardcoded functions.
 * 
 * @example
 * ```typescript
 * import { NeuroEngine } from 'neurosym';
 * import schema from './bird-logic.json';
 * 
 * // Initialize
 * const ai = new NeuroEngine(schema);
 * 
 * // Inference (Reasoning)
 * const result = ai.run({
 *   has_wings: 1.0,
 *   flies: 0.0  // It's a penguin!
 * });
 * console.log(result.is_bird);
 * 
 * // Training (Learning)
 * ai.train([
 *   { inputs: { has_wings: 1.0 }, targets: { is_bird: 0.9 } }
 * ]);
 * 
 * // Export (Save learned weights)
 * const trainedSchema = ai.export();
 * ```
 * 
 * @packageDocumentation
 */

// =============================================================================
// Main Entry Point
// =============================================================================

export { 
  NeuroEngine, 
  createEngine,
  type Evidence,
  type InferenceOutput,
  type TrainingData,
  type EngineConfig
} from './engine';

// =============================================================================
// Lower-level Classes (for advanced use)
// =============================================================================

export { NeuroGraph, createGraph, parseNeuroJSON } from './neuro-graph';
export { InferenceEngine, infer, query } from './inference';
export { createEngine as createInferenceEngine } from './inference';

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
