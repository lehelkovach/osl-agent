/**
 * NeuroEngine Module
 * 
 * The main entry point for NeuroSym.js - a unified class that combines
 * graph management and inference into a clean API.
 * 
 * Usage:
 * ```typescript
 * import { NeuroEngine } from 'neurosym';
 * import schema from './my-logic.json';
 * 
 * const ai = new NeuroEngine(schema);
 * const result = ai.run({ has_wings: 1.0, flies: 0.0 });
 * console.log(result.is_bird);
 * ```
 */

import { NeuroGraph } from './neuro-graph';
import {
  NeuroJSON,
  TruthValue,
  Rule,
  createDefaultConfig
} from './types';
import {
  clamp,
  and,
  or,
  equivalent,
  applyOperation,
  inhibit,
  support,
  mutexNormalize
} from './logic-core';

// ============================================================================
// Types
// ============================================================================

/**
 * Evidence input: variable name -> observed truth value
 */
export type Evidence = Record<string, TruthValue>;

/**
 * Inference result: variable name -> computed truth value
 */
export type InferenceOutput = Record<string, TruthValue>;

/**
 * Training example
 */
export interface TrainingData {
  inputs: Evidence;
  targets: Record<string, TruthValue>;
}

/**
 * Engine configuration
 */
export interface EngineConfig {
  /** Maximum iterations for inference convergence */
  maxIterations: number;
  /** Convergence threshold */
  convergenceThreshold: number;
  /** Learning rate for weight updates */
  learningRate: number;
  /** Damping factor for belief propagation */
  dampingFactor: number;
}

// ============================================================================
// NeuroEngine Class
// ============================================================================

/**
 * Main entry point for NeuroSym.js
 * 
 * Combines graph state management and inference engine into a unified API.
 */
export class NeuroEngine {
  private graph: NeuroGraph;
  private config: EngineConfig;

  /**
   * Creates a new NeuroEngine instance
   * 
   * @param schema NeuroJSON schema defining variables, rules, and constraints
   * @param config Optional configuration overrides
   */
  constructor(schema: NeuroJSON, config?: Partial<EngineConfig>) {
    this.graph = new NeuroGraph(schema);
    this.config = { ...createDefaultConfig(), ...config };
  }

  // ==========================================================================
  // Configuration
  // ==========================================================================

  /**
   * Gets the current configuration
   */
  getConfig(): EngineConfig {
    return { ...this.config };
  }

  /**
   * Updates the configuration
   */
  setConfig(config: Partial<EngineConfig>): void {
    this.config = { ...this.config, ...config };
  }

  // ==========================================================================
  // Inference
  // ==========================================================================

  /**
   * Runs inference with optional evidence
   * 
   * This is the main inference method. It:
   * 1. Resets the graph to priors
   * 2. Locks evidence variables (hard constraints)
   * 3. Propagates values through rules and constraints
   * 4. Returns the final state of all variables
   * 
   * @param evidence Optional variable -> value mappings to lock as evidence
   * @param iterations Optional max iterations (defaults to config)
   * @returns Object with all variable values after inference
   * 
   * @example
   * ```typescript
   * const result = ai.run({ has_wings: 1.0, flies: 0.0 });
   * console.log(result.is_bird);
   * ```
   */
  run(evidence?: Evidence, iterations?: number): InferenceOutput {
    const maxIter = iterations ?? this.config.maxIterations;
    
    // Reset to priors
    this.graph.resetToPriors();
    this.graph.unlockAll();
    
    // Lock evidence as hard constraints
    if (evidence) {
      for (const [name, value] of Object.entries(evidence)) {
        if (this.graph.hasVariable(name)) {
          this.graph.lockVariable(name, value);
        }
      }
    }
    
    // Run inference loop
    for (let i = 0; i < maxIter; i++) {
      const ruleDelta = this.forwardPass();
      const constraintDelta = this.applyConstraints();
      
      const totalDelta = Math.max(ruleDelta, constraintDelta);
      if (totalDelta < this.config.convergenceThreshold) {
        break;
      }
    }
    
    // Return all variable values
    return this.graph.getAllValues();
  }

  /**
   * Queries a specific variable given evidence
   * 
   * @param variable The variable to query
   * @param evidence Evidence to condition on
   * @returns The truth value of the queried variable
   */
  query(variable: string, evidence?: Evidence): TruthValue {
    const result = this.run(evidence);
    return result[variable] ?? 0.5;
  }

  /**
   * Gets the current state without running inference
   */
  getState(): InferenceOutput {
    return this.graph.getAllValues();
  }

  // ==========================================================================
  // Training
  // ==========================================================================

  /**
   * Trains the engine on a set of examples
   * 
   * Uses the heuristic gradient descent formula:
   * Weight_New = Weight_Old + (Error * LearningRate * Input_Strength)
   * 
   * @param data Array of training examples
   * @param epochs Number of training epochs (default: 100)
   * @returns Final average loss
   * 
   * @example
   * ```typescript
   * ai.train([
   *   { inputs: { weather_clear: 1.0 }, targets: { good_mood: 0.8 } },
   *   { inputs: { weather_clear: 0.0 }, targets: { good_mood: 0.3 } }
   * ]);
   * ```
   */
  train(data: TrainingData[], epochs: number = 100): number {
    let finalLoss = 0;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      
      for (const example of data) {
        // Run inference with inputs
        const output = this.run(example.inputs);
        
        // Compute error for each target
        for (const [targetVar, targetValue] of Object.entries(example.targets)) {
          const actualValue = output[targetVar] ?? 0.5;
          const error = targetValue - actualValue;
          epochLoss += Math.pow(error, 2);
          
          // Update weights for rules that output to this target
          this.updateWeights(targetVar, error, example.inputs);
        }
      }
      
      epochLoss /= data.length;
      finalLoss = epochLoss;
      
      // Early stopping
      if (epochLoss < 0.001) break;
    }
    
    return finalLoss;
  }

  /**
   * Updates rule weights based on error
   * Formula: Weight_New = Weight_Old + (Error * LearningRate * Input_Strength)
   */
  private updateWeights(targetVar: string, error: number, inputs: Evidence): void {
    const rules = this.graph.getRulesWithOutput(targetVar);
    
    for (const rule of rules) {
      if (rule.learnable === false) continue;
      
      // Calculate input strength (average of input values)
      let inputStrength = 0;
      for (const inputName of rule.inputs) {
        inputStrength += inputs[inputName] ?? this.graph.getValue(inputName) ?? 0.5;
      }
      inputStrength /= rule.inputs.length || 1;
      
      // Apply weight update formula
      const delta = error * this.config.learningRate * inputStrength;
      const newWeight = clamp(rule.weight + delta);
      this.graph.setRuleWeight(rule.id, newWeight);
    }
  }

  // ==========================================================================
  // Export
  // ==========================================================================

  /**
   * Exports the current schema with learned weights
   * 
   * This allows you to save the trained model and reload it later.
   * 
   * @returns NeuroJSON schema with current weights
   * 
   * @example
   * ```typescript
   * const trainedSchema = ai.export();
   * fs.writeFileSync('trained-model.json', JSON.stringify(trainedSchema, null, 2));
   * ```
   */
  export(): NeuroJSON {
    return this.graph.export();
  }

  /**
   * Exports the schema as a JSON string
   */
  exportJSON(): string {
    return JSON.stringify(this.export(), null, 2);
  }

  // ==========================================================================
  // Graph Access
  // ==========================================================================

  /**
   * Gets the underlying graph (for advanced use)
   */
  getGraph(): NeuroGraph {
    return this.graph;
  }

  /**
   * Gets the names of all variables
   */
  getVariables(): string[] {
    return this.graph.getVariableNames();
  }

  /**
   * Gets the IDs of all rules
   */
  getRules(): string[] {
    return this.graph.getRules().map(r => r.id);
  }

  /**
   * Gets the current weight of a rule
   */
  getRuleWeight(ruleId: string): TruthValue | undefined {
    return this.graph.getRule(ruleId)?.weight;
  }

  /**
   * Manually sets a rule weight
   */
  setRuleWeight(ruleId: string, weight: TruthValue): boolean {
    return this.graph.setRuleWeight(ruleId, weight);
  }

  // ==========================================================================
  // Internal Inference Methods
  // ==========================================================================

  /**
   * Single forward pass through all rules
   */
  private forwardPass(): number {
    let maxDelta = 0;
    const order = this.graph.getTopologicalOrder();
    
    for (const variableName of order) {
      const rules = this.graph.getRulesWithOutput(variableName);
      if (rules.length === 0) continue;
      if (this.graph.isLocked(variableName)) continue;
      
      // Compute contributions from all rules
      const contributions: TruthValue[] = [];
      const weights: number[] = [];
      
      for (const rule of rules) {
        const ruleValue = this.evaluateRule(rule);
        if (ruleValue !== null) {
          contributions.push(ruleValue);
          weights.push(rule.weight);
        }
      }
      
      if (contributions.length === 0) continue;
      
      // Combine contributions (weighted average with damping)
      const oldValue = this.graph.getValue(variableName) ?? 0.5;
      let totalWeight = 0;
      let weightedSum = 0;
      
      for (let i = 0; i < contributions.length; i++) {
        totalWeight += weights[i]!;
        weightedSum += contributions[i]! * weights[i]!;
      }
      
      const newContribution = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
      const dampedValue = this.config.dampingFactor * newContribution + 
                         (1 - this.config.dampingFactor) * oldValue;
      
      this.graph.setValue(variableName, dampedValue);
      maxDelta = Math.max(maxDelta, Math.abs(dampedValue - oldValue));
    }
    
    return maxDelta;
  }

  /**
   * Evaluates a single rule
   */
  private evaluateRule(rule: Rule): TruthValue | null {
    const inputValues: TruthValue[] = [];
    for (const inputName of rule.inputs) {
      const value = this.graph.getValue(inputName);
      if (value === undefined) return null;
      inputValues.push(value);
    }
    
    let result: TruthValue;
    
    switch (rule.type) {
      case 'IMPLICATION': {
        const antecedent = applyOperation(rule.op, inputValues);
        result = antecedent * rule.weight;
        break;
      }
      case 'CONJUNCTION': {
        result = and(...inputValues) * rule.weight;
        break;
      }
      case 'DISJUNCTION': {
        result = or(...inputValues) * rule.weight;
        break;
      }
      case 'EQUIVALENCE': {
        if (inputValues.length >= 2) {
          result = equivalent(inputValues[0]!, inputValues[1]!) * rule.weight;
        } else {
          result = (inputValues[0] ?? 0.5) * rule.weight;
        }
        break;
      }
      default:
        return null;
    }
    
    return clamp(result);
  }

  /**
   * Applies all constraints
   */
  private applyConstraints(): number {
    let maxDelta = 0;
    const constraints = this.graph.getConstraints();
    const mutexGroups = new Map<string, typeof constraints>();
    
    for (const constraint of constraints) {
      switch (constraint.type) {
        case 'ATTACK': {
          const sourceValue = this.graph.getValue(constraint.source);
          if (sourceValue === undefined) continue;
          
          const targets = Array.isArray(constraint.target) 
            ? constraint.target 
            : [constraint.target];
          
          for (const targetName of targets) {
            if (this.graph.isLocked(targetName)) continue;
            const targetValue = this.graph.getValue(targetName);
            if (targetValue === undefined) continue;
            
            const newValue = inhibit(targetValue, sourceValue, constraint.weight);
            this.graph.setValue(targetName, newValue);
            maxDelta = Math.max(maxDelta, Math.abs(newValue - targetValue));
          }
          break;
        }
        
        case 'SUPPORT': {
          const sourceValue = this.graph.getValue(constraint.source);
          if (sourceValue === undefined) continue;
          
          const targets = Array.isArray(constraint.target) 
            ? constraint.target 
            : [constraint.target];
          
          for (const targetName of targets) {
            if (this.graph.isLocked(targetName)) continue;
            const targetValue = this.graph.getValue(targetName);
            if (targetValue === undefined) continue;
            
            const newValue = support(targetValue, sourceValue, constraint.weight);
            this.graph.setValue(targetName, newValue);
            maxDelta = Math.max(maxDelta, Math.abs(newValue - targetValue));
          }
          break;
        }
        
        case 'MUTEX': {
          const targets = Array.isArray(constraint.target) ? constraint.target : [constraint.target];
          const key = [constraint.source, ...targets].sort().join('|');
          const group = mutexGroups.get(key) ?? [];
          group.push(constraint);
          mutexGroups.set(key, group);
          break;
        }
      }
    }
    
    // Process mutex groups
    for (const constraints of mutexGroups.values()) {
      const variables = new Set<string>();
      for (const c of constraints) {
        variables.add(c.source);
        const targets = Array.isArray(c.target) ? c.target : [c.target];
        targets.forEach(t => variables.add(t));
      }
      
      const names = Array.from(variables);
      const values = names.map(n => this.graph.getValue(n) ?? 0);
      const locked = names.map(n => this.graph.isLocked(n));
      
      const unlockedIndices = locked.map((l, i) => l ? -1 : i).filter(i => i >= 0);
      const lockedSum = locked.reduce((sum, l, i) => sum + (l ? values[i]! : 0), 0);
      const available = Math.max(0, 1 - lockedSum);
      
      const unlockedValues = unlockedIndices.map(i => values[i]!);
      const unlockedSum = unlockedValues.reduce((a, b) => a + b, 0);
      const normalized = mutexNormalize(
        unlockedValues.map(v => v * available / Math.max(0.001, unlockedSum))
      );
      
      unlockedIndices.forEach((idx, i) => {
        const name = names[idx]!;
        const oldValue = values[idx]!;
        const newValue = normalized[i] ?? oldValue;
        
        if (!this.graph.isLocked(name)) {
          this.graph.setValue(name, newValue);
          maxDelta = Math.max(maxDelta, Math.abs(newValue - oldValue));
        }
      });
    }
    
    return maxDelta;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Creates a new NeuroEngine instance
 * 
 * @param schema NeuroJSON schema
 * @param config Optional configuration
 * @returns New NeuroEngine instance
 */
export function createEngine(schema: NeuroJSON, config?: Partial<EngineConfig>): NeuroEngine {
  return new NeuroEngine(schema, config);
}
