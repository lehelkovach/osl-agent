/**
 * Inference Engine Module
 * 
 * The solver that propagates truth values through the neurosymbolic graph.
 * 
 * Features:
 * - Forward Chaining: Propagate data -> conclusions
 * - Belief Propagation: Iterative message passing until convergence
 * - Constraint Application: Handle attacks, supports, mutex
 * - Training: Learn weights from examples via backpropagation-lite
 */

import { NeuroGraph } from './neuro-graph';
import {
  TruthValue,
  Rule,
  Constraint,
  InferenceConfig,
  InferenceResult,
  TrainingExample,
  TrainingResult,
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
// Inference Engine Class
// ============================================================================

/**
 * Main inference engine for neurosymbolic reasoning
 */
export class InferenceEngine {
  private config: InferenceConfig;

  constructor(config?: Partial<InferenceConfig>) {
    this.config = { ...createDefaultConfig(), ...config };
  }

  /**
   * Updates the configuration
   */
  setConfig(config: Partial<InferenceConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Gets the current configuration
   */
  getConfig(): InferenceConfig {
    return { ...this.config };
  }

  // ==========================================================================
  // Forward Chaining
  // ==========================================================================

  /**
   * Performs a single forward pass through all rules
   * 
   * Processes rules in topological order to ensure dependencies
   * are computed before dependents.
   * 
   * @param graph The graph to perform inference on
   * @returns Maximum change in any variable's value
   */
  forwardPass(graph: NeuroGraph): number {
    let maxDelta = 0;
    const order = graph.getTopologicalOrder();
    
    for (const variableName of order) {
      // Get all rules that output to this variable
      const rules = graph.getRulesWithOutput(variableName);
      if (rules.length === 0) continue;
      
      // Skip locked variables
      if (graph.isLocked(variableName)) continue;
      
      // Compute contributions from all rules
      const contributions: TruthValue[] = [];
      const weights: number[] = [];
      
      for (const rule of rules) {
        const ruleValue = this.evaluateRule(graph, rule);
        if (ruleValue !== null) {
          contributions.push(ruleValue);
          weights.push(rule.weight);
        }
      }
      
      if (contributions.length === 0) continue;
      
      // Combine contributions (weighted average with damping)
      const oldValue = graph.getValue(variableName) ?? 0.5;
      let totalWeight = 0;
      let weightedSum = 0;
      
      for (let i = 0; i < contributions.length; i++) {
        totalWeight += weights[i]!;
        weightedSum += contributions[i]! * weights[i]!;
      }
      
      const newContribution = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
      
      // Apply damping to smooth updates
      const dampedValue = this.config.dampingFactor * newContribution + 
                         (1 - this.config.dampingFactor) * oldValue;
      
      graph.setValue(variableName, dampedValue);
      
      const delta = Math.abs(dampedValue - oldValue);
      maxDelta = Math.max(maxDelta, delta);
    }
    
    return maxDelta;
  }

  /**
   * Evaluates a single rule given current variable values
   */
  evaluateRule(graph: NeuroGraph, rule: Rule): TruthValue | null {
    // Get input values
    const inputValues: TruthValue[] = [];
    for (const inputName of rule.inputs) {
      const value = graph.getValue(inputName);
      if (value === undefined) return null;
      inputValues.push(value);
    }
    
    // Apply operation based on rule type
    let result: TruthValue;
    
    switch (rule.type) {
      case 'IMPLICATION': {
        // For implication, we compute: if antecedent then consequent
        // The rule "fires" proportionally to the antecedent
        const antecedent = applyOperation(rule.op, inputValues);
        // The output becomes more true as antecedent becomes true
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

  // ==========================================================================
  // Constraint Application
  // ==========================================================================

  /**
   * Applies all constraints to the graph
   * 
   * @param graph The graph to apply constraints to
   * @returns Maximum change in any variable's value
   */
  applyConstraints(graph: NeuroGraph): number {
    let maxDelta = 0;
    
    const constraints = graph.getConstraints();
    
    // Group mutex constraints for batch processing
    const mutexGroups = new Map<string, Constraint[]>();
    
    for (const constraint of constraints) {
      switch (constraint.type) {
        case 'ATTACK': {
          const delta = this.applyAttack(graph, constraint);
          maxDelta = Math.max(maxDelta, delta);
          break;
        }
        
        case 'SUPPORT': {
          const delta = this.applySupport(graph, constraint);
          maxDelta = Math.max(maxDelta, delta);
          break;
        }
        
        case 'MUTEX': {
          // Collect mutex constraints
          const key = this.getMutexKey(constraint);
          const group = mutexGroups.get(key) ?? [];
          group.push(constraint);
          mutexGroups.set(key, group);
          break;
        }
      }
    }
    
    // Process mutex groups
    for (const constraints of mutexGroups.values()) {
      const delta = this.applyMutex(graph, constraints);
      maxDelta = Math.max(maxDelta, delta);
    }
    
    return maxDelta;
  }

  /**
   * Applies an attack constraint
   */
  private applyAttack(graph: NeuroGraph, constraint: Constraint): number {
    const sourceValue = graph.getValue(constraint.source);
    if (sourceValue === undefined) return 0;
    
    const targets = Array.isArray(constraint.target) 
      ? constraint.target 
      : [constraint.target];
    
    let maxDelta = 0;
    
    for (const targetName of targets) {
      if (graph.isLocked(targetName)) continue;
      
      const targetValue = graph.getValue(targetName);
      if (targetValue === undefined) continue;
      
      const newValue = inhibit(targetValue, sourceValue, constraint.weight);
      graph.setValue(targetName, newValue);
      
      maxDelta = Math.max(maxDelta, Math.abs(newValue - targetValue));
    }
    
    return maxDelta;
  }

  /**
   * Applies a support constraint
   */
  private applySupport(graph: NeuroGraph, constraint: Constraint): number {
    const sourceValue = graph.getValue(constraint.source);
    if (sourceValue === undefined) return 0;
    
    const targets = Array.isArray(constraint.target) 
      ? constraint.target 
      : [constraint.target];
    
    let maxDelta = 0;
    
    for (const targetName of targets) {
      if (graph.isLocked(targetName)) continue;
      
      const targetValue = graph.getValue(targetName);
      if (targetValue === undefined) continue;
      
      const newValue = support(targetValue, sourceValue, constraint.weight);
      graph.setValue(targetName, newValue);
      
      maxDelta = Math.max(maxDelta, Math.abs(newValue - targetValue));
    }
    
    return maxDelta;
  }

  /**
   * Applies mutex constraints to ensure mutual exclusion
   */
  private applyMutex(graph: NeuroGraph, constraints: Constraint[]): number {
    // Collect all variables involved in mutex
    const variables = new Set<string>();
    for (const c of constraints) {
      variables.add(c.source);
      const targets = Array.isArray(c.target) ? c.target : [c.target];
      targets.forEach(t => variables.add(t));
    }
    
    // Get current values
    const names = Array.from(variables);
    const values: TruthValue[] = [];
    const locked: boolean[] = [];
    
    for (const name of names) {
      values.push(graph.getValue(name) ?? 0);
      locked.push(graph.isLocked(name));
    }
    
    // Only normalize unlocked variables
    const unlockedIndices = locked.map((l, i) => l ? -1 : i).filter(i => i >= 0);
    const lockedSum = locked.reduce((sum, l, i) => sum + (l ? values[i]! : 0), 0);
    
    // Available space for unlocked variables
    const available = Math.max(0, 1 - lockedSum);
    
    // Normalize unlocked variables
    const unlockedValues = unlockedIndices.map(i => values[i]!);
    const normalized = mutexNormalize(unlockedValues.map(v => v * available / Math.max(0.001, unlockedValues.reduce((a, b) => a + b, 0))));
    
    let maxDelta = 0;
    unlockedIndices.forEach((idx, i) => {
      const name = names[idx]!;
      const oldValue = values[idx]!;
      const newValue = normalized[i] ?? oldValue;
      
      if (!graph.isLocked(name)) {
        graph.setValue(name, newValue);
        maxDelta = Math.max(maxDelta, Math.abs(newValue - oldValue));
      }
    });
    
    return maxDelta;
  }

  /**
   * Gets a canonical key for mutex grouping
   */
  private getMutexKey(constraint: Constraint): string {
    const targets = Array.isArray(constraint.target) ? constraint.target : [constraint.target];
    const all = [constraint.source, ...targets].sort();
    return all.join('|');
  }

  // ==========================================================================
  // Full Inference
  // ==========================================================================

  /**
   * Runs full inference until convergence or max iterations
   * 
   * @param graph The graph to perform inference on
   * @param recordHistory Whether to record value history
   * @returns Inference result with final states
   */
  infer(graph: NeuroGraph, recordHistory = false): InferenceResult {
    const history: Array<Record<string, TruthValue>> = [];
    
    if (recordHistory) {
      history.push(graph.getAllValues());
    }
    
    let iteration = 0;
    let converged = false;
    
    while (iteration < this.config.maxIterations) {
      // Forward pass
      const ruleDelta = this.forwardPass(graph);
      
      // Apply constraints
      const constraintDelta = this.applyConstraints(graph);
      
      const totalDelta = Math.max(ruleDelta, constraintDelta);
      
      if (recordHistory) {
        history.push(graph.getAllValues());
      }
      
      iteration++;
      
      if (totalDelta < this.config.convergenceThreshold) {
        converged = true;
        break;
      }
    }
    
    // Build result
    const states: Record<string, { value: TruthValue; lower: TruthValue; upper: TruthValue; locked: boolean; gradient: number }> = {};
    for (const name of graph.getVariableNames()) {
      const state = graph.getState(name);
      if (state) {
        states[name] = { ...state };
      }
    }
    
    return {
      states,
      iterations: iteration,
      converged,
      history: recordHistory ? history : undefined
    };
  }

  /**
   * Runs inference with specific evidence
   * 
   * @param graph The graph to perform inference on
   * @param evidence Variable name -> value mappings to lock
   * @returns Inference result
   */
  inferWithEvidence(
    graph: NeuroGraph, 
    evidence: Record<string, TruthValue>
  ): InferenceResult {
    // Reset to priors first
    graph.resetToPriors();
    graph.unlockAll();
    
    // Lock evidence
    for (const [name, value] of Object.entries(evidence)) {
      graph.lockVariable(name, value);
    }
    
    // Run inference
    return this.infer(graph);
  }

  // ==========================================================================
  // Training / Learning
  // ==========================================================================

  /**
   * Computes loss for a single example
   */
  computeLoss(
    graph: NeuroGraph, 
    example: TrainingExample
  ): number {
    // Set inputs as evidence
    const result = this.inferWithEvidence(graph, example.inputs);
    
    // Compute MSE loss
    let totalLoss = 0;
    let count = 0;
    
    for (const [name, expected] of Object.entries(example.outputs)) {
      const actual = result.states[name]?.value ?? 0.5;
      totalLoss += Math.pow(expected - actual, 2);
      count++;
    }
    
    return count > 0 ? totalLoss / count : 0;
  }

  /**
   * Trains the graph on a set of examples
   * 
   * Uses gradient-free optimization (finite differences) for simplicity.
   * 
   * @param graph The graph to train
   * @param examples Training examples
   * @param epochs Number of training epochs
   * @returns Training result
   */
  train(
    graph: NeuroGraph, 
    examples: TrainingExample[], 
    epochs: number = 100
  ): TrainingResult {
    const lossHistory: number[] = [];
    const epsilon = 0.01; // For finite differences
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Compute average loss
      let epochLoss = 0;
      for (const example of examples) {
        epochLoss += this.computeLoss(graph, example);
      }
      epochLoss /= examples.length;
      lossHistory.push(epochLoss);
      
      // Early stopping if loss is very low
      if (epochLoss < 0.001) break;
      
      // Update learnable weights using gradient approximation
      const rules = graph.getRules().filter(r => r.learnable !== false);
      
      for (const rule of rules) {
        const currentWeight = rule.weight;
        
        // Compute gradient via finite differences
        graph.setRuleWeight(rule.id, clamp(currentWeight + epsilon));
        let lossPlus = 0;
        for (const example of examples) {
          lossPlus += this.computeLoss(graph, example);
        }
        lossPlus /= examples.length;
        
        graph.setRuleWeight(rule.id, clamp(currentWeight - epsilon));
        let lossMinus = 0;
        for (const example of examples) {
          lossMinus += this.computeLoss(graph, example);
        }
        lossMinus /= examples.length;
        
        // Approximate gradient
        const gradient = (lossPlus - lossMinus) / (2 * epsilon);
        
        // Update weight with gradient descent
        const newWeight = clamp(currentWeight - this.config.learningRate * gradient);
        graph.setRuleWeight(rule.id, newWeight);
      }
    }
    
    // Collect final weights
    const weights: Record<string, TruthValue> = {};
    for (const rule of graph.getRules()) {
      weights[rule.id] = rule.weight;
    }
    
    return {
      loss: lossHistory[lossHistory.length - 1] ?? 0,
      epochs: lossHistory.length,
      lossHistory,
      weights
    };
  }

  // ==========================================================================
  // Query Helpers
  // ==========================================================================

  /**
   * Queries the probability of a variable given evidence
   */
  query(
    graph: NeuroGraph,
    queryVariable: string,
    evidence: Record<string, TruthValue>
  ): TruthValue {
    const result = this.inferWithEvidence(graph, evidence);
    return result.states[queryVariable]?.value ?? 0.5;
  }

  /**
   * Finds the most likely explanation (MAP) for observed evidence
   * 
   * Returns the values of unobserved variables that best explain the evidence.
   */
  explain(
    graph: NeuroGraph,
    evidence: Record<string, TruthValue>
  ): Record<string, TruthValue> {
    const result = this.inferWithEvidence(graph, evidence);
    
    const explanation: Record<string, TruthValue> = {};
    for (const name of graph.getVariableNames()) {
      if (!(name in evidence)) {
        explanation[name] = result.states[name]?.value ?? 0.5;
      }
    }
    
    return explanation;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Creates a new inference engine with default configuration
 */
export function createEngine(config?: Partial<InferenceConfig>): InferenceEngine {
  return new InferenceEngine(config);
}

/**
 * Convenience function to run inference on a graph
 */
export function infer(graph: NeuroGraph, config?: Partial<InferenceConfig>): InferenceResult {
  const engine = new InferenceEngine(config);
  return engine.infer(graph);
}

/**
 * Convenience function to query a variable
 */
export function query(
  graph: NeuroGraph,
  queryVariable: string,
  evidence: Record<string, TruthValue>,
  config?: Partial<InferenceConfig>
): TruthValue {
  const engine = new InferenceEngine(config);
  return engine.query(graph, queryVariable, evidence);
}
