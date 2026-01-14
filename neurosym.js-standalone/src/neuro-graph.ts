/**
 * NeuroGraph Module
 * 
 * An in-memory graph structure representing the "World State" for neurosymbolic reasoning.
 * Manages variables (nodes), rules, constraints, and their runtime states.
 * 
 * Key responsibilities:
 * - Load/export NeuroJSON documents
 * - Track variable states (current values, bounds, locks)
 * - Provide efficient access patterns for inference
 * - Handle evidence injection (locking variables to observed values)
 */

import {
  NeuroJSON,
  Variable,
  Rule,
  Constraint,
  VariableState,
  TruthValue,
  ValidationResult,
  validateNeuroJSON
} from './types';
import { clamp } from './logic-core';

// ============================================================================
// NeuroGraph Class
// ============================================================================

/**
 * In-memory graph structure for neurosymbolic reasoning
 */
export class NeuroGraph {
  /** Schema version */
  readonly version: string;
  
  /** Graph name */
  readonly name: string;
  
  /** Variable definitions */
  private variables: Map<string, Variable>;
  
  /** Rules indexed by ID */
  private rules: Map<string, Rule>;
  
  /** Constraints indexed by ID */
  private constraints: Map<string, Constraint>;
  
  /** Current runtime states for variables */
  private states: Map<string, VariableState>;
  
  /** Index: variable -> rules where it appears as input */
  private variableToInputRules: Map<string, Set<string>>;
  
  /** Index: variable -> rules where it appears as output */
  private variableToOutputRules: Map<string, Set<string>>;
  
  /** Index: variable -> constraints where it's the source */
  private variableToSourceConstraints: Map<string, Set<string>>;
  
  /** Index: variable -> constraints where it's the target */
  private variableToTargetConstraints: Map<string, Set<string>>;

  constructor(doc?: NeuroJSON) {
    this.version = doc?.version ?? '1.0';
    this.name = doc?.name ?? 'unnamed';
    this.variables = new Map();
    this.rules = new Map();
    this.constraints = new Map();
    this.states = new Map();
    this.variableToInputRules = new Map();
    this.variableToOutputRules = new Map();
    this.variableToSourceConstraints = new Map();
    this.variableToTargetConstraints = new Map();
    
    if (doc) {
      this.load(doc);
    }
  }

  // ==========================================================================
  // Loading and Exporting
  // ==========================================================================

  /**
   * Loads a NeuroJSON document into the graph
   */
  load(doc: NeuroJSON): ValidationResult {
    // Validate first
    const validation = validateNeuroJSON(doc);
    if (!validation.valid) {
      return validation;
    }
    
    // Clear existing data
    this.clear();
    
    // Load variables
    for (const [name, variable] of Object.entries(doc.variables)) {
      this.addVariable(name, variable);
    }
    
    // Load rules
    for (const rule of doc.rules) {
      this.addRule(rule);
    }
    
    // Load constraints
    for (const constraint of doc.constraints) {
      this.addConstraint(constraint);
    }
    
    return { valid: true, errors: [] };
  }

  /**
   * Exports the graph as a NeuroJSON document
   */
  export(): NeuroJSON {
    const variables: Record<string, Variable> = {};
    for (const [name, variable] of this.variables) {
      variables[name] = { ...variable };
    }
    
    return {
      version: this.version,
      name: this.name,
      variables,
      rules: Array.from(this.rules.values()).map(r => ({ ...r })),
      constraints: Array.from(this.constraints.values()).map(c => ({ ...c }))
    };
  }

  /**
   * Clears all data from the graph
   */
  clear(): void {
    this.variables.clear();
    this.rules.clear();
    this.constraints.clear();
    this.states.clear();
    this.variableToInputRules.clear();
    this.variableToOutputRules.clear();
    this.variableToSourceConstraints.clear();
    this.variableToTargetConstraints.clear();
  }

  // ==========================================================================
  // Variable Management
  // ==========================================================================

  /**
   * Adds a variable to the graph
   */
  addVariable(name: string, variable: Variable): void {
    this.variables.set(name, { ...variable });
    
    // Initialize runtime state
    this.states.set(name, {
      value: variable.prior,
      lower: 0,
      upper: 1,
      locked: variable.locked ?? false,
      gradient: 0
    });
    
    // Initialize indexes
    if (!this.variableToInputRules.has(name)) {
      this.variableToInputRules.set(name, new Set());
    }
    if (!this.variableToOutputRules.has(name)) {
      this.variableToOutputRules.set(name, new Set());
    }
    if (!this.variableToSourceConstraints.has(name)) {
      this.variableToSourceConstraints.set(name, new Set());
    }
    if (!this.variableToTargetConstraints.has(name)) {
      this.variableToTargetConstraints.set(name, new Set());
    }
  }

  /**
   * Gets a variable definition
   */
  getVariable(name: string): Variable | undefined {
    return this.variables.get(name);
  }

  /**
   * Gets all variable names
   */
  getVariableNames(): string[] {
    return Array.from(this.variables.keys());
  }

  /**
   * Checks if a variable exists
   */
  hasVariable(name: string): boolean {
    return this.variables.has(name);
  }

  /**
   * Gets the number of variables
   */
  get variableCount(): number {
    return this.variables.size;
  }

  // ==========================================================================
  // State Management
  // ==========================================================================

  /**
   * Gets the current state of a variable
   */
  getState(name: string): VariableState | undefined {
    return this.states.get(name);
  }

  /**
   * Gets the current truth value of a variable
   */
  getValue(name: string): TruthValue | undefined {
    return this.states.get(name)?.value;
  }

  /**
   * Sets the truth value of a variable (if not locked)
   */
  setValue(name: string, value: TruthValue): boolean {
    const state = this.states.get(name);
    if (!state) return false;
    if (state.locked) return false;
    
    state.value = clamp(value);
    return true;
  }

  /**
   * Gets all current variable values
   */
  getAllValues(): Record<string, TruthValue> {
    const values: Record<string, TruthValue> = {};
    for (const [name, state] of this.states) {
      values[name] = state.value;
    }
    return values;
  }

  /**
   * Sets multiple variable values at once
   */
  setValues(values: Record<string, TruthValue>): void {
    for (const [name, value] of Object.entries(values)) {
      this.setValue(name, value);
    }
  }

  /**
   * Resets all variables to their prior values
   */
  resetToPriors(): void {
    for (const [name, variable] of this.variables) {
      const state = this.states.get(name);
      if (state && !state.locked) {
        state.value = variable.prior;
        state.gradient = 0;
      }
    }
  }

  /**
   * Locks a variable to a specific value (evidence injection)
   */
  lockVariable(name: string, value: TruthValue): boolean {
    const state = this.states.get(name);
    if (!state) return false;
    
    state.value = clamp(value);
    state.locked = true;
    return true;
  }

  /**
   * Unlocks a variable
   */
  unlockVariable(name: string): boolean {
    const state = this.states.get(name);
    if (!state) return false;
    
    state.locked = false;
    return true;
  }

  /**
   * Unlocks all variables
   */
  unlockAll(): void {
    for (const state of this.states.values()) {
      state.locked = false;
    }
  }

  /**
   * Checks if a variable is locked
   */
  isLocked(name: string): boolean {
    return this.states.get(name)?.locked ?? false;
  }

  /**
   * Gets the gradient for a variable
   */
  getGradient(name: string): number {
    return this.states.get(name)?.gradient ?? 0;
  }

  /**
   * Sets the gradient for a variable
   */
  setGradient(name: string, gradient: number): void {
    const state = this.states.get(name);
    if (state) {
      state.gradient = gradient;
    }
  }

  /**
   * Accumulates gradient for a variable
   */
  accumulateGradient(name: string, delta: number): void {
    const state = this.states.get(name);
    if (state) {
      state.gradient += delta;
    }
  }

  /**
   * Clears all gradients
   */
  clearGradients(): void {
    for (const state of this.states.values()) {
      state.gradient = 0;
    }
  }

  // ==========================================================================
  // Rule Management
  // ==========================================================================

  /**
   * Adds a rule to the graph
   */
  addRule(rule: Rule): void {
    this.rules.set(rule.id, { ...rule });
    
    // Update indexes
    for (const input of rule.inputs) {
      let inputRules = this.variableToInputRules.get(input);
      if (!inputRules) {
        inputRules = new Set();
        this.variableToInputRules.set(input, inputRules);
      }
      inputRules.add(rule.id);
    }
    
    let outputRules = this.variableToOutputRules.get(rule.output);
    if (!outputRules) {
      outputRules = new Set();
      this.variableToOutputRules.set(rule.output, outputRules);
    }
    outputRules.add(rule.id);
  }

  /**
   * Gets a rule by ID
   */
  getRule(id: string): Rule | undefined {
    return this.rules.get(id);
  }

  /**
   * Gets all rules
   */
  getRules(): Rule[] {
    return Array.from(this.rules.values());
  }

  /**
   * Gets rules that use a variable as input
   */
  getRulesWithInput(variableName: string): Rule[] {
    const ruleIds = this.variableToInputRules.get(variableName);
    if (!ruleIds) return [];
    return Array.from(ruleIds)
      .map(id => this.rules.get(id))
      .filter((r): r is Rule => r !== undefined);
  }

  /**
   * Gets rules that output to a variable
   */
  getRulesWithOutput(variableName: string): Rule[] {
    const ruleIds = this.variableToOutputRules.get(variableName);
    if (!ruleIds) return [];
    return Array.from(ruleIds)
      .map(id => this.rules.get(id))
      .filter((r): r is Rule => r !== undefined);
  }

  /**
   * Updates a rule's weight
   */
  setRuleWeight(ruleId: string, weight: TruthValue): boolean {
    const rule = this.rules.get(ruleId);
    if (!rule) return false;
    rule.weight = clamp(weight);
    return true;
  }

  /**
   * Gets the number of rules
   */
  get ruleCount(): number {
    return this.rules.size;
  }

  // ==========================================================================
  // Constraint Management
  // ==========================================================================

  /**
   * Adds a constraint to the graph
   */
  addConstraint(constraint: Constraint): void {
    this.constraints.set(constraint.id, { ...constraint });
    
    // Update source index
    let sourceConstraints = this.variableToSourceConstraints.get(constraint.source);
    if (!sourceConstraints) {
      sourceConstraints = new Set();
      this.variableToSourceConstraints.set(constraint.source, sourceConstraints);
    }
    sourceConstraints.add(constraint.id);
    
    // Update target index(es)
    const targets = Array.isArray(constraint.target) ? constraint.target : [constraint.target];
    for (const target of targets) {
      let targetConstraints = this.variableToTargetConstraints.get(target);
      if (!targetConstraints) {
        targetConstraints = new Set();
        this.variableToTargetConstraints.set(target, targetConstraints);
      }
      targetConstraints.add(constraint.id);
    }
  }

  /**
   * Gets a constraint by ID
   */
  getConstraint(id: string): Constraint | undefined {
    return this.constraints.get(id);
  }

  /**
   * Gets all constraints
   */
  getConstraints(): Constraint[] {
    return Array.from(this.constraints.values());
  }

  /**
   * Gets constraints where variable is the source
   */
  getConstraintsFromSource(variableName: string): Constraint[] {
    const constraintIds = this.variableToSourceConstraints.get(variableName);
    if (!constraintIds) return [];
    return Array.from(constraintIds)
      .map(id => this.constraints.get(id))
      .filter((c): c is Constraint => c !== undefined);
  }

  /**
   * Gets constraints where variable is a target
   */
  getConstraintsToTarget(variableName: string): Constraint[] {
    const constraintIds = this.variableToTargetConstraints.get(variableName);
    if (!constraintIds) return [];
    return Array.from(constraintIds)
      .map(id => this.constraints.get(id))
      .filter((c): c is Constraint => c !== undefined);
  }

  /**
   * Gets the number of constraints
   */
  get constraintCount(): number {
    return this.constraints.size;
  }

  // ==========================================================================
  // Graph Analysis
  // ==========================================================================

  /**
   * Gets the topological order of variables for forward inference
   * 
   * Returns variables sorted so that dependencies come before dependents.
   * Variables with no dependencies come first.
   */
  getTopologicalOrder(): string[] {
    const inDegree = new Map<string, number>();
    const result: string[] = [];
    
    // Initialize in-degrees
    for (const name of this.variables.keys()) {
      inDegree.set(name, 0);
    }
    
    // Count incoming edges from rules
    for (const rule of this.rules.values()) {
      const current = inDegree.get(rule.output) ?? 0;
      inDegree.set(rule.output, current + 1);
    }
    
    // Kahn's algorithm
    const queue = Array.from(this.variables.keys())
      .filter(name => (inDegree.get(name) ?? 0) === 0);
    
    while (queue.length > 0) {
      const node = queue.shift()!;
      result.push(node);
      
      // Process rules where this node is an input
      const affectedRules = this.getRulesWithInput(node);
      for (const rule of affectedRules) {
        const current = inDegree.get(rule.output) ?? 0;
        const newDegree = current - 1;
        inDegree.set(rule.output, newDegree);
        if (newDegree === 0) {
          queue.push(rule.output);
        }
      }
    }
    
    // If we didn't process all nodes, there's a cycle
    // In that case, add remaining nodes at the end
    for (const name of this.variables.keys()) {
      if (!result.includes(name)) {
        result.push(name);
      }
    }
    
    return result;
  }

  /**
   * Gets variables that have no incoming rules (source nodes)
   */
  getSourceVariables(): string[] {
    return Array.from(this.variables.keys())
      .filter(name => {
        const rules = this.variableToOutputRules.get(name);
        return !rules || rules.size === 0;
      });
  }

  /**
   * Gets variables that have no outgoing rules (sink nodes)
   */
  getSinkVariables(): string[] {
    return Array.from(this.variables.keys())
      .filter(name => {
        const rules = this.variableToInputRules.get(name);
        return !rules || rules.size === 0;
      });
  }

  /**
   * Creates a deep copy of the graph
   */
  clone(): NeuroGraph {
    return new NeuroGraph(this.export());
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Creates a new empty NeuroGraph
 */
export function createGraph(name?: string): NeuroGraph {
  return new NeuroGraph({
    version: '1.0',
    name,
    variables: {},
    rules: [],
    constraints: []
  });
}

/**
 * Creates a NeuroGraph from a JSON string
 */
export function parseNeuroJSON(json: string): NeuroGraph {
  const doc = JSON.parse(json) as NeuroJSON;
  return new NeuroGraph(doc);
}
