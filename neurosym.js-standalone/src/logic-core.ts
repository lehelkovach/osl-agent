/**
 * Logic Core Module
 * 
 * Implements fuzzy logic operations using Lukasiewicz T-Norms.
 * All functions are pure (no side effects) and operate on truth values in [0, 1].
 * 
 * Lukasiewicz logic is chosen because:
 * - It satisfies logical properties (associativity, commutativity, etc.)
 * - AND/OR are duals under negation
 * - It allows for "partial" truth values (fuzzy reasoning)
 * - Implication has a natural interpretation: A -> B is "at least as true as B"
 */

import { TruthValue, Operation } from './types';

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Clamps a value to the [0, 1] range
 */
export function clamp(value: number): TruthValue {
  return Math.max(0, Math.min(1, value));
}

/**
 * Checks if a value is a valid truth value
 */
export function isValidTruthValue(value: number): boolean {
  return typeof value === 'number' && !isNaN(value) && value >= 0 && value <= 1;
}

// ============================================================================
// Lukasiewicz T-Norms (Fuzzy Logic Operations)
// ============================================================================

/**
 * Lukasiewicz Negation: NOT(a) = 1 - a
 * 
 * @param a Truth value to negate
 * @returns Negated truth value
 */
export function not(a: TruthValue): TruthValue {
  return clamp(1 - a);
}

/**
 * Lukasiewicz T-Norm (AND): a ∧ b = max(0, a + b - 1)
 * 
 * Properties:
 * - Commutativity: a ∧ b = b ∧ a
 * - Associativity: (a ∧ b) ∧ c = a ∧ (b ∧ c)
 * - Identity: a ∧ 1 = a
 * - Annihilator: a ∧ 0 = 0
 * 
 * @param values Array of truth values to conjoin
 * @returns Conjunction of all values
 */
export function and(...values: TruthValue[]): TruthValue {
  if (values.length === 0) return 1; // Empty AND is true (identity)
  if (values.length === 1) return clamp(values[0]!);
  
  // For multiple values, apply pairwise: max(0, sum - (n-1))
  const sum = values.reduce((acc, v) => acc + v, 0);
  return clamp(sum - (values.length - 1));
}

/**
 * Lukasiewicz T-Conorm (OR): a ∨ b = min(1, a + b)
 * 
 * Properties:
 * - Commutativity: a ∨ b = b ∨ a
 * - Associativity: (a ∨ b) ∨ c = a ∨ (b ∨ c)
 * - Identity: a ∨ 0 = a
 * - Annihilator: a ∨ 1 = 1
 * 
 * @param values Array of truth values to disjoin
 * @returns Disjunction of all values
 */
export function or(...values: TruthValue[]): TruthValue {
  if (values.length === 0) return 0; // Empty OR is false (identity)
  if (values.length === 1) return clamp(values[0]!);
  
  // De Morgan's dual: a ∨ b = ¬(¬a ∧ ¬b)
  // Simplified: min(1, sum)
  const sum = values.reduce((acc, v) => acc + v, 0);
  return clamp(sum);
}

/**
 * Lukasiewicz Implication: a → b = min(1, 1 - a + b)
 * 
 * Intuitively: "The degree to which B is at least as true as A"
 * 
 * Properties:
 * - 1 → b = b (if premise is certain, conclusion follows)
 * - 0 → b = 1 (false implies anything - vacuous truth)
 * - a → 1 = 1 (anything implies true)
 * - a → 0 = ¬a (implicating false is negation)
 * 
 * @param antecedent The "if" part (a)
 * @param consequent The "then" part (b)
 * @returns Truth value of the implication
 */
export function implies(antecedent: TruthValue, consequent: TruthValue): TruthValue {
  return clamp(1 - antecedent + consequent);
}

/**
 * Lukasiewicz Equivalence: a ↔ b = 1 - |a - b|
 * 
 * Intuitively: "The degree to which A and B have the same truth value"
 * 
 * @param a First truth value
 * @param b Second truth value
 * @returns Truth value of equivalence
 */
export function equivalent(a: TruthValue, b: TruthValue): TruthValue {
  return clamp(1 - Math.abs(a - b));
}

// ============================================================================
// Weighted Operations
// ============================================================================

/**
 * Weighted average of truth values
 * 
 * @param values Array of [value, weight] pairs
 * @returns Weighted average truth value
 */
export function weightedAverage(values: Array<[TruthValue, number]>): TruthValue {
  if (values.length === 0) return 0.5; // Neutral prior
  
  let totalWeight = 0;
  let weightedSum = 0;
  
  for (const [value, weight] of values) {
    totalWeight += weight;
    weightedSum += value * weight;
  }
  
  if (totalWeight === 0) return 0.5;
  return clamp(weightedSum / totalWeight);
}

/**
 * Weighted AND: applies weights to inputs before conjunction
 * 
 * Each input is softened toward 1 based on its weight:
 * softened = weight * value + (1 - weight) * 1
 * 
 * @param inputs Array of [value, weight] pairs
 * @returns Weighted conjunction
 */
export function weightedAnd(inputs: Array<[TruthValue, number]>): TruthValue {
  if (inputs.length === 0) return 1;
  
  const softened = inputs.map(([value, weight]) => {
    // Weight of 1 keeps original value, weight of 0 ignores it (treats as 1)
    return weight * value + (1 - weight);
  });
  
  return and(...softened);
}

/**
 * Weighted OR: applies weights to inputs before disjunction
 * 
 * Each input is softened toward 0 based on its weight:
 * softened = weight * value
 * 
 * @param inputs Array of [value, weight] pairs
 * @returns Weighted disjunction
 */
export function weightedOr(inputs: Array<[TruthValue, number]>): TruthValue {
  if (inputs.length === 0) return 0;
  
  const softened = inputs.map(([value, weight]) => {
    // Weight of 1 keeps original value, weight of 0 ignores it (treats as 0)
    return weight * value;
  });
  
  return or(...softened);
}

// ============================================================================
// Argumentation Logic (Inhibition/Attack Relations)
// ============================================================================

/**
 * Attack inhibition: reduces target's truth value based on attacker's strength
 * 
 * Implements: target' = target * (1 - attacker * weight)
 * 
 * When attacker is fully true (1) and weight is 1, target becomes 0.
 * When attacker is false (0), target is unchanged.
 * 
 * @param targetValue Current truth value of target
 * @param attackerValue Truth value of the attacking argument
 * @param weight Strength of the attack relation
 * @returns Inhibited target value
 */
export function inhibit(
  targetValue: TruthValue,
  attackerValue: TruthValue,
  weight: TruthValue = 1.0
): TruthValue {
  const inhibitionFactor = attackerValue * weight;
  return clamp(targetValue * (1 - inhibitionFactor));
}

/**
 * Support reinforcement: increases target's truth value based on supporter
 * 
 * Implements: target' = target + (1 - target) * supporter * weight
 * 
 * This ensures the result stays in [0, 1] while moving target toward 1.
 * 
 * @param targetValue Current truth value of target
 * @param supporterValue Truth value of the supporting argument
 * @param weight Strength of the support relation
 * @returns Reinforced target value
 */
export function support(
  targetValue: TruthValue,
  supporterValue: TruthValue,
  weight: TruthValue = 1.0
): TruthValue {
  const supportFactor = supporterValue * weight;
  return clamp(targetValue + (1 - targetValue) * supportFactor);
}

/**
 * Mutual exclusion constraint: ensures values sum to at most 1
 * 
 * Normalizes values proportionally if their sum exceeds 1.
 * 
 * @param values Array of truth values that should be mutually exclusive
 * @returns Normalized values that sum to at most 1
 */
export function mutexNormalize(values: TruthValue[]): TruthValue[] {
  if (values.length === 0) return [];
  
  const sum = values.reduce((acc, v) => acc + v, 0);
  
  if (sum <= 1) return values.map(clamp);
  
  // Normalize proportionally
  return values.map(v => clamp(v / sum));
}

// ============================================================================
// Operation Dispatcher
// ============================================================================

/**
 * Applies an operation to input values
 * 
 * @param op The operation to apply
 * @param inputs Array of input truth values
 * @param weights Optional weights for WEIGHTED operation
 * @returns Result of the operation
 */
export function applyOperation(
  op: Operation,
  inputs: TruthValue[],
  weights?: number[]
): TruthValue {
  switch (op) {
    case 'IDENTITY':
      return inputs.length > 0 ? clamp(inputs[0]!) : 0.5;
    
    case 'AND':
      return and(...inputs);
    
    case 'OR':
      return or(...inputs);
    
    case 'NOT':
      return inputs.length > 0 ? not(inputs[0]!) : 0.5;
    
    case 'WEIGHTED':
      if (!weights || weights.length !== inputs.length) {
        // Fall back to simple average if weights don't match
        return inputs.length > 0 
          ? inputs.reduce((a, b) => a + b, 0) / inputs.length 
          : 0.5;
      }
      return weightedAverage(inputs.map((v, i) => [v, weights[i]!]));
    
    default:
      throw new Error(`Unknown operation: ${op as string}`);
  }
}

// ============================================================================
// Gradient Computations (for Learning)
// ============================================================================

/**
 * Computes the gradient of AND w.r.t. each input
 * 
 * For Lukasiewicz AND, the gradient is 1 for all inputs when the result > 0,
 * and 0 otherwise (subgradient at the boundary).
 * 
 * @param inputs Input truth values
 * @param outputGradient Gradient from the output
 * @returns Array of gradients for each input
 */
export function andGradient(inputs: TruthValue[], outputGradient: number): number[] {
  const result = and(...inputs);
  // Gradient flows through if result > 0 (not hitting the max(0, ...) boundary)
  if (result > 0) {
    return inputs.map(() => outputGradient);
  }
  return inputs.map(() => 0);
}

/**
 * Computes the gradient of OR w.r.t. each input
 * 
 * @param inputs Input truth values
 * @param outputGradient Gradient from the output
 * @returns Array of gradients for each input
 */
export function orGradient(inputs: TruthValue[], outputGradient: number): number[] {
  const result = or(...inputs);
  // Gradient flows through if result < 1 (not hitting the min(1, ...) boundary)
  if (result < 1) {
    return inputs.map(() => outputGradient);
  }
  return inputs.map(() => 0);
}

/**
 * Computes the gradient of implication w.r.t. antecedent and consequent
 * 
 * d(implies)/d(a) = -1 (when not saturated)
 * d(implies)/d(b) = 1 (when not saturated)
 * 
 * @param antecedent Input antecedent value
 * @param consequent Input consequent value
 * @param outputGradient Gradient from the output
 * @returns Gradients for [antecedent, consequent]
 */
export function impliesGradient(
  antecedent: TruthValue, 
  consequent: TruthValue, 
  outputGradient: number
): [number, number] {
  const result = implies(antecedent, consequent);
  // Gradient flows if not saturated at 1
  if (result < 1) {
    return [-outputGradient, outputGradient];
  }
  return [0, 0];
}

/**
 * Computes the gradient of inhibition w.r.t. target and attacker
 * 
 * target' = target * (1 - attacker * weight)
 * d(target')/d(target) = (1 - attacker * weight)
 * d(target')/d(attacker) = -target * weight
 * 
 * @param targetValue Target's current value
 * @param attackerValue Attacker's value
 * @param weight Attack weight
 * @param outputGradient Gradient from output
 * @returns Gradients for [target, attacker]
 */
export function inhibitGradient(
  targetValue: TruthValue,
  attackerValue: TruthValue,
  weight: TruthValue,
  outputGradient: number
): [number, number] {
  const dTarget = (1 - attackerValue * weight) * outputGradient;
  const dAttacker = -targetValue * weight * outputGradient;
  return [dTarget, dAttacker];
}
