"""
Logic Core Module (Python port)

Implements fuzzy logic operations using Lukasiewicz T-Norms.
All functions are pure (no side effects) and operate on truth values in [0, 1].
"""

from typing import List, Tuple


def clamp(value: float) -> float:
    """Clamps a value to the [0, 1] range."""
    return max(0.0, min(1.0, value))


def is_valid_truth_value(value: float) -> bool:
    """Checks if a value is a valid truth value."""
    return isinstance(value, (int, float)) and 0.0 <= value <= 1.0


# =============================================================================
# Lukasiewicz T-Norms (Fuzzy Logic Operations)
# =============================================================================

def fuzzy_not(a: float) -> float:
    """Lukasiewicz Negation: NOT(a) = 1 - a"""
    return clamp(1.0 - a)


def fuzzy_and(*values: float) -> float:
    """
    Lukasiewicz T-Norm (AND): max(0, sum(inputs) - (n-1))
    
    Properties:
    - Commutativity: a ∧ b = b ∧ a
    - Associativity: (a ∧ b) ∧ c = a ∧ (b ∧ c)
    - Identity: a ∧ 1 = a
    - Annihilator: a ∧ 0 = 0
    """
    if len(values) == 0:
        return 1.0  # Empty AND is true (identity)
    if len(values) == 1:
        return clamp(values[0])
    
    total = sum(values)
    return clamp(total - (len(values) - 1))


def fuzzy_or(*values: float) -> float:
    """
    Lukasiewicz T-Conorm (OR): min(1, sum(inputs))
    
    Properties:
    - Commutativity: a ∨ b = b ∨ a
    - Associativity: (a ∨ b) ∨ c = a ∨ (b ∨ c)
    - Identity: a ∨ 0 = a
    - Annihilator: a ∨ 1 = 1
    """
    if len(values) == 0:
        return 0.0  # Empty OR is false (identity)
    if len(values) == 1:
        return clamp(values[0])
    
    total = sum(values)
    return clamp(total)


def implies(antecedent: float, consequent: float) -> float:
    """
    Lukasiewicz Implication: a → b = min(1, 1 - a + b)
    
    Intuitively: "The degree to which B is at least as true as A"
    
    Properties:
    - 1 → b = b (if premise is certain, conclusion follows)
    - 0 → b = 1 (false implies anything - vacuous truth)
    - a → 1 = 1 (anything implies true)
    - a → 0 = ¬a (implicating false is negation)
    """
    return clamp(1.0 - antecedent + consequent)


def equivalent(a: float, b: float) -> float:
    """
    Lukasiewicz Equivalence: a ↔ b = 1 - |a - b|
    
    Intuitively: "The degree to which A and B have the same truth value"
    """
    return clamp(1.0 - abs(a - b))


# =============================================================================
# Weighted Operations
# =============================================================================

def weighted_average(values: List[Tuple[float, float]]) -> float:
    """
    Weighted average of truth values.
    
    Args:
        values: List of (value, weight) tuples
    
    Returns:
        Weighted average truth value
    """
    if not values:
        return 0.5  # Neutral prior
    
    total_weight = sum(w for _, w in values)
    if total_weight == 0:
        return 0.5
    
    weighted_sum = sum(v * w for v, w in values)
    return clamp(weighted_sum / total_weight)


# =============================================================================
# Argumentation Logic (Inhibition/Attack Relations)
# =============================================================================

def inhibit(target_value: float, attacker_value: float, weight: float = 1.0) -> float:
    """
    Attack inhibition: reduces target's truth value based on attacker's strength.
    
    Formula: target' = target * (1 - attacker * weight)
    
    When attacker is fully true (1) and weight is 1, target becomes 0.
    When attacker is false (0), target is unchanged.
    """
    inhibition_factor = attacker_value * weight
    return clamp(target_value * (1.0 - inhibition_factor))


def support(target_value: float, supporter_value: float, weight: float = 1.0) -> float:
    """
    Support reinforcement: increases target's truth value based on supporter.
    
    Formula: target' = target + (1 - target) * supporter * weight
    
    This ensures the result stays in [0, 1] while moving target toward 1.
    """
    support_factor = supporter_value * weight
    return clamp(target_value + (1.0 - target_value) * support_factor)


def mutex_normalize(values: List[float]) -> List[float]:
    """
    Mutual exclusion constraint: ensures values sum to at most 1.
    Normalizes values proportionally if their sum exceeds 1.
    """
    if not values:
        return []
    
    total = sum(values)
    if total <= 1.0:
        return [clamp(v) for v in values]
    
    # Normalize proportionally
    return [clamp(v / total) for v in values]


# =============================================================================
# Operation Dispatcher
# =============================================================================

def apply_operation(op: str, inputs: List[float], weights: List[float] = None) -> float:
    """
    Applies an operation to input values.
    
    Args:
        op: Operation name (IDENTITY, AND, OR, NOT, WEIGHTED)
        inputs: List of input truth values
        weights: Optional weights for WEIGHTED operation
    
    Returns:
        Result of the operation
    """
    op = op.upper()
    
    if op == "IDENTITY":
        return clamp(inputs[0]) if inputs else 0.5
    elif op == "AND":
        return fuzzy_and(*inputs)
    elif op == "OR":
        return fuzzy_or(*inputs)
    elif op == "NOT":
        return fuzzy_not(inputs[0]) if inputs else 0.5
    elif op == "WEIGHTED":
        if weights and len(weights) == len(inputs):
            return weighted_average(list(zip(inputs, weights)))
        return sum(inputs) / len(inputs) if inputs else 0.5
    else:
        raise ValueError(f"Unknown operation: {op}")
