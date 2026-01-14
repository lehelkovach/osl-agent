"""
NeuroJSON Protocol Types (Python port)

Defines the serializable format for neurosymbolic logic graphs.
Compatible with the JavaScript NeuroSym.js library.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TypedDict


# Type aliases
TruthValue = float


class Variable(TypedDict, total=False):
    """A variable (node) in the logic graph."""
    type: str  # 'bool' or 'continuous'
    prior: float  # Initial truth value [0, 1]
    description: str  # Optional description
    locked: bool  # If true, value won't change during inference


class Rule(TypedDict, total=False):
    """A rule defining logical relationships between variables."""
    id: str
    type: str  # IMPLICATION, EQUIVALENCE, CONJUNCTION, DISJUNCTION
    inputs: List[str]  # Input variable names
    output: str  # Output variable name
    op: str  # AND, OR, NOT, IDENTITY, WEIGHTED
    weight: float  # Confidence in this rule [0, 1]
    learnable: bool  # Whether weight can be updated
    description: str


class Constraint(TypedDict, total=False):
    """A constraint (e.g., attack relation in argumentation)."""
    id: str
    type: str  # ATTACK, SUPPORT, MUTEX
    source: str  # Source variable
    target: str  # Target variable (or list)
    weight: float  # Strength of constraint
    description: str


class NeuroJSON(TypedDict, total=False):
    """Complete NeuroJSON document defining a logic graph."""
    version: str
    name: str
    description: str
    variables: Dict[str, Variable]
    rules: List[Rule]
    constraints: List[Constraint]
    metadata: Dict[str, Any]


@dataclass
class VariableState:
    """Runtime state of a variable during inference."""
    value: float = 0.5
    lower: float = 0.0
    upper: float = 1.0
    locked: bool = False
    gradient: float = 0.0


@dataclass
class EngineConfig:
    """Configuration for the inference engine."""
    max_iterations: int = 100
    convergence_threshold: float = 0.001
    learning_rate: float = 0.1
    damping_factor: float = 0.5


@dataclass
class InferenceResult:
    """Result of an inference pass."""
    states: Dict[str, VariableState]
    iterations: int
    converged: bool
    history: Optional[List[Dict[str, float]]] = None


def create_default_config() -> EngineConfig:
    """Creates a default inference config."""
    return EngineConfig()


def validate_neuro_json(doc: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validates a NeuroJSON document.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not isinstance(doc, dict):
        return False, ["Document must be a dictionary"]
    
    # Check version
    if "version" not in doc or not isinstance(doc.get("version"), str):
        errors.append("version: must be a string")
    
    # Check variables
    if "variables" not in doc or not isinstance(doc.get("variables"), dict):
        errors.append("variables: must be a dictionary")
    else:
        for name, var in doc["variables"].items():
            if not isinstance(var, dict):
                errors.append(f"variables.{name}: must be a dictionary")
                continue
            if var.get("type") not in ("bool", "continuous"):
                errors.append(f"variables.{name}.type: must be 'bool' or 'continuous'")
            prior = var.get("prior")
            if prior is not None and (not isinstance(prior, (int, float)) or prior < 0 or prior > 1):
                errors.append(f"variables.{name}.prior: must be between 0 and 1")
    
    # Check rules
    if "rules" not in doc or not isinstance(doc.get("rules"), list):
        errors.append("rules: must be a list")
    else:
        valid_types = {"IMPLICATION", "EQUIVALENCE", "CONJUNCTION", "DISJUNCTION"}
        for i, rule in enumerate(doc["rules"]):
            if not isinstance(rule, dict):
                errors.append(f"rules[{i}]: must be a dictionary")
                continue
            if not isinstance(rule.get("id"), str):
                errors.append(f"rules[{i}].id: must be a string")
            if rule.get("type") not in valid_types:
                errors.append(f"rules[{i}].type: must be one of {valid_types}")
            if not isinstance(rule.get("inputs"), list):
                errors.append(f"rules[{i}].inputs: must be a list")
            if not isinstance(rule.get("output"), str):
                errors.append(f"rules[{i}].output: must be a string")
            weight = rule.get("weight")
            if weight is not None and (not isinstance(weight, (int, float)) or weight < 0 or weight > 1):
                errors.append(f"rules[{i}].weight: must be between 0 and 1")
    
    # Check constraints
    if "constraints" not in doc or not isinstance(doc.get("constraints"), list):
        errors.append("constraints: must be a list")
    else:
        valid_constraint_types = {"ATTACK", "SUPPORT", "MUTEX"}
        for i, constraint in enumerate(doc["constraints"]):
            if not isinstance(constraint, dict):
                errors.append(f"constraints[{i}]: must be a dictionary")
                continue
            if not isinstance(constraint.get("id"), str):
                errors.append(f"constraints[{i}].id: must be a string")
            if constraint.get("type") not in valid_constraint_types:
                errors.append(f"constraints[{i}].type: must be one of {valid_constraint_types}")
    
    return len(errors) == 0, errors
