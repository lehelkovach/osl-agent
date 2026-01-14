"""
NeuroEngine Module (Python port)

The main entry point for neurosymbolic reasoning.
Combines graph management and inference into a clean API.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .logic import (
    clamp,
    fuzzy_and,
    fuzzy_or,
    equivalent,
    apply_operation,
    inhibit,
    support,
    mutex_normalize,
)
from .types import (
    NeuroJSON,
    Variable,
    Rule,
    Constraint,
    VariableState,
    EngineConfig,
    TruthValue,
    create_default_config,
)


# Type aliases
Evidence = Dict[str, TruthValue]
InferenceOutput = Dict[str, TruthValue]


@dataclass
class TrainingData:
    """Training example for learning."""
    inputs: Dict[str, TruthValue]
    targets: Dict[str, TruthValue]


class NeuroEngine:
    """
    Main entry point for NeuroSym Python.
    
    Combines graph state management and inference engine into a unified API.
    
    Example:
        schema = {
            "version": "1.0",
            "variables": {
                "raining": {"type": "bool", "prior": 0.3},
                "wet_ground": {"type": "bool", "prior": 0.1},
            },
            "rules": [{
                "id": "rain_wets",
                "type": "IMPLICATION",
                "inputs": ["raining"],
                "output": "wet_ground",
                "op": "IDENTITY",
                "weight": 0.95,
            }],
            "constraints": [],
        }
        
        engine = NeuroEngine(schema)
        result = engine.run({"raining": 1.0})
        print(result["wet_ground"])  # ~0.95
    """

    def __init__(self, schema: NeuroJSON, config: Optional[EngineConfig] = None):
        self.config = config or create_default_config()
        self._variables: Dict[str, Variable] = {}
        self._rules: Dict[str, Rule] = {}
        self._constraints: Dict[str, Constraint] = {}
        self._states: Dict[str, VariableState] = {}
        self._var_to_input_rules: Dict[str, List[str]] = {}
        self._var_to_output_rules: Dict[str, List[str]] = {}
        
        self._load(schema)

    def _load(self, schema: NeuroJSON) -> None:
        """Loads a NeuroJSON schema."""
        # Load variables
        for name, var in schema.get("variables", {}).items():
            self._variables[name] = var
            prior = var.get("prior", 0.5)
            self._states[name] = VariableState(
                value=prior,
                locked=var.get("locked", False),
            )
            self._var_to_input_rules[name] = []
            self._var_to_output_rules[name] = []
        
        # Load rules
        for rule in schema.get("rules", []):
            rule_id = rule["id"]
            self._rules[rule_id] = rule
            
            # Index by input/output
            for input_var in rule.get("inputs", []):
                if input_var in self._var_to_input_rules:
                    self._var_to_input_rules[input_var].append(rule_id)
            
            output_var = rule.get("output")
            if output_var and output_var in self._var_to_output_rules:
                self._var_to_output_rules[output_var].append(rule_id)
        
        # Load constraints
        for constraint in schema.get("constraints", []):
            self._constraints[constraint["id"]] = constraint

    # =========================================================================
    # Inference
    # =========================================================================

    def run(self, evidence: Optional[Evidence] = None, iterations: Optional[int] = None) -> InferenceOutput:
        """
        Runs inference with optional evidence.
        
        This is the main inference method. It:
        1. Resets the graph to priors
        2. Locks evidence variables (hard constraints)
        3. Propagates values through rules and constraints
        4. Returns the final state of all variables
        
        Args:
            evidence: Optional variable -> value mappings to lock
            iterations: Optional max iterations (defaults to config)
        
        Returns:
            Dict with all variable values after inference
        """
        max_iter = iterations or self.config.max_iterations
        
        # Reset to priors
        self._reset_to_priors()
        
        # Lock evidence
        if evidence:
            for name, value in evidence.items():
                if name in self._states:
                    self._states[name].value = clamp(value)
                    self._states[name].locked = True
        
        # Run inference loop
        for _ in range(max_iter):
            rule_delta = self._forward_pass()
            constraint_delta = self._apply_constraints()
            
            total_delta = max(rule_delta, constraint_delta)
            if total_delta < self.config.convergence_threshold:
                break
        
        return self._get_all_values()

    def query(self, variable: str, evidence: Optional[Evidence] = None) -> TruthValue:
        """Queries a specific variable given evidence."""
        result = self.run(evidence)
        return result.get(variable, 0.5)

    def _reset_to_priors(self) -> None:
        """Resets all variables to their prior values."""
        for name, var in self._variables.items():
            state = self._states[name]
            state.value = var.get("prior", 0.5)
            state.locked = var.get("locked", False)

    def _get_all_values(self) -> Dict[str, TruthValue]:
        """Gets current values of all variables."""
        return {name: state.value for name, state in self._states.items()}

    def _forward_pass(self) -> float:
        """Single forward pass through all rules."""
        max_delta = 0.0
        
        # Process variables in order (simple iteration, no topo sort needed for now)
        for var_name in self._variables:
            rule_ids = self._var_to_output_rules.get(var_name, [])
            if not rule_ids:
                continue
            
            state = self._states[var_name]
            if state.locked:
                continue
            
            # Compute contributions from all rules
            contributions: List[TruthValue] = []
            weights: List[float] = []
            
            for rule_id in rule_ids:
                rule = self._rules[rule_id]
                rule_value = self._evaluate_rule(rule)
                if rule_value is not None:
                    contributions.append(rule_value)
                    weights.append(rule.get("weight", 1.0))
            
            if not contributions:
                continue
            
            # Combine contributions (weighted average with damping)
            old_value = state.value
            total_weight = sum(weights)
            weighted_sum = sum(c * w for c, w in zip(contributions, weights))
            
            new_contribution = weighted_sum / total_weight if total_weight > 0 else 0.5
            damped_value = (
                self.config.damping_factor * new_contribution +
                (1 - self.config.damping_factor) * old_value
            )
            
            state.value = clamp(damped_value)
            max_delta = max(max_delta, abs(damped_value - old_value))
        
        return max_delta

    def _evaluate_rule(self, rule: Rule) -> Optional[TruthValue]:
        """Evaluates a single rule."""
        # Get input values
        input_values: List[TruthValue] = []
        for input_name in rule.get("inputs", []):
            state = self._states.get(input_name)
            if state is None:
                return None
            input_values.append(state.value)
        
        rule_type = rule.get("type", "IMPLICATION")
        op = rule.get("op", "IDENTITY")
        weight = rule.get("weight", 1.0)
        
        if rule_type == "IMPLICATION":
            antecedent = apply_operation(op, input_values)
            result = antecedent * weight
        elif rule_type == "CONJUNCTION":
            result = fuzzy_and(*input_values) * weight
        elif rule_type == "DISJUNCTION":
            result = fuzzy_or(*input_values) * weight
        elif rule_type == "EQUIVALENCE":
            if len(input_values) >= 2:
                result = equivalent(input_values[0], input_values[1]) * weight
            else:
                result = (input_values[0] if input_values else 0.5) * weight
        else:
            return None
        
        return clamp(result)

    def _apply_constraints(self) -> float:
        """Applies all constraints."""
        max_delta = 0.0
        
        for constraint in self._constraints.values():
            c_type = constraint.get("type", "")
            
            if c_type == "ATTACK":
                delta = self._apply_attack(constraint)
                max_delta = max(max_delta, delta)
            elif c_type == "SUPPORT":
                delta = self._apply_support(constraint)
                max_delta = max(max_delta, delta)
            # MUTEX handling would go here
        
        return max_delta

    def _apply_attack(self, constraint: Constraint) -> float:
        """Applies an attack constraint."""
        source = constraint.get("source", "")
        source_state = self._states.get(source)
        if source_state is None:
            return 0.0
        
        target = constraint.get("target", "")
        targets = [target] if isinstance(target, str) else target
        weight = constraint.get("weight", 1.0)
        
        max_delta = 0.0
        for target_name in targets:
            target_state = self._states.get(target_name)
            if target_state is None or target_state.locked:
                continue
            
            old_value = target_state.value
            new_value = inhibit(old_value, source_state.value, weight)
            target_state.value = new_value
            max_delta = max(max_delta, abs(new_value - old_value))
        
        return max_delta

    def _apply_support(self, constraint: Constraint) -> float:
        """Applies a support constraint."""
        source = constraint.get("source", "")
        source_state = self._states.get(source)
        if source_state is None:
            return 0.0
        
        target = constraint.get("target", "")
        targets = [target] if isinstance(target, str) else target
        weight = constraint.get("weight", 1.0)
        
        max_delta = 0.0
        for target_name in targets:
            target_state = self._states.get(target_name)
            if target_state is None or target_state.locked:
                continue
            
            old_value = target_state.value
            new_value = support(old_value, source_state.value, weight)
            target_state.value = new_value
            max_delta = max(max_delta, abs(new_value - old_value))
        
        return max_delta

    # =========================================================================
    # Training
    # =========================================================================

    def train(self, data: List[TrainingData], epochs: int = 100) -> float:
        """
        Trains the engine on examples.
        
        Uses the heuristic gradient descent formula:
        Weight_New = Weight_Old + (Error × LearningRate × Input_Strength)
        
        Args:
            data: List of training examples
            epochs: Number of training epochs
        
        Returns:
            Final average loss
        """
        final_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for example in data:
                output = self.run(example.inputs)
                
                for target_var, target_value in example.targets.items():
                    actual_value = output.get(target_var, 0.5)
                    error = target_value - actual_value
                    epoch_loss += error ** 2
                    
                    # Update weights
                    self._update_weights(target_var, error, example.inputs)
            
            epoch_loss /= len(data) if data else 1
            final_loss = epoch_loss
            
            if epoch_loss < 0.001:
                break
        
        return final_loss

    def _update_weights(self, target_var: str, error: float, inputs: Evidence) -> None:
        """Updates rule weights based on error."""
        rule_ids = self._var_to_output_rules.get(target_var, [])
        
        for rule_id in rule_ids:
            rule = self._rules.get(rule_id)
            if rule is None or rule.get("learnable") is False:
                continue
            
            # Calculate input strength
            input_strength = 0.0
            rule_inputs = rule.get("inputs", [])
            for input_name in rule_inputs:
                input_strength += inputs.get(input_name, self._states.get(input_name, VariableState()).value)
            input_strength /= len(rule_inputs) if rule_inputs else 1
            
            # Apply weight update formula
            delta = error * self.config.learning_rate * input_strength
            new_weight = clamp(rule.get("weight", 1.0) + delta)
            rule["weight"] = new_weight

    # =========================================================================
    # Export
    # =========================================================================

    def export(self) -> NeuroJSON:
        """Exports the current schema with learned weights."""
        return {
            "version": "1.0",
            "variables": dict(self._variables),
            "rules": list(self._rules.values()),
            "constraints": list(self._constraints.values()),
        }

    def export_state(self) -> Dict[str, TruthValue]:
        """Exports current variable states (for DB writeback)."""
        return self._get_all_values()

    # =========================================================================
    # Accessors
    # =========================================================================

    def get_variables(self) -> List[str]:
        """Gets all variable names."""
        return list(self._variables.keys())

    def get_rules(self) -> List[str]:
        """Gets all rule IDs."""
        return list(self._rules.keys())

    def get_rule_weight(self, rule_id: str) -> Optional[float]:
        """Gets a rule's weight."""
        rule = self._rules.get(rule_id)
        return rule.get("weight") if rule else None

    def set_rule_weight(self, rule_id: str, weight: float) -> bool:
        """Sets a rule's weight."""
        rule = self._rules.get(rule_id)
        if rule:
            rule["weight"] = clamp(weight)
            return True
        return False

    def get_value(self, variable: str) -> Optional[TruthValue]:
        """Gets current value of a variable."""
        state = self._states.get(variable)
        return state.value if state else None

    def set_value(self, variable: str, value: TruthValue) -> bool:
        """Sets value of a variable (if not locked)."""
        state = self._states.get(variable)
        if state and not state.locked:
            state.value = clamp(value)
            return True
        return False

    def lock_variable(self, variable: str, value: TruthValue) -> bool:
        """Locks a variable as evidence."""
        state = self._states.get(variable)
        if state:
            state.value = clamp(value)
            state.locked = True
            return True
        return False
