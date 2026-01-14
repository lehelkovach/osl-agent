"""
NeuroService Module

Integrates NeuroSym reasoning with KnowShowGo graph database.
Converts KSG Nodes/Associations to NeuroJSON and runs inference.

Usage:
    from knowshowgo.neuro_service import NeuroService
    
    service = NeuroService(db)
    
    # Run inference on a context window around a node
    updated_nodes = await service.solve_context(center_node_id, depth=2)
    
    # Query a specific node's truth value given evidence
    truth = service.query_node(node_id, evidence={"rain_node": 1.0})
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass

from .models import Node, Association, LogicType, LogicMeta
from .neuro import NeuroEngine
from .neuro.types import NeuroJSON, Variable, Rule, Constraint, TruthValue


@dataclass
class ContextGraph:
    """A subgraph extracted from the database for inference."""
    nodes: Dict[str, Node]  # node_id -> Node
    associations: Dict[str, Association]  # assoc_id -> Association
    center_node_id: str


class NeuroService:
    """
    Service that bridges KnowShowGo graph with NeuroSym reasoning.
    
    Converts KSG Nodes/Edges to NeuroJSON format and runs inference.
    Supports:
    - Context-windowed inference (solveContext)
    - Evidence injection
    - Grounding of first-order rules to instances
    """

    def __init__(self, db=None):
        """
        Initialize NeuroService.
        
        Args:
            db: Optional database adapter with methods like:
                - get_node(id) -> Node
                - get_neighborhood(center_id, depth) -> (nodes, associations)
                - bulk_update_nodes(updates)
        """
        self.db = db
        self.config = {
            "max_iterations": 50,
            "convergence_threshold": 0.001,
            "learning_rate": 0.1,
            "damping_factor": 0.5,
        }

    def set_config(self, **kwargs) -> None:
        """Updates service configuration."""
        self.config.update(kwargs)

    # =========================================================================
    # Context Extraction
    # =========================================================================

    def extract_context(
        self,
        nodes: List[Node],
        associations: List[Association],
        center_node_id: str,
    ) -> ContextGraph:
        """
        Extracts a context graph from nodes and associations.
        
        This is used when you already have the subgraph loaded.
        """
        return ContextGraph(
            nodes={n.id: n for n in nodes},
            associations={a.id: a for a in associations},
            center_node_id=center_node_id,
        )

    async def fetch_context(self, center_node_id: str, depth: int = 2) -> ContextGraph:
        """
        Fetches a context subgraph from the database.
        
        Uses neighborhood/context window approach - doesn't load whole DB.
        
        Args:
            center_node_id: ID of the center node
            depth: Number of hops to include
        
        Returns:
            ContextGraph with nodes and associations
        """
        if self.db is None:
            raise RuntimeError("Database not configured")
        
        nodes, associations = await self.db.get_neighborhood(center_node_id, depth)
        return self.extract_context(nodes, associations, center_node_id)

    # =========================================================================
    # Conversion to NeuroJSON
    # =========================================================================

    def to_neuro_json(self, context: ContextGraph) -> NeuroJSON:
        """
        Converts a KSG context graph to NeuroJSON format.
        
        Mapping:
        - KSG Nodes -> NeuroJSON Variables
        - KSG Associations with logic_meta -> NeuroJSON Rules/Constraints
        """
        variables: Dict[str, Variable] = {}
        rules: List[Rule] = []
        constraints: List[Constraint] = []
        
        # Convert nodes to variables
        for node_id, node in context.nodes.items():
            var_name = self._node_to_var_name(node)
            variables[var_name] = {
                "type": "bool",
                "prior": node.prior,
                "locked": node.is_locked,
            }
        
        # Convert associations to rules/constraints
        rule_counter = 0
        constraint_counter = 0
        
        for assoc_id, assoc in context.associations.items():
            if assoc.logic_meta is None:
                continue
            
            source_node = context.nodes.get(assoc.source_id)
            target_node = context.nodes.get(assoc.target_id)
            
            if source_node is None or target_node is None:
                continue
            
            source_var = self._node_to_var_name(source_node)
            target_var = self._node_to_var_name(target_node)
            logic = assoc.logic_meta
            
            if logic.type == LogicType.IMPLIES:
                rule_counter += 1
                rules.append({
                    "id": f"rule_{rule_counter}_{assoc.id[:8]}",
                    "type": "IMPLICATION",
                    "inputs": [source_var],
                    "output": target_var,
                    "op": logic.op,
                    "weight": logic.weight,
                    "learnable": logic.learnable,
                })
            elif logic.type == LogicType.ATTACKS:
                constraint_counter += 1
                constraints.append({
                    "id": f"attack_{constraint_counter}_{assoc.id[:8]}",
                    "type": "ATTACK",
                    "source": source_var,
                    "target": target_var,
                    "weight": logic.weight,
                })
            elif logic.type == LogicType.SUPPORTS:
                constraint_counter += 1
                constraints.append({
                    "id": f"support_{constraint_counter}_{assoc.id[:8]}",
                    "type": "SUPPORT",
                    "source": source_var,
                    "target": target_var,
                    "weight": logic.weight,
                })
            elif logic.type == LogicType.DEPENDS:
                # DEPENDS can be modeled as a weaker implication
                rule_counter += 1
                rules.append({
                    "id": f"depends_{rule_counter}_{assoc.id[:8]}",
                    "type": "IMPLICATION",
                    "inputs": [source_var],
                    "output": target_var,
                    "op": logic.op,
                    "weight": logic.weight * 0.5,  # Weaker influence
                    "learnable": logic.learnable,
                })
        
        return {
            "version": "1.0",
            "name": f"context_{context.center_node_id[:8]}",
            "variables": variables,
            "rules": rules,
            "constraints": constraints,
        }

    def _node_to_var_name(self, node: Node) -> str:
        """Converts a node to a variable name for NeuroJSON."""
        # Use node ID as variable name (ensures uniqueness)
        return f"node_{node.id}"

    def _var_name_to_node_id(self, var_name: str) -> str:
        """Extracts node ID from variable name."""
        if var_name.startswith("node_"):
            return var_name[5:]
        return var_name

    # =========================================================================
    # Inference
    # =========================================================================

    def run_inference(
        self,
        context: ContextGraph,
        evidence: Optional[Dict[str, TruthValue]] = None,
        iterations: Optional[int] = None,
    ) -> Dict[str, TruthValue]:
        """
        Runs inference on a context graph.
        
        Args:
            context: The context graph to reason over
            evidence: Optional node_id -> truth_value mappings
            iterations: Optional max iterations
        
        Returns:
            Dict of node_id -> updated truth_value
        """
        # Convert to NeuroJSON
        schema = self.to_neuro_json(context)
        
        # Create engine
        from .neuro.types import EngineConfig
        config = EngineConfig(
            max_iterations=iterations or self.config["max_iterations"],
            convergence_threshold=self.config["convergence_threshold"],
            learning_rate=self.config["learning_rate"],
            damping_factor=self.config["damping_factor"],
        )
        engine = NeuroEngine(schema, config)
        
        # Convert evidence node IDs to variable names
        var_evidence = {}
        if evidence:
            for node_id, value in evidence.items():
                var_name = f"node_{node_id}"
                if var_name in schema["variables"]:
                    var_evidence[var_name] = value
        
        # Run inference
        result = engine.run(var_evidence, iterations)
        
        # Convert back to node IDs
        node_results = {}
        for var_name, value in result.items():
            node_id = self._var_name_to_node_id(var_name)
            node_results[node_id] = value
        
        return node_results

    async def solve_context(
        self,
        center_node_id: str,
        depth: int = 2,
        evidence: Optional[Dict[str, TruthValue]] = None,
        write_back: bool = True,
    ) -> Dict[str, TruthValue]:
        """
        The main "Solve" routine from the integration design.
        
        1. Fetches subgraph neighborhood
        2. Converts to NeuroJSON
        3. Runs inference (diffusion)
        4. Optionally writes back to DB
        
        Args:
            center_node_id: ID of the center node
            depth: Number of hops to include (context window)
            evidence: Optional evidence to inject
            write_back: Whether to update nodes in DB
        
        Returns:
            Dict of node_id -> updated truth_value
        """
        # 1. Fetch Subgraph
        context = await self.fetch_context(center_node_id, depth)
        
        # 2-3. Convert to NeuroJSON and run inference
        results = self.run_inference(context, evidence)
        
        # 4. Write Back to DB
        if write_back and self.db is not None:
            updates = [
                {"id": node_id, "truth_value": value}
                for node_id, value in results.items()
            ]
            await self.db.bulk_update_nodes(updates)
        
        return results

    def query_node(
        self,
        context: ContextGraph,
        node_id: str,
        evidence: Optional[Dict[str, TruthValue]] = None,
    ) -> TruthValue:
        """
        Queries the truth value of a specific node given evidence.
        
        Args:
            context: The context graph
            node_id: The node to query
            evidence: Optional evidence
        
        Returns:
            Truth value of the queried node
        """
        results = self.run_inference(context, evidence)
        return results.get(node_id, 0.5)

    # =========================================================================
    # Grounding (First-Order Logic to Instances)
    # =========================================================================

    def ground_abstract_rules(
        self,
        context: ContextGraph,
        abstract_rules: List[Dict[str, Any]],
    ) -> ContextGraph:
        """
        Grounds abstract rules to specific instances.
        
        Abstract Rule Example:
            forall X: Smoker(X) -> Cancer(X)
        
        If we have an instance "Alice" tagged as "Smoker", we create:
            Alice(Smoker) -> Alice(Cancer)
        
        Args:
            context: The context graph with instances
            abstract_rules: List of abstract rule definitions
        
        Returns:
            Updated context with grounded associations
        """
        # This is a simplified implementation
        # Full FOL grounding would require type/tag matching
        
        grounded_associations = dict(context.associations)
        
        for rule in abstract_rules:
            # Extract rule pattern
            source_type = rule.get("source_type")  # e.g., "Smoker"
            target_type = rule.get("target_type")  # e.g., "Cancer"
            logic_type = rule.get("logic_type", LogicType.IMPLIES)
            weight = rule.get("weight", 1.0)
            
            if not source_type or not target_type:
                continue
            
            # Find instances matching source type
            for node_id, node in context.nodes.items():
                node_tags = node.payload.get("tags", [])
                if source_type in node_tags or source_type in node.prototype_ids:
                    # Create or find corresponding target node
                    # In practice, you'd look up or create the target node
                    # Here we just demonstrate the concept
                    target_var_id = f"{node_id}_{target_type}"
                    
                    # Create grounded association
                    assoc = Association.create(
                        source_id=node_id,
                        target_id=target_var_id,
                        relation=f"grounded_{source_type}_{target_type}",
                        logic_type=logic_type,
                        logic_weight=weight,
                    )
                    grounded_associations[assoc.id] = assoc
        
        return ContextGraph(
            nodes=context.nodes,
            associations=grounded_associations,
            center_node_id=context.center_node_id,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_neuro_service(db=None) -> NeuroService:
    """Creates a new NeuroService instance."""
    return NeuroService(db)


def run_local_inference(
    nodes: List[Node],
    associations: List[Association],
    evidence: Optional[Dict[str, TruthValue]] = None,
) -> Dict[str, TruthValue]:
    """
    Runs inference locally without a database.
    
    Useful for testing or when you have data in memory.
    """
    service = NeuroService()
    
    if not nodes:
        return {}
    
    context = service.extract_context(
        nodes=nodes,
        associations=associations,
        center_node_id=nodes[0].id,
    )
    
    return service.run_inference(context, evidence)
