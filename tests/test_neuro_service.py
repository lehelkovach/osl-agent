"""Tests for NeuroService - KSG + NeuroSym integration."""

import pytest
from knowshowgo.models import Node, Association, LogicType, LogicMeta
from knowshowgo.neuro_service import NeuroService, run_local_inference
from knowshowgo.neuro import NeuroEngine, fuzzy_and, fuzzy_or, fuzzy_not, implies


class TestLogicCore:
    """Tests for the Python port of NeuroSym logic core."""

    def test_fuzzy_not(self):
        assert fuzzy_not(0) == 1.0
        assert fuzzy_not(1) == 0.0
        assert abs(fuzzy_not(0.3) - 0.7) < 0.001

    def test_fuzzy_and(self):
        assert fuzzy_and() == 1.0  # Empty AND
        assert fuzzy_and(0.7) == 0.7  # Single value
        assert fuzzy_and(1, 1) == 1.0
        assert fuzzy_and(0, 1) == 0.0
        # Lukasiewicz: max(0, 0.8 + 0.9 - 1) = 0.7
        assert abs(fuzzy_and(0.8, 0.9) - 0.7) < 0.001

    def test_fuzzy_or(self):
        assert fuzzy_or() == 0.0  # Empty OR
        assert fuzzy_or(0.7) == 0.7  # Single value
        assert fuzzy_or(0, 0) == 0.0
        assert fuzzy_or(1, 1) == 1.0
        # Lukasiewicz: min(1, 0.3 + 0.4) = 0.7
        assert abs(fuzzy_or(0.3, 0.4) - 0.7) < 0.001

    def test_implies(self):
        assert implies(0, 0) == 1.0  # False implies anything
        assert implies(0, 1) == 1.0
        assert implies(1, 1) == 1.0
        assert implies(1, 0) == 0.0  # True doesn't imply false
        # min(1, 1 - 0.8 + 0.5) = 0.7
        assert abs(implies(0.8, 0.5) - 0.7) < 0.001


class TestNeuroEngine:
    """Tests for the Python NeuroEngine."""

    def test_basic_inference(self):
        schema = {
            "version": "1.0",
            "variables": {
                "A": {"type": "bool", "prior": 0.3},
                "B": {"type": "bool", "prior": 0.1},
            },
            "rules": [{
                "id": "a_to_b",
                "type": "IMPLICATION",
                "inputs": ["A"],
                "output": "B",
                "op": "IDENTITY",
                "weight": 0.9,
            }],
            "constraints": [],
        }
        
        engine = NeuroEngine(schema)
        result = engine.run({"A": 1.0})
        
        assert result["A"] == 1.0  # Evidence locked
        assert result["B"] > 0.5  # Should increase due to implication

    def test_attack_constraint(self):
        schema = {
            "version": "1.0",
            "variables": {
                "attacker": {"type": "bool", "prior": 0.8},
                "target": {"type": "bool", "prior": 0.7},
            },
            "rules": [],
            "constraints": [{
                "id": "attack_1",
                "type": "ATTACK",
                "source": "attacker",
                "target": "target",
                "weight": 1.0,
            }],
        }
        
        engine = NeuroEngine(schema)
        result = engine.run()
        
        # Target should be reduced by attack
        assert result["target"] < 0.7

    def test_causal_chain(self):
        schema = {
            "version": "1.0",
            "variables": {
                "rain": {"type": "bool", "prior": 0.2},
                "wet": {"type": "bool", "prior": 0.1},
                "slippery": {"type": "bool", "prior": 0.05},
            },
            "rules": [
                {
                    "id": "rain_wet",
                    "type": "IMPLICATION",
                    "inputs": ["rain"],
                    "output": "wet",
                    "op": "IDENTITY",
                    "weight": 0.95,
                },
                {
                    "id": "wet_slip",
                    "type": "IMPLICATION",
                    "inputs": ["wet"],
                    "output": "slippery",
                    "op": "IDENTITY",
                    "weight": 0.8,
                },
            ],
            "constraints": [],
        }
        
        engine = NeuroEngine(schema)
        
        # Given rain, wet and slippery should increase
        result = engine.run({"rain": 1.0})
        
        assert result["rain"] == 1.0
        assert result["wet"] > 0.5
        assert result["slippery"] > 0.2

    def test_training(self):
        from knowshowgo.neuro.engine import TrainingData
        
        schema = {
            "version": "1.0",
            "variables": {
                "input": {"type": "bool", "prior": 0.5},
                "output": {"type": "bool", "prior": 0.5},
            },
            "rules": [{
                "id": "learnable",
                "type": "IMPLICATION",
                "inputs": ["input"],
                "output": "output",
                "op": "IDENTITY",
                "weight": 0.5,
                "learnable": True,
            }],
            "constraints": [],
        }
        
        engine = NeuroEngine(schema)
        initial_weight = engine.get_rule_weight("learnable")
        
        # Train toward high output
        data = [
            TrainingData(inputs={"input": 1.0}, targets={"output": 0.95}),
        ]
        engine.train(data, epochs=50)
        
        final_weight = engine.get_rule_weight("learnable")
        
        # Weight should have increased
        assert final_weight > initial_weight

    def test_export(self):
        schema = {
            "version": "1.0",
            "variables": {"A": {"type": "bool", "prior": 0.5}},
            "rules": [],
            "constraints": [],
        }
        
        engine = NeuroEngine(schema)
        exported = engine.export()
        
        assert exported["version"] == "1.0"
        assert "A" in exported["variables"]


class TestNeuroService:
    """Tests for NeuroService - KSG integration."""

    def create_test_nodes(self):
        """Creates test nodes for inference."""
        penguin = Node.create(
            prototype_id="animal",
            payload={"name": "Penguin", "tags": ["bird"]},
            prior=0.5,
        )
        bird = Node.create(
            prototype_id="concept",
            payload={"name": "Bird"},
            prior=0.3,
        )
        fly = Node.create(
            prototype_id="ability",
            payload={"name": "Fly"},
            prior=0.5,
        )
        return penguin, bird, fly

    def test_to_neuro_json(self):
        """Tests conversion of KSG graph to NeuroJSON."""
        penguin, bird, fly = self.create_test_nodes()
        
        # Create implication: penguin -> bird
        penguin_is_bird = Association.create_implies(
            source_id=penguin.id,
            target_id=bird.id,
            weight=1.0,
        )
        
        # Create attack: penguin attacks fly (penguins don't fly)
        penguin_no_fly = Association.create_attacks(
            source_id=penguin.id,
            target_id=fly.id,
            weight=1.0,
        )
        
        service = NeuroService()
        context = service.extract_context(
            nodes=[penguin, bird, fly],
            associations=[penguin_is_bird, penguin_no_fly],
            center_node_id=penguin.id,
        )
        
        schema = service.to_neuro_json(context)
        
        assert schema["version"] == "1.0"
        assert len(schema["variables"]) == 3
        assert len(schema["rules"]) == 1  # One IMPLIES
        assert len(schema["constraints"]) == 1  # One ATTACK

    def test_run_inference(self):
        """Tests running inference on KSG nodes."""
        penguin, bird, fly = self.create_test_nodes()
        
        # Penguin implies bird
        penguin_is_bird = Association.create_implies(
            source_id=penguin.id,
            target_id=bird.id,
            weight=0.95,
        )
        
        # Penguin attacks fly
        penguin_no_fly = Association.create_attacks(
            source_id=penguin.id,
            target_id=fly.id,
            weight=0.9,
        )
        
        service = NeuroService()
        context = service.extract_context(
            nodes=[penguin, bird, fly],
            associations=[penguin_is_bird, penguin_no_fly],
            center_node_id=penguin.id,
        )
        
        # Run inference with penguin as evidence (true)
        results = service.run_inference(context, evidence={penguin.id: 1.0})
        
        # Bird should be high (penguin implies bird)
        assert results[bird.id] > 0.5
        
        # Fly should be low (penguin attacks fly)
        assert results[fly.id] < 0.5

    def test_local_inference_convenience(self):
        """Tests the run_local_inference convenience function."""
        node_a = Node.create(prototype_id="concept", prior=0.5)
        node_b = Node.create(prototype_id="concept", prior=0.1)
        
        assoc = Association.create_implies(
            source_id=node_a.id,
            target_id=node_b.id,
            weight=0.9,
        )
        
        results = run_local_inference(
            nodes=[node_a, node_b],
            associations=[assoc],
            evidence={node_a.id: 1.0},
        )
        
        assert results[node_a.id] == 1.0
        assert results[node_b.id] > 0.3


class TestModelUpdates:
    """Tests for the updated KSG models."""

    def test_node_has_neurosymbolic_fields(self):
        """Tests that Node has truth_value, prior, is_locked."""
        node = Node.create(
            prototype_id="test",
            prior=0.7,
            truth_value=0.8,
            is_locked=True,
        )
        
        assert node.prior == 0.7
        assert node.truth_value == 0.8
        assert node.is_locked is True

    def test_node_default_values(self):
        """Tests Node default neurosymbolic values."""
        node = Node.create(prototype_id="test")
        
        assert node.prior == 0.5
        assert node.truth_value == 0.5  # Defaults to prior
        assert node.is_locked is False

    def test_association_has_logic_meta(self):
        """Tests that Association can have logic_meta."""
        assoc = Association.create(
            source_id="a",
            target_id="b",
            relation="implies",
            logic_type=LogicType.IMPLIES,
            logic_weight=0.9,
        )
        
        assert assoc.logic_meta is not None
        assert assoc.logic_meta.type == LogicType.IMPLIES
        assert assoc.logic_meta.weight == 0.9

    def test_create_implies_helper(self):
        """Tests the create_implies helper."""
        assoc = Association.create_implies("a", "b", weight=0.8)
        
        assert assoc.logic_meta.type == LogicType.IMPLIES
        assert assoc.logic_meta.weight == 0.8

    def test_create_attacks_helper(self):
        """Tests the create_attacks helper."""
        assoc = Association.create_attacks("a", "b", weight=0.7)
        
        assert assoc.logic_meta.type == LogicType.ATTACKS
        assert assoc.logic_meta.weight == 0.7

    def test_logic_meta_to_dict(self):
        """Tests LogicMeta serialization."""
        meta = LogicMeta(
            type=LogicType.IMPLIES,
            weight=0.9,
            op="AND",
            learnable=False,
        )
        
        d = meta.to_dict()
        
        assert d["type"] == LogicType.IMPLIES
        assert d["weight"] == 0.9
        assert d["op"] == "AND"
        assert d["learnable"] is False

    def test_logic_meta_from_dict(self):
        """Tests LogicMeta deserialization."""
        d = {
            "type": "ATTACKS",
            "weight": 0.8,
            "op": "IDENTITY",
            "learnable": True,
        }
        
        meta = LogicMeta.from_dict(d)
        
        assert meta.type == "ATTACKS"
        assert meta.weight == 0.8
        assert meta.op == "IDENTITY"
        assert meta.learnable is True
