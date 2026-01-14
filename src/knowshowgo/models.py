from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def gen_id() -> str:
    return uuid4().hex


@dataclass(frozen=True)
class Prototype:
    """Immutable prototype; edits create a new version via previous_version_id."""

    id: str
    name: str
    schema_meta: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=now_utc)
    created_by: Optional[str] = None
    previous_version_id: Optional[str] = None

    @staticmethod
    def base(
        name: str,
        schema_meta: Dict[str, Any],
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        previous_version_id: Optional[str] = None,
    ) -> "Prototype":
        return Prototype(
            id=gen_id(),
            name=name,
            schema_meta=schema_meta,
            tags=tags or [],
            created_at=now_utc(),
            created_by=created_by,
            previous_version_id=previous_version_id,
        )


@dataclass
class Node:
    """Instance node; prototype_id is optional (can be UNDEFINED).
    
    Neurosymbolic fields (for fuzzy reasoning):
    - truth_value: Current fuzzy state [0.0 - 1.0]
    - prior: Base probability before evidence (default 0.5)
    - is_locked: If TRUE, this value is "Evidence" and cannot be overwritten by logic
    """

    id: str
    prototype_id: Optional[str]
    prototype_ids: List[str] = field(default_factory=list)  # optional multi-prototype set
    payload: Dict[str, Any] = field(default_factory=dict)
    associations: List[str] = field(default_factory=list)
    embedding_ref: Optional[str] = None  # points to embedding record (versioned in store)
    weight: float = 0.0
    created_at: datetime = field(default_factory=now_utc)
    created_by: Optional[str] = None  # e.g., DID/user token
    context_ids: List[str] = field(default_factory=list)  # CONTEXT nodes that conditionally activate this node
    # Neurosymbolic fields
    truth_value: float = 0.5  # Current fuzzy state [0.0 - 1.0]
    prior: float = 0.5  # Base probability before evidence
    is_locked: bool = False  # If TRUE, value is evidence and won't be changed by inference

    @staticmethod
    def create(
        prototype_id: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
        context_ids: Optional[List[str]] = None,
        prototype_ids: Optional[List[str]] = None,
        prior: float = 0.5,
        truth_value: Optional[float] = None,
        is_locked: bool = False,
    ) -> "Node":
        proto_list = prototype_ids[:] if prototype_ids else []
        if prototype_id and prototype_id not in proto_list:
            proto_list.append(prototype_id)
        return Node(
            id=gen_id(),
            prototype_id=prototype_id,
            prototype_ids=proto_list,
            payload=payload or {},
            associations=[],
            embedding_ref=None,
            weight=0.0,
            created_at=now_utc(),
            created_by=created_by,
            context_ids=context_ids or [],
            truth_value=truth_value if truth_value is not None else prior,
            prior=prior,
            is_locked=is_locked,
        )


# Logic edge types for neurosymbolic reasoning
class LogicType:
    IMPLIES = "IMPLIES"    # A -> B: if A then B
    ATTACKS = "ATTACKS"    # A defeats B (argumentation)
    SUPPORTS = "SUPPORTS"  # A reinforces B
    DEPENDS = "DEPENDS"    # A depends on B
    MUTEX = "MUTEX"        # Mutual exclusion


class LogicOp:
    AND = "AND"          # Conjunction: all inputs required
    OR = "OR"            # Disjunction: any input contributes
    IDENTITY = "IDENTITY"  # Pass-through


@dataclass
class LogicMeta:
    """Logic metadata for neurosymbolic edges (Code as Data).
    
    Defines how this edge participates in fuzzy inference.
    """
    type: str = LogicType.IMPLIES  # IMPLIES, ATTACKS, SUPPORTS, DEPENDS, MUTEX
    weight: float = 1.0  # Confidence/strength of this rule [0.0 - 1.0]
    op: str = LogicOp.IDENTITY  # AND, OR, IDENTITY (for multi-input rules)
    learnable: bool = True  # Whether weight can be updated by training

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "weight": self.weight,
            "op": self.op,
            "learnable": self.learnable,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LogicMeta":
        return LogicMeta(
            type=data.get("type", LogicType.IMPLIES),
            weight=data.get("weight", 1.0),
            op=data.get("op", LogicOp.IDENTITY),
            learnable=data.get("learnable", True),
        )


@dataclass
class Association:
    """Edge as first-class node with mutable weight and optional ordering metadata.
    
    Neurosymbolic extension:
    - logic_meta: Defines how this edge participates in fuzzy reasoning
    """

    id: str
    source_id: str
    target_id: str
    relation: str
    weight: float = 0.0
    created_at: datetime = field(default_factory=now_utc)
    created_by: Optional[str] = None
    previous_version_id: Optional[str] = None
    position: Optional[int] = None  # optional ordering metadata
    metadata: Dict[str, Any] = field(default_factory=dict)  # e.g., member_prototype for SET membership
    # Neurosymbolic logic metadata
    logic_meta: Optional[LogicMeta] = None

    @staticmethod
    def create(
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 0.0,
        created_by: Optional[str] = None,
        position: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        logic_type: Optional[str] = None,
        logic_weight: float = 1.0,
        logic_op: str = LogicOp.IDENTITY,
    ) -> "Association":
        logic_meta = None
        if logic_type:
            logic_meta = LogicMeta(type=logic_type, weight=logic_weight, op=logic_op)
        return Association(
            id=gen_id(),
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            created_at=now_utc(),
            created_by=created_by,
            position=position,
            metadata=metadata or {},
            logic_meta=logic_meta,
        )

    @staticmethod
    def create_implies(
        source_id: str,
        target_id: str,
        relation: str = "implies",
        weight: float = 1.0,
        created_by: Optional[str] = None,
    ) -> "Association":
        """Creates an implication edge: source -> target (if source then target)."""
        return Association.create(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            logic_type=LogicType.IMPLIES,
            logic_weight=weight,
            created_by=created_by,
        )

    @staticmethod
    def create_attacks(
        source_id: str,
        target_id: str,
        relation: str = "attacks",
        weight: float = 1.0,
        created_by: Optional[str] = None,
    ) -> "Association":
        """Creates an attack edge: source defeats target (argumentation)."""
        return Association.create(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            logic_type=LogicType.ATTACKS,
            logic_weight=weight,
            created_by=created_by,
        )


# Core prototype names
ROOT_PROTOTYPE = "NODE"
ASSOCIATION_PROTOTYPE = "ASSOCIATION"
OBJECT_PROTOTYPE = "OBJECT"
OBJECT_PROPERTY_PROTOTYPE = "OBJECT_PROPERTY"
TAG_PROTOTYPE = "TAG"
USER_PROTOTYPE = "USER"
CONTEXT_PROTOTYPE = "CONTEXT"
UNDEFINED_PROTOTYPE = "UNDEFINED"
SET_PROTOTYPE = "SET"
QUANTITY_PROTOTYPE = "QUANTITY"


def seed_core_prototypes(created_by: Optional[str] = None) -> Dict[str, Prototype]:
    """Generate immutable core prototypes for bootstrapping."""

    def proto(name: str, description: str, extra_tags: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> Prototype:
        schema = {"description": description}
        if meta:
            schema.update(meta)
        tags = extra_tags or []
        return Prototype.base(name=name, schema_meta=schema, tags=tags, created_by=created_by)

    return {
        ROOT_PROTOTYPE: proto(ROOT_PROTOTYPE, "Base node prototype"),
        UNDEFINED_PROTOTYPE: proto(UNDEFINED_PROTOTYPE, "Prototype placeholder when none specified"),
        ASSOCIATION_PROTOTYPE: proto(ASSOCIATION_PROTOTYPE, "Edge between nodes with relation and weight"),
        OBJECT_PROTOTYPE: proto(OBJECT_PROTOTYPE, "Semantic object; holds tags and properties"),
        OBJECT_PROPERTY_PROTOTYPE: proto(OBJECT_PROPERTY_PROTOTYPE, "Property node contained within an OBJECT"),
        TAG_PROTOTYPE: proto(
            TAG_PROTOTYPE,
            "Lexical label/tag for objects",
            extra_tags=["PROPERTY"],
            meta={"parent_prototype": OBJECT_PROPERTY_PROTOTYPE},
        ),
        USER_PROTOTYPE: proto(USER_PROTOTYPE, "User node for attribution/voting/DID profile pointer"),
        CONTEXT_PROTOTYPE: proto(CONTEXT_PROTOTYPE, "Context selector node"),
        SET_PROTOTYPE: proto(SET_PROTOTYPE, "Unordered collection/set of nodes"),
        QUANTITY_PROTOTYPE: proto(
            QUANTITY_PROTOTYPE,
            "Quantity value framed with a category prototype",
            extra_tags=["PROPERTY"],
            meta={"parent_prototype": OBJECT_PROPERTY_PROTOTYPE},
        ),
    }
