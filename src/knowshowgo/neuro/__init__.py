"""
NeuroSym Python Module

Python port of NeuroSym.js for integration with KnowShowGo.
Provides neurosymbolic reasoning capabilities using Lukasiewicz fuzzy logic.
"""

from .logic import (
    clamp,
    fuzzy_and,
    fuzzy_or,
    fuzzy_not,
    implies,
    equivalent,
    inhibit,
    support,
)
from .engine import NeuroEngine
from .types import NeuroJSON, Variable, Rule, Constraint

__all__ = [
    # Logic functions
    "clamp",
    "fuzzy_and",
    "fuzzy_or",
    "fuzzy_not",
    "implies",
    "equivalent",
    "inhibit",
    "support",
    # Engine
    "NeuroEngine",
    # Types
    "NeuroJSON",
    "Variable",
    "Rule",
    "Constraint",
]
