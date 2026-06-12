"""
High-level analysis workflows for MatKit.

These functions combine parsers, geometry utilities, and domain-specific
post-processing into workflows that can be used from Python or the CLI.
"""

from .adsorption import (
    DEFAULT_INTERNAL_BOND_CUTOFFS,
    analyze_adsorbate_geometry,
    analyze_so4_cu2o,
    flatten_so4_metrics,
)
from .recommendations import suggest_next_analyses

__all__ = [
    "DEFAULT_INTERNAL_BOND_CUTOFFS",
    "analyze_adsorbate_geometry",
    "analyze_so4_cu2o",
    "flatten_so4_metrics",
    "suggest_next_analyses",
]
