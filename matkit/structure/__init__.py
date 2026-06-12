"""
matkit.structure - 结构分析模块
Structure analysis module for MatKit

提供晶体和分子结构的几何计算、分子识别和表面分析功能。
Provides geometry calculation, molecule identification, and surface analysis
for crystal and molecular structures.
"""

from .geometry import (
    calc_distance,
    calc_angle,
    calc_dihedral,
    calc_bond_lengths,
    calc_plane_normal,
    calc_angle_between_planes,
    calc_center_of_mass,
    calc_rmsd,
)

from .molecule import (
    identify_molecules,
    match_molecule,
    find_adsorption_sites,
)

from .surface import (
    identify_surface_layers,
    calc_surface_area,
    analyze_surface_relaxation,
)

__all__ = [
    # geometry
    "calc_distance",
    "calc_angle",
    "calc_dihedral",
    "calc_bond_lengths",
    "calc_plane_normal",
    "calc_angle_between_planes",
    "calc_center_of_mass",
    "calc_rmsd",
    # molecule
    "identify_molecules",
    "match_molecule",
    "find_adsorption_sites",
    # surface
    "identify_surface_layers",
    "calc_surface_area",
    "analyze_surface_relaxation",
]
