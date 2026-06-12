"""
matkit.energy - 能量计算模块
Energy calculation module for MatKit

提供表面能、吸附能和掺杂能（缺陷形成能）的计算功能。
Provides surface energy, adsorption energy, and doping energy
(defect formation energy) calculations.
"""

from .surface_energy import (
    build_simple_substance_database,
    calc_surface_energy,
    calc_surface_energies_from_files,
    calc_surface_excess_energy,
    calc_surface_excess_energies_batch,
    calc_surface_excess_energy_from_directory,
    find_energy_file,
    find_structure_file,
    iter_calculation_directories,
    read_calculation_energy,
)

from .adsorption_energy import (
    calc_adsorption_energy,
    calc_adsorption_energies_batch,
)

from .doping_energy import (
    calc_doping_energy,
    calc_doping_energies_table,
)

__all__ = [
    # surface energy
    "build_simple_substance_database",
    "calc_surface_energy",
    "calc_surface_energies_from_files",
    "calc_surface_excess_energy",
    "calc_surface_excess_energies_batch",
    "calc_surface_excess_energy_from_directory",
    "find_energy_file",
    "find_structure_file",
    "iter_calculation_directories",
    "read_calculation_energy",
    # adsorption energy
    "calc_adsorption_energy",
    "calc_adsorption_energies_batch",
    # doping energy
    "calc_doping_energy",
    "calc_doping_energies_table",
]
