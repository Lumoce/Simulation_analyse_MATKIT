"""
matkit.energy.surface_energy - 表面能计算
Surface energy calculation for MatKit

提供表面能的计算功能，支持直接数值输入和从 VASP OUTCAR 文件读取。
Provides surface energy calculation, supporting direct numerical input
and reading from VASP OUTCAR files.
"""

import os
import re

import numpy as np


_ENERGY_NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
_RE_TOTEN = re.compile(r"free\s+energy\s+TOTEN\s*=\s*" + _ENERGY_NUMBER)
_RE_WITHOUT_ENTROPY = re.compile(r"energy\s+without\s+entropy\s*=\s*" + _ENERGY_NUMBER)
_RE_OSZICAR_F = re.compile(r"^\s*\d+\s+F=\s*" + _ENERGY_NUMBER)

ENERGY_FILE_CANDIDATES = (
    "OUTCAR",
    "outcar",
    "OSZICAR",
    "oszicar",
    "log",
    "LOG",
    "vasp.log",
    "VASP.log",
    "stdout",
    "stdout.log",
    "slurm.out",
)

STRUCTURE_FILE_CANDIDATES = (
    "POSCAR",
    "poscar",
    "CONTCAR",
    "contcar",
)


def _normalise_energy_number(value):
    """Convert VASP-style D exponents to a float."""
    return float(value.replace("D", "E").replace("d", "e"))


def find_structure_file(directory):
    """
    Find a POSCAR/CONTCAR-like structure file inside a calculation directory.
    """
    directory = str(directory)
    if os.path.isfile(directory):
        return directory

    for name in STRUCTURE_FILE_CANDIDATES:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate

    return None


def find_energy_file(path):
    """
    Find an OUTCAR/OSZICAR/log-like energy file.

    If ``path`` already points to a file, it is returned unchanged.  If it is a
    directory, common VASP output names are checked first, then log/out-like
    files are considered deterministically in sorted order.
    """
    path = str(path)
    if os.path.isfile(path):
        return path

    if not os.path.isdir(path):
        return None

    for name in ENERGY_FILE_CANDIDATES:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate

    for name in sorted(os.listdir(path)):
        lower = name.lower()
        candidate = os.path.join(path, name)
        if not os.path.isfile(candidate):
            continue
        if "outcar" in lower or lower.endswith(".log") or lower.endswith(".out"):
            return candidate

    return None


def read_calculation_energy(filepath):
    """
    Read the final total energy from OUTCAR, OSZICAR, or a VASP log file.

    The parser prefers the last ``free energy TOTEN`` value.  If a log does not
    contain TOTEN, it falls back to ``energy without entropy`` or OSZICAR-style
    ``F=`` lines.
    """
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"能量文件不存在: {filepath}")

    toten_energy = None
    fallback_energy = None
    is_converged = False

    with open(filepath, "r") as f:
        for line in f:
            if "reached required accuracy" in line:
                is_converged = True

            match = _RE_TOTEN.search(line)
            if match:
                toten_energy = _normalise_energy_number(match.group(1))
                continue

            match = _RE_WITHOUT_ENTROPY.search(line)
            if match:
                fallback_energy = _normalise_energy_number(match.group(1))
                continue

            match = _RE_OSZICAR_F.search(line)
            if match:
                fallback_energy = _normalise_energy_number(match.group(1))

    final_energy = toten_energy if toten_energy is not None else fallback_energy
    if final_energy is None:
        raise ValueError(f"无法从能量文件中解析最终能量: {filepath}")

    return {
        "final_energy": final_energy,
        "is_converged": is_converged,
        "energy_file": filepath,
    }


def _read_outcar_energy(filepath):
    """
    从 VASP OUTCAR 文件中读取总能量。
    Read total energy from a VASP OUTCAR file.

    Parameters
    ----------
    filepath : str
        OUTCAR 文件路径。
        Path to the OUTCAR file.

    Returns
    -------
    float
        总能量（eV）。
        Total energy in eV.

    Raises
    ------
    FileNotFoundError
        如果文件不存在。
        If the file does not exist.
    ValueError
        如果无法从文件中解析能量。
        If energy cannot be parsed from the file.
    """
    return read_calculation_energy(filepath)["final_energy"]


def calc_surface_energy(E_slab, E_bulk_per_atom, n_atoms_slab, surface_area, n_surfaces=2):
    """
    计算表面能。
    Calculate surface energy.

    公式：gamma = (E_slab - N * E_bulk) / (2 * A)

    其中 E_slab 是板模型总能量，E_bulk 是每个体相原子的能量，
    N 是板模型中的原子数，A 是表面积，2 表示上下两个表面。
    Formula: gamma = (E_slab - N * E_bulk) / (2 * A)

    Where E_slab is the slab total energy, E_bulk is the bulk energy per atom,
    N is the number of atoms in the slab, A is the surface area, and 2 accounts
    for both top and bottom surfaces.

    Parameters
    ----------
    E_slab : float
        板模型总能量（eV）。
        Total energy of the slab model in eV.
    E_bulk_per_atom : float
        体相每个原子的能量（eV）。
        Bulk energy per atom in eV.
    n_atoms_slab : int
        板模型中的原子总数。
        Total number of atoms in the slab model.
    surface_area : float
        单侧表面积（Å²）。
        Surface area of one side in Angstroms squared.
    n_surfaces : int, optional
        表面数量，默认为 2（上下两个表面）。对于非对称板模型可设为 1。
        Number of surfaces, default 2 (top and bottom). Set to 1 for asymmetric slabs.

    Returns
    -------
    dict
        表面能计算结果：
        Surface energy calculation result:
        - 'surface_energy_eV_A2': float - 表面能（eV/Å²）/ Surface energy in eV/A^2
        - 'surface_energy_J_m2': float - 表面能（J/m²）/ Surface energy in J/m^2
        - 'E_slab': float - 板模型总能量 / Slab total energy
        - 'E_bulk_total': float - 体相参考总能量 / Bulk reference total energy
        - 'excess_energy': float - 多余能量 / Excess energy
        - 'n_atoms_slab': int - 板模型原子数 / Number of atoms in slab
        - 'surface_area_A2': float - 表面积 / Surface area
        - 'n_surfaces': int - 表面数 / Number of surfaces

    Raises
    ------
    ValueError
        如果输入验证失败。
        If input validation fails.

    Examples
    --------
    >>> result = calc_surface_energy(-20.0, -3.5, 6, 12.0)
    >>> result['surface_energy_eV_A2']
    0.083333...
    """
    # 输入验证 / Input validation
    if not isinstance(E_slab, (int, float)):
        raise ValueError(
            f"E_slab 必须是数值类型，但得到 {type(E_slab).__name__}。"
            f"E_slab must be a numeric type, but got {type(E_slab).__name__}."
        )

    if not isinstance(E_bulk_per_atom, (int, float)):
        raise ValueError(
            f"E_bulk_per_atom 必须是数值类型，但得到 {type(E_bulk_per_atom).__name__}。"
            f"E_bulk_per_atom must be a numeric type, but got {type(E_bulk_per_atom).__name__}."
        )

    if not isinstance(n_atoms_slab, (int, np.integer)) or n_atoms_slab <= 0:
        raise ValueError(
            f"n_atoms_slab 必须是正整数，但得到 {n_atoms_slab}。"
            f"n_atoms_slab must be a positive integer, but got {n_atoms_slab}."
        )

    if not isinstance(surface_area, (int, float)) or surface_area <= 0:
        raise ValueError(
            f"surface_area 必须是正数，但得到 {surface_area}。"
            f"surface_area must be a positive number, but got {surface_area}."
        )

    if not isinstance(n_surfaces, (int, np.integer)) or n_surfaces <= 0:
        raise ValueError(
            f"n_surfaces 必须是正整数，但得到 {n_surfaces}。"
            f"n_surfaces must be a positive integer, but got {n_surfaces}."
        )

    E_slab = float(E_slab)
    E_bulk_per_atom = float(E_bulk_per_atom)
    n_atoms_slab = int(n_atoms_slab)
    surface_area = float(surface_area)
    n_surfaces = int(n_surfaces)

    # 计算体相参考总能量 / Compute bulk reference total energy
    E_bulk_total = n_atoms_slab * E_bulk_per_atom

    # 多余能量 / Excess energy
    excess_energy = E_slab - E_bulk_total

    # 表面能 / Surface energy
    surface_energy_eV_A2 = excess_energy / (n_surfaces * surface_area)

    # 单位转换：1 eV/Å² = 16.0217656 J/m²
    # Unit conversion: 1 eV/A^2 = 16.0217656 J/m^2
    EV_A2_TO_J_M2 = 16.0217656
    surface_energy_J_m2 = surface_energy_eV_A2 * EV_A2_TO_J_M2

    return {
        'surface_energy_eV_A2': surface_energy_eV_A2,
        'surface_energy_J_m2': surface_energy_J_m2,
        'E_slab': E_slab,
        'E_bulk_total': E_bulk_total,
        'excess_energy': excess_energy,
        'n_atoms_slab': n_atoms_slab,
        'surface_area_A2': surface_area,
        'n_surfaces': n_surfaces,
    }


def calc_surface_excess_energy(E_slab, composition, chemical_potentials, n_surfaces=2):
    """
    Calculate per-surface excess energy from elemental chemical potentials.

    Formula:
        E_surface = (E_slab - sum_i n_i * mu_i) / n_surfaces

    This intentionally does not normalise by surface area.  It is useful for
    comparing slabs built from the same lateral cell, where the desired quantity
    is the excess energy per exposed surface rather than eV/A^2.
    """
    if not isinstance(E_slab, (int, float)):
        raise ValueError("E_slab 必须是数值类型。")
    if not isinstance(composition, dict) or len(composition) == 0:
        raise ValueError("composition 必须是非空字典。")
    if not isinstance(chemical_potentials, dict) or len(chemical_potentials) == 0:
        raise ValueError("chemical_potentials 必须是非空字典。")
    if not isinstance(n_surfaces, (int, np.integer)) or n_surfaces <= 0:
        raise ValueError(f"n_surfaces 必须是正整数，但得到 {n_surfaces}。")

    missing = sorted(elem for elem in composition if elem not in chemical_potentials)
    if missing:
        raise ValueError(
            "单质库缺少以下元素的化学势: "
            + ", ".join(missing)
        )

    reference_terms = []
    reference_energy = 0.0
    for elem in sorted(composition):
        count = int(composition[elem])
        if count < 0:
            raise ValueError(f"元素 {elem} 的原子数不能为负数。")
        mu = float(chemical_potentials[elem])
        contribution = count * mu
        reference_energy += contribution
        reference_terms.append({
            "element": elem,
            "count": count,
            "mu_eV_per_atom": mu,
            "reference_energy_eV": contribution,
        })

    E_slab = float(E_slab)
    excess_energy = E_slab - reference_energy
    surface_energy_per_surface = excess_energy / int(n_surfaces)

    return {
        "surface_excess_energy_eV_per_surface": surface_energy_per_surface,
        "excess_energy_eV": excess_energy,
        "E_slab": E_slab,
        "reference_energy_eV": reference_energy,
        "composition": {elem: int(composition[elem]) for elem in sorted(composition)},
        "chemical_potentials": {
            elem: float(chemical_potentials[elem]) for elem in sorted(composition)
        },
        "reference_terms": reference_terms,
        "n_surfaces": int(n_surfaces),
    }


def _infer_task_identity(path):
    """Infer task number and suffix from names like task.12_siteA."""
    name = os.path.basename(os.path.normpath(str(path)))
    match = re.search(r"task[._-]*(\d+)(.*)$", name, re.IGNORECASE)
    if match:
        number = match.group(1)
        suffix = match.group(2).lstrip("._-")
        return number, suffix

    match = re.search(r"(\d+)(.*)$", name)
    if match:
        return match.group(1), match.group(2).lstrip("._-")

    return None, ""


def build_simple_substance_database(database_dir):
    """
    Build elemental chemical potentials from a simple-substance calculation tree.

    Each reference calculation directory should contain a POSCAR/CONTCAR and an
    OUTCAR/OSZICAR/log-like file.  The reference must contain exactly one element;
    its chemical potential is ``final_energy / atom_count``.
    """
    from matkit.parsers import read_poscar

    database_dir = str(database_dir)
    if not os.path.isdir(database_dir):
        raise FileNotFoundError(f"单质库文件夹不存在: {database_dir}")

    references = {}
    for current_dir, dirnames, _filenames in os.walk(database_dir):
        dirnames.sort()
        structure_file = find_structure_file(current_dir)
        energy_file = find_energy_file(current_dir)
        if structure_file is None or energy_file is None:
            continue

        poscar_data = read_poscar(structure_file)
        composition = poscar_data["n_atoms"]
        if len(composition) != 1:
            continue

        element = next(iter(composition))
        atom_count = int(composition[element])
        if atom_count <= 0:
            continue

        energy_data = read_calculation_energy(energy_file)
        entry = {
            "element": element,
            "mu_eV_per_atom": energy_data["final_energy"] / atom_count,
            "total_energy_eV": energy_data["final_energy"],
            "atom_count": atom_count,
            "directory": current_dir,
            "structure_file": structure_file,
            "energy_file": energy_file,
            "is_converged": energy_data["is_converged"],
            "duplicate_references": [],
        }

        if element not in references:
            references[element] = entry
        else:
            references[element]["duplicate_references"].append(entry)

    if not references:
        raise ValueError(f"未在单质库中找到可用的 POSCAR + 能量文件: {database_dir}")

    return references


def _chemical_potentials_from_database(references):
    return {
        elem: data["mu_eV_per_atom"]
        for elem, data in references.items()
    }


def calc_surface_excess_energy_from_directory(
        slab_path,
        simple_substance_db="simple_substance_database",
        n_surfaces=2,
        substance_references=None):
    """
    Calculate surface excess energy from one slab calculation directory or file.
    """
    from matkit.parsers import read_poscar

    slab_path = str(slab_path)
    slab_dir = slab_path if os.path.isdir(slab_path) else os.path.dirname(slab_path)
    if not slab_dir:
        slab_dir = "."

    structure_file = find_structure_file(slab_dir)
    if structure_file is None:
        raise FileNotFoundError(f"未找到 POSCAR/CONTCAR: {slab_dir}")

    energy_file = find_energy_file(slab_path)
    if energy_file is None:
        energy_file = find_energy_file(slab_dir)
    if energy_file is None:
        raise FileNotFoundError(f"未找到 OUTCAR/OSZICAR/log 能量文件: {slab_dir}")

    if substance_references is None:
        substance_references = build_simple_substance_database(simple_substance_db)

    poscar_data = read_poscar(structure_file)
    energy_data = read_calculation_energy(energy_file)
    chemical_potentials = _chemical_potentials_from_database(substance_references)

    result = calc_surface_excess_energy(
        E_slab=energy_data["final_energy"],
        composition=poscar_data["n_atoms"],
        chemical_potentials=chemical_potentials,
        n_surfaces=n_surfaces,
    )

    task_number, task_suffix = _infer_task_identity(slab_dir)
    result.update({
        "label": os.path.basename(os.path.normpath(slab_dir)),
        "task_number": task_number,
        "task_suffix": task_suffix,
        "slab_dir": slab_dir,
        "structure_file": structure_file,
        "energy_file": energy_file,
        "is_converged": energy_data["is_converged"],
        "reference_database": simple_substance_db,
        "reference_files": {
            elem: {
                "mu_eV_per_atom": ref["mu_eV_per_atom"],
                "directory": ref["directory"],
                "energy_file": ref["energy_file"],
                "structure_file": ref["structure_file"],
                "atom_count": ref["atom_count"],
                "total_energy_eV": ref["total_energy_eV"],
            }
            for elem, ref in substance_references.items()
            if elem in result["composition"]
        },
    })

    return result


def iter_calculation_directories(root_dir):
    """
    Yield calculation directories containing both structure and energy files.
    """
    root_dir = str(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"计算结果根目录不存在: {root_dir}")

    for current_dir, dirnames, _filenames in os.walk(root_dir):
        dirnames.sort()
        if find_structure_file(current_dir) is not None and find_energy_file(current_dir) is not None:
            yield current_dir
            dirnames[:] = []


def calc_surface_excess_energies_batch(
        root_dir,
        simple_substance_db="simple_substance_database",
        n_surfaces=2,
        strict=False):
    """
    Batch-calculate per-surface excess energies under a root directory.
    """
    references = build_simple_substance_database(simple_substance_db)
    results = []

    for calc_dir in iter_calculation_directories(root_dir):
        try:
            result = calc_surface_excess_energy_from_directory(
                calc_dir,
                simple_substance_db=simple_substance_db,
                n_surfaces=n_surfaces,
                substance_references=references,
            )
            result["status"] = "ok"
        except Exception as exc:
            if strict:
                raise
            task_number, task_suffix = _infer_task_identity(calc_dir)
            result = {
                "status": "error",
                "label": os.path.basename(os.path.normpath(calc_dir)),
                "task_number": task_number,
                "task_suffix": task_suffix,
                "slab_dir": calc_dir,
                "error": str(exc),
            }
        results.append(result)

    return results


def calc_surface_energies_from_files(slab_outcar, bulk_outcar, n_bulk_atoms):
    """
    从 VASP OUTCAR 文件计算表面能的便捷函数。
    Convenience function to calculate surface energy from VASP OUTCAR files.

    注意：此函数仅读取能量值，表面积需要另外提供或通过其他方式计算。
    此函数返回能量部分，表面积需结合 matkit.structure.calc_surface_area 使用。
    Note: This function only reads energy values. Surface area needs to be
    provided separately or calculated via matkit.structure.calc_surface_area.
    This function returns the energy components; surface area calculation
    should be done separately.

    Parameters
    ----------
    slab_outcar : str
        板模型 OUTCAR 文件路径。
        Path to the slab model OUTCAR file.
    bulk_outcar : str
        体相 OUTCAR 文件路径。
        Path to the bulk OUTCAR file.
    n_bulk_atoms : int
        体相计算中的原子数。
        Number of atoms in the bulk calculation.

    Returns
    -------
    dict
        包含以下键的字典：
        Dict with the following keys:
        - 'E_slab': float - 板模型总能量（eV）/ Slab total energy in eV
        - 'E_bulk': float - 体相总能量（eV）/ Bulk total energy in eV
        - 'E_bulk_per_atom': float - 体相每个原子的能量（eV）/ Bulk energy per atom in eV
        - 'n_bulk_atoms': int - 体相原子数 / Number of bulk atoms
        - 'slab_outcar': str - 板模型 OUTCAR 路径 / Slab OUTCAR path
        - 'bulk_outcar': str - 体相 OUTCAR 路径 / Bulk OUTCAR path

    Raises
    ------
    ValueError
        如果输入验证失败。
        If input validation fails.
    FileNotFoundError
        如果 OUTCAR 文件不存在。
        If OUTCAR files do not exist.

    Examples
    --------
    >>> result = calc_surface_energies_from_files('slab/OUTCAR', 'bulk/OUTCAR', 2)
    >>> E_slab = result['E_slab']
    >>> E_bulk_per_atom = result['E_bulk_per_atom']
    """
    if not isinstance(slab_outcar, str) or not slab_outcar.strip():
        raise ValueError(
            "slab_outcar 必须是非空字符串路径。"
            "slab_outcar must be a non-empty string path."
        )

    if not isinstance(bulk_outcar, str) or not bulk_outcar.strip():
        raise ValueError(
            "bulk_outcar 必须是非空字符串路径。"
            "bulk_outcar must be a non-empty string path."
        )

    if not isinstance(n_bulk_atoms, (int, np.integer)) or n_bulk_atoms <= 0:
        raise ValueError(
            f"n_bulk_atoms 必须是正整数，但得到 {n_bulk_atoms}。"
            f"n_bulk_atoms must be a positive integer, but got {n_bulk_atoms}."
        )

    # 读取能量 / Read energies
    E_slab = _read_outcar_energy(slab_outcar)
    E_bulk = _read_outcar_energy(bulk_outcar)
    E_bulk_per_atom = E_bulk / n_bulk_atoms

    return {
        'E_slab': E_slab,
        'E_bulk': E_bulk,
        'E_bulk_per_atom': E_bulk_per_atom,
        'n_bulk_atoms': int(n_bulk_atoms),
        'slab_outcar': slab_outcar,
        'bulk_outcar': bulk_outcar,
    }
