"""
matkit.structure.molecule - 分子识别与分析
Molecule identification and analysis for MatKit

提供基于距离的分子识别、化学式匹配和吸附位点分析功能。
Provides distance-based molecule identification, chemical formula matching,
and adsorption site analysis.
"""

import numpy as np
from collections import Counter, defaultdict


def _parse_formula(formula):
    """
    解析化学式字符串为元素计数字典。
    Parse a chemical formula string into an element count dictionary.

    Parameters
    ----------
    formula : str
        化学式字符串，如 "SO4", "C6H5N3"。
        Chemical formula string, e.g., "SO4", "C6H5N3".

    Returns
    -------
    dict
        元素到原子数的映射，如 {'S': 1, 'O': 4}。
        Element to atom count mapping, e.g., {'S': 1, 'O': 4}.
    """
    import re

    # 匹配元素符号和可选的数字后缀
    # Match element symbols and optional numeric suffixes
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)

    if not matches:
        raise ValueError(
            f"无法解析化学式 '{formula}'。"
            f"Cannot parse formula '{formula}'."
        )

    counts = {}
    for element, num_str in matches:
        if not element:
            continue
        num = int(num_str) if num_str else 1
        counts[element] = counts.get(element, 0) + num

    return counts


def _build_adjacency(atoms, coords, bond_cutoffs):
    """
    根据原子间距离和截断值构建邻接表。
    Build adjacency list based on interatomic distances and cutoff values.

    Parameters
    ----------
    atoms : list of str
        元素符号列表。
        List of element symbols.
    coords : np.ndarray, shape (N, 3)
        原子坐标。
        Atomic coordinates.
    bond_cutoffs : dict
        键长截断值字典，键为元素对元组，值为截断距离。
        Bond cutoff dictionary, keys are element pair tuples, values are cutoff distances.

    Returns
    -------
    dict
        邻接表，键为原子索引，值为相邻原子索引列表。
        Adjacency list, keys are atom indices, values are lists of neighbor indices.
    """
    n_atoms = len(atoms)
    adjacency = defaultdict(list)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            elem_pair = (atoms[i], atoms[j])
            # 尝试两种顺序的元素对 / Try both orderings of element pair
            cutoff = bond_cutoffs.get(elem_pair, None)
            if cutoff is None:
                cutoff = bond_cutoffs.get((atoms[j], atoms[i]), None)

            if cutoff is not None and dist <= cutoff:
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency


def _find_connected_components(adjacency, n_atoms):
    """
    使用广度优先搜索 (BFS) 查找连通分量。
    Find connected components using Breadth-First Search (BFS).

    Parameters
    ----------
    adjacency : dict
        邻接表。
        Adjacency list.
    n_atoms : int
        原子总数。
        Total number of atoms.

    Returns
    -------
    list of list
        连通分量列表，每个分量是一个原子索引列表。
        List of connected components, each component is a list of atom indices.
    """
    visited = set()
    components = []

    for i in range(n_atoms):
        if i in visited:
            continue
        # BFS
        component = []
        queue = [i]
        visited.add(i)
        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))

    return components


def _generate_formula(elements):
    """
    根据元素列表生成化学式字符串。
    Generate a chemical formula string from a list of elements.

    使用 Hill 系统排序：C 在前，H 其次，其余按字母顺序。
    Uses Hill system ordering: C first, H second, rest alphabetically.

    Parameters
    ----------
    elements : list of str
        元素符号列表。
        List of element symbols.

    Returns
    -------
    str
        化学式字符串。
        Chemical formula string.
    """
    counts = Counter(elements)

    # Hill 系统排序 / Hill system ordering
    ordered = []
    if 'C' in counts:
        ordered.append('C')
        del counts['C']
    if 'H' in counts:
        ordered.append('H')
        del counts['H']
    # 剩余元素按字母顺序 / Remaining elements in alphabetical order
    for elem in sorted(counts.keys()):
        ordered.append(elem)

    formula_parts = []
    for elem in ordered:
        count = counts.get(elem, 0)
        if elem in ('C', 'H'):
            # 从原始 Counter 中获取 / Get from original Counter
            count = Counter(elements).get(elem, 0)
        else:
            count = Counter(elements).get(elem, 0)
        if count == 1:
            formula_parts.append(elem)
        else:
            formula_parts.append(f"{elem}{count}")

    return "".join(formula_parts)


def identify_molecules(atoms, coords, bond_cutoffs):
    """
    通过原子间距离构建邻接图，识别连通分量（分子）。
    Build adjacency graph from interatomic distances and find connected components (molecules).

    Parameters
    ----------
    atoms : list of str
        所有原子的元素符号列表，如 ['Cu', 'O', 'O', 'S', 'O', 'O', 'O']。
        List of element symbols for all atoms, e.g., ['Cu', 'O', 'O', 'S', 'O', 'O', 'O'].
    coords : array_like, shape (N, 3)
        所有原子的笛卡尔坐标。
        Cartesian coordinates of all atoms.
    bond_cutoffs : dict
        键长截断值字典。键为元素对元组（排序无关），值为最大键长。
        例如 {('O', 'Cu'): 2.5, ('S', 'O'): 2.0}。
        Bond cutoff dictionary. Keys are element pair tuples (order-independent),
        values are maximum bond lengths.
        E.g., {('O', 'Cu'): 2.5, ('S', 'O'): 2.0}.

    Returns
    -------
    list of dict
        分子列表，每个分子为字典：
        Molecule list, each molecule is a dict:
        - 'indices': list of int - 原子在原始体系中的索引 / Atom indices in the original system
        - 'elements': list of str - 元素符号列表 / List of element symbols
        - 'coords': np.ndarray, shape (M, 3) - 分子坐标 / Molecular coordinates
        - 'formula': str - 化学式 / Chemical formula

    Raises
    ------
    ValueError
        如果输入验证失败。
        If input validation fails.

    Examples
    --------
    >>> atoms = ['Cu', 'O', 'S', 'O', 'O', 'O']
    >>> coords = np.array([[0,0,0], [1.8,0,0], [5,0,0], [5.5,0,0], [5,1.5,0], [5,-1.5,0]])
    >>> cutoffs = {('O','Cu'): 2.5, ('S','O'): 2.0}
    >>> mols = identify_molecules(atoms, coords, cutoffs)
    """
    coords = np.asarray(coords, dtype=np.float64)

    if not isinstance(atoms, (list, tuple)):
        raise ValueError(
            "atoms 必须是列表或元组。"
            "atoms must be a list or tuple."
        )

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords 必须是形状为 (N, 3) 的数组，但得到形状 {coords.shape}。"
            f"coords must have shape (N, 3), but got shape {coords.shape}."
        )

    if len(atoms) != coords.shape[0]:
        raise ValueError(
            f"atoms 和 coords 的长度不匹配：atoms 有 {len(atoms)} 个，"
            f"coords 有 {coords.shape[0]} 个。"
            f"atoms and coords length mismatch: atoms has {len(atoms)}, "
            f"coords has {coords.shape[0]}."
        )

    if not isinstance(bond_cutoffs, dict):
        raise ValueError(
            "bond_cutoffs 必须是字典。"
            "bond_cutoffs must be a dict."
        )

    # 验证 bond_cutoffs 的键 / Validate bond_cutoffs keys
    for key, value in bond_cutoffs.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                f"bond_cutoffs 的键必须是包含两个元素符号的元组，但得到 {key}。"
                f"bond_cutoffs keys must be tuples of two element symbols, but got {key}."
            )
        if not isinstance(key[0], str) or not isinstance(key[1], str):
            raise ValueError(
                f"bond_cutoffs 的键元素必须是字符串，但得到 ({type(key[0]).__name__}, {type(key[1]).__name__})。"
                f"bond_cutoffs key elements must be strings, but got ({type(key[0]).__name__}, {type(key[1]).__name__})."
            )
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(
                f"bond_cutoffs 的值必须是正数，但得到 {value}。"
                f"bond_cutoffs values must be positive numbers, but got {value}."
            )

    n_atoms = len(atoms)

    # 构建邻接表 / Build adjacency list
    adjacency = _build_adjacency(atoms, coords, bond_cutoffs)

    # 查找连通分量 / Find connected components
    components = _find_connected_components(adjacency, n_atoms)

    # 构建分子列表 / Build molecule list
    molecules = []
    for component in components:
        mol_indices = sorted(component)
        mol_elements = [atoms[i] for i in mol_indices]
        mol_coords = coords[mol_indices].copy()
        mol_formula = _generate_formula(mol_elements)

        molecules.append({
            'indices': mol_indices,
            'elements': mol_elements,
            'coords': mol_coords,
            'formula': mol_formula,
        })

    return molecules


def match_molecule(molecules, formula):
    """
    在分子列表中查找匹配指定化学式的分子。
    Find a molecule matching a specified chemical formula in a molecule list.

    Parameters
    ----------
    molecules : list of dict
        由 identify_molecules 返回的分子列表。
        Molecule list returned by identify_molecules.
    formula : str
        要匹配的化学式字符串，如 "SO4", "C6H5N3"。
        Chemical formula string to match, e.g., "SO4", "C6H5N3".

    Returns
    -------
    dict or None
        匹配的分子字典，如果没有匹配则返回 None。
        Matched molecule dict, or None if no match found.

    Raises
    ------
    ValueError
        如果输入验证失败。
        If input validation fails.

    Examples
    --------
    >>> mols = [{'formula': 'SO4', 'indices': [0,1,2,3,4], ...}]
    >>> match_molecule(mols, 'SO4')  # returns the SO4 molecule dict
    """
    if not isinstance(molecules, list):
        raise ValueError(
            "molecules 必须是列表。"
            "molecules must be a list."
        )

    if not isinstance(formula, str) or not formula.strip():
        raise ValueError(
            "formula 必须是非空字符串。"
            "formula must be a non-empty string."
        )

    # 解析目标化学式 / Parse target formula
    target_counts = _parse_formula(formula)

    for mol in molecules:
        if not isinstance(mol, dict) or 'formula' not in mol:
            continue

        try:
            mol_counts = _parse_formula(mol['formula'])
        except ValueError:
            continue

        if mol_counts == target_counts:
            return mol

    return None


def find_adsorption_sites(molecule, surface_elements, coords, cutoff):
    """
    查找分子中与表面原子距离在截断值以内的吸附位点。
    Find adsorption sites: atoms in the molecule that are within cutoff distance
    of surface atoms.

    Parameters
    ----------
    molecule : dict
        分子字典，由 identify_molecules 返回，包含 'indices', 'elements', 'coords' 键。
        Molecule dict returned by identify_molecules, with 'indices', 'elements', 'coords' keys.
    surface_elements : list of str
        所有原子中表面原子的元素标记列表（与 coords 等长）。
        对于表面原子，标记为其元素符号；对于非表面原子，标记为 None 或空字符串。
        Element labels for surface atoms (same length as coords).
        For surface atoms, labeled with element symbol; for non-surface atoms,
        labeled as None or empty string.
    coords : array_like, shape (N, 3)
        所有原子的笛卡尔坐标。
        Cartesian coordinates of all atoms.
    cutoff : float
        吸附位点的距离截断值（Å）。
        Distance cutoff for adsorption sites in Angstroms.

    Returns
    -------
    list of dict
        吸附位点信息列表，每个条目为字典：
        Adsorption site info list, each entry is a dict:
        - 'mol_index': int - 分子内原子索引 / Atom index within molecule
        - 'global_index': int - 全局原子索引 / Global atom index
        - 'element': str - 元素符号 / Element symbol
        - 'min_distance': float - 到最近表面原子的距离 / Distance to nearest surface atom
        - 'surface_neighbors': list of dict - 附近的表面原子信息 / Nearby surface atom info

    Raises
    ------
    ValueError
        如果输入验证失败。
        If input validation fails.

    Examples
    --------
    >>> mol = {'indices': [5,6,7], 'elements': ['S','O','O'], 'coords': np.array([[5,0,0],[5.5,0,0],[5,1.5,0]])}
    >>> surface_elems = ['Cu','Cu','Cu',None,None,None,None]
    >>> coords = np.array([[0,0,0],[2,0,0],[4,0,0],[5,0,0],[5.5,0,0],[5,0,0],[5,1.5,0],[5,-1.5,0]])
    >>> find_adsorption_sites(mol, surface_elems, coords, 3.0)
    """
    coords = np.asarray(coords, dtype=np.float64)

    if not isinstance(molecule, dict):
        raise ValueError(
            "molecule 必须是字典。"
            "molecule must be a dict."
        )

    required_keys = {'indices', 'elements', 'coords'}
    if not required_keys.issubset(molecule.keys()):
        missing = required_keys - set(molecule.keys())
        raise ValueError(
            f"molecule 字典缺少必要的键：{missing}。"
            f"molecule dict is missing required keys: {missing}."
        )

    if not isinstance(surface_elements, (list, tuple)):
        raise ValueError(
            "surface_elements 必须是列表或元组。"
            "surface_elements must be a list or tuple."
        )

    if len(surface_elements) != coords.shape[0]:
        raise ValueError(
            f"surface_elements 和 coords 的长度不匹配：surface_elements 有 {len(surface_elements)} 个，"
            f"coords 有 {coords.shape[0]} 个。"
            f"surface_elements and coords length mismatch: surface_elements has {len(surface_elements)}, "
            f"coords has {coords.shape[0]}."
        )

    if not isinstance(cutoff, (int, float)) or cutoff <= 0:
        raise ValueError(
            f"cutoff 必须是正数，但得到 {cutoff}。"
            f"cutoff must be a positive number, but got {cutoff}."
        )

    # 获取表面原子的全局索引 / Get global indices of surface atoms
    surface_indices = []
    for i, elem in enumerate(surface_elements):
        if elem is not None and elem != '':
            surface_indices.append(i)

    if not surface_indices:
        return []

    surface_coords = coords[surface_indices]

    # 检查分子中的每个原子 / Check each atom in the molecule
    adsorption_sites = []

    for mol_i, global_idx in enumerate(molecule['indices']):
        atom_coord = coords[global_idx]
        distances = np.linalg.norm(surface_coords - atom_coord, axis=1)

        min_dist = np.min(distances)
        if min_dist <= cutoff:
            # 找到所有在截断值内的表面原子
            # Find all surface atoms within cutoff
            within_mask = distances <= cutoff
            neighbor_global_indices = [surface_indices[k] for k in range(len(surface_indices)) if within_mask[k]]
            neighbor_distances = distances[within_mask]

            surface_neighbors = []
            for n_idx, n_dist in zip(neighbor_global_indices, neighbor_distances):
                surface_neighbors.append({
                    'surface_index': int(n_idx),
                    'element': surface_elements[n_idx],
                    'distance': float(n_dist),
                })

            adsorption_sites.append({
                'mol_index': mol_i,
                'global_index': int(global_idx),
                'element': molecule['elements'][mol_i],
                'min_distance': float(min_dist),
                'surface_neighbors': surface_neighbors,
            })

    return adsorption_sites
