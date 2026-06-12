"""
matkit.structure.surface - 表面分析
Surface analysis for MatKit

提供表面层识别、表面积计算和表面弛豫分析功能。
Provides surface layer identification, surface area calculation,
and surface relaxation analysis.
"""

import numpy as np


def identify_surface_layers(atoms, coords, element, tolerance=0.5):
    """
    通过 z 坐标聚类识别表面层。
    Identify surface layers by z-coordinate clustering.

    将指定元素的原子按 z 坐标排序，然后根据给定的容差进行分层聚类。
    Sorts atoms of the specified element by z-coordinate, then clusters them
    into layers based on the given tolerance.

    Parameters
    ----------
    atoms : list of str
        所有原子的元素符号列表。
        List of element symbols for all atoms.
    coords : array_like, shape (N, 3)
        所有原子的笛卡尔坐标。
        Cartesian coordinates of all atoms.
    element : str
        要分析的表面元素符号，如 'Cu', 'Pt'。
        Element symbol of the surface to analyze, e.g., 'Cu', 'Pt'.
    tolerance : float, optional
        z 坐标聚类容差（Å），默认为 0.5。
        z-coordinate clustering tolerance in Angstroms, default 0.5.

    Returns
    -------
    list of dict
        表面层列表，按 z_avg 从小到大排序，每层为字典：
        Surface layer list, sorted by z_avg ascending, each layer is a dict:
        - 'indices': list of int - 该层原子的全局索引 / Global atom indices in this layer
        - 'z_avg': float - 该层平均 z 坐标 / Average z-coordinate of this layer
        - 'z_min': float - 该层最小 z 坐标 / Minimum z-coordinate of this layer
        - 'z_max': float - 该层最大 z 坐标 / Maximum z-coordinate of this layer

    Raises
    ------
    ValueError
        如果输入验证失败或没有找到指定元素的原子。
        If input validation fails or no atoms of the specified element are found.

    Examples
    --------
    >>> atoms = ['Cu','Cu','Cu','Cu','Cu','Cu']
    >>> coords = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,3],[1,0,3],[0,1,3]])
    >>> identify_surface_layers(atoms, coords, 'Cu', tolerance=0.5)
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

    if not isinstance(element, str) or not element.strip():
        raise ValueError(
            "element 必须是非空字符串。"
            "element must be a non-empty string."
        )

    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError(
            f"tolerance 必须是正数，但得到 {tolerance}。"
            f"tolerance must be a positive number, but got {tolerance}."
        )

    # 筛选指定元素的原子 / Filter atoms of the specified element
    target_indices = [i for i, a in enumerate(atoms) if a == element]

    if not target_indices:
        raise ValueError(
            f"没有找到元素 '{element}' 的原子。"
            f"No atoms of element '{element}' found."
        )

    target_coords = coords[target_indices]
    z_values = target_coords[:, 2]

    # 按 z 坐标排序 / Sort by z-coordinate
    sort_order = np.argsort(z_values)
    sorted_z = z_values[sort_order]
    sorted_indices = [target_indices[i] for i in sort_order]

    # 聚类分层 / Cluster into layers
    layers = []
    current_layer_indices = [sorted_indices[0]]
    current_z_values = [sorted_z[0]]

    for i in range(1, len(sorted_z)):
        if sorted_z[i] - np.mean(current_z_values) <= tolerance:
            current_layer_indices.append(sorted_indices[i])
            current_z_values.append(sorted_z[i])
        else:
            # 保存当前层 / Save current layer
            z_arr = np.array(current_z_values)
            layers.append({
                'indices': sorted(current_layer_indices),
                'z_avg': float(np.mean(z_arr)),
                'z_min': float(np.min(z_arr)),
                'z_max': float(np.max(z_arr)),
            })
            # 开始新层 / Start new layer
            current_layer_indices = [sorted_indices[i]]
            current_z_values = [sorted_z[i]]

    # 保存最后一层 / Save last layer
    if current_layer_indices:
        z_arr = np.array(current_z_values)
        layers.append({
            'indices': sorted(current_layer_indices),
            'z_avg': float(np.mean(z_arr)),
            'z_min': float(np.min(z_arr)),
            'z_max': float(np.max(z_arr)),
        })

    # 按 z_avg 排序 / Sort by z_avg
    layers.sort(key=lambda x: x['z_avg'])

    return layers


def calc_surface_area(lattice, miller_index=(0, 0, 1)):
    """
    从晶格向量计算表面面积。
    Calculate surface area from lattice vectors.

    根据密勒指数 (h, k, l) 和晶格向量计算对应的表面面积。
    Calculates the surface area corresponding to the given Miller indices
    and lattice vectors.

    Parameters
    ----------
    lattice : array_like, shape (3, 3)
        晶格向量，每行为一个晶格矢量 [a, b, c]。
        Lattice vectors, each row is a lattice vector [a, b, c].
    miller_index : tuple of int, optional
        密勒指数 (h, k, l)，默认为 (0, 0, 1)。
        Miller indices (h, k, l), default (0, 0, 1).

    Returns
    -------
    dict
        包含以下键的字典：
        Dict with the following keys:
        - 'surface_area_A2': float - 表面面积（Å²）/ Surface area in Angstroms squared
        - 'miller_index': tuple - 密勒指数 / Miller indices
        - 'surface_vectors': np.ndarray, shape (2, 3) - 表面基矢 / Surface basis vectors

    Raises
    ------
    ValueError
        如果输入验证失败或密勒指数全为零。
        If input validation fails or Miller indices are all zero.

    Examples
    --------
    >>> lattice = np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 4.0]])
    >>> calc_surface_area(lattice, (0, 0, 1))
    """
    lattice = np.asarray(lattice, dtype=np.float64)

    if lattice.ndim != 2 or lattice.shape != (3, 3):
        raise ValueError(
            f"lattice 必须是形状为 (3, 3) 的数组，但得到形状 {lattice.shape}。"
            f"lattice must have shape (3, 3), but got shape {lattice.shape}."
        )

    if not isinstance(miller_index, (list, tuple)) or len(miller_index) != 3:
        raise ValueError(
            f"miller_index 必须是包含 3 个整数的元组，但得到 {miller_index}。"
            f"miller_index must be a tuple of 3 integers, but got {miller_index}."
        )

    h, k, l = miller_index

    if not all(isinstance(x, (int, np.integer)) for x in (h, k, l)):
        raise ValueError(
            f"密勒指数必须为整数，但得到 ({h}, {k}, {l})。"
            f"Miller indices must be integers, but got ({h}, {k}, {l})."
        )

    if h == 0 and k == 0 and l == 0:
        raise ValueError(
            "密勒指数 (0, 0, 0) 无效。"
            "Miller index (0, 0, 0) is invalid."
        )

    h, k, l = int(h), int(k), int(l)

    a = lattice[0]
    b = lattice[1]
    c = lattice[2]

    # 计算倒格子矢量 / Compute reciprocal lattice vectors
    volume = np.dot(a, np.cross(b, c))

    if abs(volume) < 1e-12:
        raise ValueError(
            "晶格体积为零，晶格向量可能共面。"
            "Lattice volume is zero, lattice vectors may be coplanar."
        )

    # 倒格子矢量 / Reciprocal lattice vectors
    a_star = np.cross(b, c) / volume
    b_star = np.cross(c, a) / volume
    c_star = np.cross(a, b) / volume

    # 面法向量 = h*a* + k*b* + l*c*
    # Surface normal = h*a* + k*b* + l*c*
    normal = h * a_star + k * b_star + l * c_star
    normal = normal / np.linalg.norm(normal)

    # 找到两个与法向量垂直的晶格矢量作为表面基矢
    # Find two lattice vectors perpendicular to the normal as surface basis vectors
    # 使用叉积方法构造表面基矢
    # Use cross product method to construct surface basis vectors

    # 选择两个不平行于法向量的晶格矢量
    # Select two lattice vectors not parallel to the normal
    candidates = [a, b, c]
    in_plane = []
    for vec in candidates:
        projection = np.dot(vec, normal)
        in_plane_vec = vec - projection * normal
        if np.linalg.norm(in_plane_vec) > 1e-8:
            in_plane.append(in_plane_vec)

    if len(in_plane) < 2:
        raise ValueError(
            "无法找到足够的非共线表面基矢。"
            "Cannot find enough non-collinear surface basis vectors."
        )

    # 使用 Gram-Schmidt 正交化
    # Use Gram-Schmidt orthogonalization
    v1 = in_plane[0]
    v1 = v1 / np.linalg.norm(v1)

    v2_raw = in_plane[1]
    v2 = v2_raw - np.dot(v2_raw, v1) * v1
    if np.linalg.norm(v2) < 1e-8:
        # 尝试第三个候选 / Try third candidate
        if len(in_plane) > 2:
            v2_raw = in_plane[2]
            v2 = v2_raw - np.dot(v2_raw, v1) * v1
        if np.linalg.norm(v2) < 1e-8:
            raise ValueError(
                "无法构造两个独立的表面基矢。"
                "Cannot construct two independent surface basis vectors."
            )
    v2 = v2 / np.linalg.norm(v2)

    # 计算表面基矢的长度（使用晶格矢量的投影）
    # Calculate lengths of surface basis vectors (using lattice vector projections)
    # 找到实际晶格矢量的投影长度
    # Find projection lengths of actual lattice vectors

    # 更精确的方法：直接使用晶格矢量叉积
    # More precise method: use cross product of lattice vectors directly
    # 对于 (hkl) 面，面积 = |d_hkl| * |a x b| / |n . (a x b)|
    # 但更简单的方法是使用两个表面基矢的叉积

    # 使用投影到表面平面的晶格矢量来计算面积
    # Use lattice vectors projected onto the surface plane to compute area
    projected = []
    for vec in candidates:
        proj = vec - np.dot(vec, normal) * normal
        if np.linalg.norm(proj) > 1e-8:
            projected.append(proj)

    # 选择两个投影后面积最大的组合
    # Choose the combination of two with the largest projected area
    max_area = 0
    best_pair = None
    for i in range(len(projected)):
        for j in range(i + 1, len(projected)):
            area = np.linalg.norm(np.cross(projected[i], projected[j]))
            if area > max_area:
                max_area = area
                best_pair = (projected[i], projected[j])

    if best_pair is not None and max_area > 1e-8:
        surface_area = max_area
        surface_vectors = np.array(best_pair)
    else:
        # 回退到正交化方法 / Fall back to orthogonalization method
        surface_area = np.linalg.norm(np.cross(v1, v2))
        surface_vectors = np.array([v1, v2])

    return {
        'surface_area_A2': float(surface_area),
        'miller_index': (h, k, l),
        'surface_vectors': surface_vectors,
    }


def analyze_surface_relaxation(coords_clean, coords_adsorbed, surface_indices, tolerance=0.3):
    """
    比较清洁表面和吸附后表面，找出发生弛豫的原子。
    Compare clean vs adsorbed surface to find relaxed atoms.

    Parameters
    ----------
    coords_clean : array_like, shape (N, 3)
        清洁表面的原子坐标。
        Atomic coordinates of the clean surface.
    coords_adsorbed : array_like, shape (M, 3)
        吸附后体系的原子坐标（可能包含吸附物原子）。
        Atomic coordinates of the adsorbed system (may include adsorbate atoms).
    surface_indices : list of int
        表面原子的索引列表（在 coords_clean 中的索引）。
        List of surface atom indices (indices in coords_clean).
    tolerance : float, optional
        弛豫判定阈值（Å），位移大于此值视为弛豫，默认为 0.3。
        Relaxation threshold in Angstroms, displacements larger than this
        are considered relaxed, default 0.3.

    Returns
    -------
    dict
        弛豫分析结果字典：
        Relaxation analysis result dict:
        - 'relaxed_atoms': list of dict - 发生弛豫的原子信息 / Relaxed atom info
            - 'index': int - 原子索引 / Atom index
            - 'displacement': float - 位移量（Å）/ Displacement in Angstroms
            - 'displacement_vector': np.ndarray - 位移矢量 / Displacement vector
            - 'dz': float - z 方向位移 / z-direction displacement
            - 'dx': float - x 方向位移 / x-direction displacement
            - 'dy': float - y 方向位移 / y-direction displacement
        - 'n_relaxed': int - 弛豫原子数 / Number of relaxed atoms
        - 'n_unrelaxed': int - 未弛豫原子数 / Number of unrelaxed atoms
        - 'avg_displacement': float - 平均位移 / Average displacement
        - 'max_displacement': float - 最大位移 / Maximum displacement
        - 'all_displacements': list of dict - 所有表面原子的位移信息 / Displacement info for all surface atoms

    Raises
    ------
    ValueError
        如果输入验证失败。
        If input validation fails.

    Examples
    --------
    >>> coords_clean = np.array([[0,0,0], [2,0,0], [0,2,0]])
    >>> coords_adsorbed = np.array([[0,0,0], [2,0,0.1], [0,2,-0.05]])
    >>> analyze_surface_relaxation(coords_clean, coords_adsorbed, [0,1,2])
    """
    coords_clean = np.asarray(coords_clean, dtype=np.float64)
    coords_adsorbed = np.asarray(coords_adsorbed, dtype=np.float64)

    if coords_clean.ndim != 2 or coords_clean.shape[1] != 3:
        raise ValueError(
            f"coords_clean 必须是形状为 (N, 3) 的数组，但得到形状 {coords_clean.shape}。"
            f"coords_clean must have shape (N, 3), but got shape {coords_clean.shape}."
        )

    if coords_adsorbed.ndim != 2 or coords_adsorbed.shape[1] != 3:
        raise ValueError(
            f"coords_adsorbed 必须是形状为 (M, 3) 的数组，但得到形状 {coords_adsorbed.shape}。"
            f"coords_adsorbed must have shape (M, 3), but got shape {coords_adsorbed.shape}."
        )

    n_clean = coords_clean.shape[0]
    n_adsorbed = coords_adsorbed.shape[0]

    if n_adsorbed < n_clean:
        raise ValueError(
            f"吸附后体系的原子数 ({n_adsorbed}) 不能少于清洁表面 ({n_clean})。"
            f"Adsorbed system atom count ({n_adsorbed}) cannot be less than "
            f"clean surface ({n_clean})."
        )

    if not isinstance(surface_indices, (list, tuple)):
        raise ValueError(
            "surface_indices 必须是列表或元组。"
            "surface_indices must be a list or tuple."
        )

    for idx in surface_indices:
        if not isinstance(idx, (int, np.integer)):
            raise ValueError(
                f"surface_indices 中的索引必须为整数，但得到 {type(idx).__name__}。"
                f"Indices in surface_indices must be integers, but got {type(idx).__name__}."
            )
        if idx < 0 or idx >= n_clean:
            raise ValueError(
                f"索引 {idx} 超出范围 [0, {n_clean - 1}]。"
                f"Index {idx} is out of range [0, {n_clean - 1}]."
            )

    if not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise ValueError(
            f"tolerance 必须为非负数，但得到 {tolerance}。"
            f"tolerance must be non-negative, but got {tolerance}."
        )

    # 计算所有表面原子的位移
    # Compute displacements for all surface atoms
    all_displacements = []
    relaxed_atoms = []

    for idx in surface_indices:
        disp_vector = coords_adsorbed[idx] - coords_clean[idx]
        disp_mag = np.linalg.norm(disp_vector)

        disp_info = {
            'index': int(idx),
            'displacement': float(disp_mag),
            'displacement_vector': disp_vector.copy(),
            'dx': float(disp_vector[0]),
            'dy': float(disp_vector[1]),
            'dz': float(disp_vector[2]),
        }
        all_displacements.append(disp_info)

        if disp_mag > tolerance:
            relaxed_atoms.append(disp_info)

    # 按位移量排序弛豫原子 / Sort relaxed atoms by displacement
    relaxed_atoms.sort(key=lambda x: x['displacement'], reverse=True)

    displacements = [d['displacement'] for d in all_displacements]

    return {
        'relaxed_atoms': relaxed_atoms,
        'n_relaxed': len(relaxed_atoms),
        'n_unrelaxed': len(surface_indices) - len(relaxed_atoms),
        'avg_displacement': float(np.mean(displacements)) if displacements else 0.0,
        'max_displacement': float(np.max(displacements)) if displacements else 0.0,
        'all_displacements': all_displacements,
    }
