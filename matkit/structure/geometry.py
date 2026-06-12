"""
matkit.structure.geometry - 几何计算函数
Geometry calculation functions for MatKit

提供原子坐标的几何计算功能，包括距离、角度、二面角、平面法向量、质心和RMSD等。
Provides geometric calculations for atomic coordinates, including distance,
angle, dihedral, plane normal, center of mass, and RMSD.
"""

import numpy as np


def calc_distance(p1, p2):
    """
    计算两点之间的欧几里得距离。
    Calculate the Euclidean distance between two points.

    Parameters
    ----------
    p1 : array_like
        第一个点的坐标，形状为 (3,) 或 (N, 3)。
        Coordinates of the first point, shape (3,) or (N, 3).
    p2 : array_like
        第二个点的坐标，形状为 (3,) 或 (N, 3)。
        Coordinates of the second point, shape (3,) or (N, 3).

    Returns
    -------
    float or np.ndarray
        两点之间的距离。如果输入为 (N, 3) 形状，则返回 (N,) 形状的数组。
        Distance between two points. If input is (N, 3), returns (N,) array.

    Examples
    --------
    >>> calc_distance(np.array([0, 0, 0]), np.array([1, 1, 1]))
    1.7320508075688772
    >>> calc_distance(np.array([[0,0,0],[1,0,0]]), np.array([[1,0,0],[3,0,0]]))
    array([1., 2.])
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    if p1.ndim == 1:
        if p1.shape != (3,):
            raise ValueError(
                f"p1 必须是形状为 (3,) 的数组，但得到形状 {p1.shape}。"
                f"p1 must have shape (3,), but got shape {p1.shape}."
            )
        if p2.shape != (3,):
            raise ValueError(
                f"p2 必须是形状为 (3,) 的数组，但得到形状 {p2.shape}。"
                f"p2 must have shape (3,), but got shape {p2.shape}."
            )
    elif p1.ndim == 2:
        if p1.shape[1] != 3:
            raise ValueError(
                f"p1 的第二维必须为 3，但得到形状 {p1.shape}。"
                f"p1 second dimension must be 3, but got shape {p1.shape}."
            )
        if p2.shape[1] != 3:
            raise ValueError(
                f"p2 的第二维必须为 3，但得到形状 {p2.shape}。"
                f"p2 second dimension must be 3, but got shape {p2.shape}."
            )
        if p1.shape != p2.shape:
            raise ValueError(
                f"p1 和 p2 的形状必须相同，但得到 {p1.shape} 和 {p2.shape}。"
                f"p1 and p2 must have the same shape, but got {p1.shape} and {p2.shape}."
            )
    else:
        raise ValueError(
            f"输入必须是 1D (3,) 或 2D (N, 3) 数组，但 p1 的维度为 {p1.ndim}。"
            f"Input must be 1D (3,) or 2D (N, 3) array, but p1 has ndim {p1.ndim}."
        )

    diff = p2 - p1
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def calc_angle(p1, p2, p3):
    """
    计算在 p2 处的角度（p1-p2-p3），单位为度。
    Calculate the angle at p2 (p1-p2-p3) in degrees.

    Parameters
    ----------
    p1 : array_like, shape (3,)
        第一个点的坐标。
        Coordinates of the first point.
    p2 : array_like, shape (3,)
        顶点（角度所在位置）的坐标。
        Coordinates of the vertex (where the angle is located).
    p3 : array_like, shape (3,)
        第三个点的坐标。
        Coordinates of the third point.

    Returns
    -------
    float
        角度值，单位为度，范围 [0, 180]。
        Angle in degrees, range [0, 180].

    Examples
    --------
    >>> calc_angle(np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    90.0
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    for name, arr in [("p1", p1), ("p2", p2), ("p3", p3)]:
        if arr.shape != (3,):
            raise ValueError(
                f"{name} 必须是形状为 (3,) 的数组，但得到形状 {arr.shape}。"
                f"{name} must have shape (3,), but got shape {arr.shape}."
            )

    v1 = p1 - p2
    v2 = p3 - p2

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-12 or norm2 < 1e-12:
        raise ValueError(
            "向量长度为零，无法计算角度。p1 或 p3 与 p2 重合。"
            "Vector length is zero, cannot calculate angle. "
            "p1 or p3 coincides with p2."
        )

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    # 数值保护，防止浮点误差导致 cos 超出 [-1, 1] 范围
    # Numerical guard against floating point errors pushing cos outside [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def calc_dihedral(p1, p2, p3, p4):
    """
    计算二面角 p1-p2-p3-p4，单位为度。
    Calculate the dihedral angle p1-p2-p3-p4 in degrees.

    二面角定义为包含 p1-p2-p3 的平面与包含 p2-p3-p4 的平面之间的夹角。
    The dihedral angle is defined as the angle between the plane containing
    p1-p2-p3 and the plane containing p2-p3-p4.

    Parameters
    ----------
    p1 : array_like, shape (3,)
        第一个点的坐标。
        Coordinates of the first point.
    p2 : array_like, shape (3,)
        第二个点的坐标。
        Coordinates of the second point.
    p3 : array_like, shape (3,)
        第三个点的坐标。
        Coordinates of the third point.
    p4 : array_like, shape (3,)
        第四个点的坐标。
        Coordinates of the fourth point.

    Returns
    -------
    float
        二面角，单位为度，范围 (-180, 180]。
        Dihedral angle in degrees, range (-180, 180].

    Examples
    --------
    >>> # 反式构象 (trans conformation)
    >>> calc_dihedral([1,0,0], [0,0,0], [0,1,0], [0,2,0])
    180.0
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    p4 = np.asarray(p4, dtype=np.float64)

    for name, arr in [("p1", p1), ("p2", p2), ("p3", p3), ("p4", p4)]:
        if arr.shape != (3,):
            raise ValueError(
                f"{name} 必须是形状为 (3,) 的数组，但得到形状 {arr.shape}。"
                f"{name} must have shape (3,), but got shape {arr.shape}."
            )

    b1 = p2 - p1  # bond vector p1 -> p2
    b2 = p3 - p2  # bond vector p2 -> p3
    b3 = p4 - p3  # bond vector p3 -> p4

    # 两个平面的法向量 / Normal vectors of the two planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)

    if norm_n1 < 1e-12 or norm_n2 < 1e-12:
        raise ValueError(
            "平面法向量为零，无法计算二面角。可能存在共线点。"
            "Plane normal vector is zero, cannot calculate dihedral. "
            "Collinear points may exist."
        )

    n1 = n1 / norm_n1
    n2 = n2 / norm_n2

    # 二面角的余弦值 / Cosine of the dihedral angle
    cos_dihedral = np.dot(n1, n2)
    cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)

    # 使用叉积的符号确定二面角的正负号
    # Use the sign of the cross product to determine the sign of the dihedral
    sign = np.sign(np.dot(np.cross(n1, n2), b2))

    angle = np.degrees(np.arccos(cos_dihedral))
    if sign < 0:
        angle = -angle

    return angle


def calc_bond_lengths(coords, indices_pairs):
    """
    批量计算键长。
    Batch calculation of bond lengths.

    Parameters
    ----------
    coords : array_like, shape (N, 3)
        所有原子的笛卡尔坐标。
        Cartesian coordinates of all atoms.
    indices_pairs : array_like, shape (M, 2)
        原子索引对，每行为一对 [i, j]。
        Pairs of atom indices, each row is a pair [i, j].

    Returns
    -------
    np.ndarray, shape (M,)
        每对原子之间的键长。
        Bond lengths for each pair of atoms.

    Examples
    --------
    >>> coords = np.array([[0,0,0], [1,0,0], [0,1,0]])
    >>> calc_bond_lengths(coords, [[0,1], [0,2]])
    array([1., 1.])
    """
    coords = np.asarray(coords, dtype=np.float64)
    indices_pairs = np.asarray(indices_pairs, dtype=np.int64)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords 必须是形状为 (N, 3) 的数组，但得到形状 {coords.shape}。"
            f"coords must have shape (N, 3), but got shape {coords.shape}."
        )

    if indices_pairs.ndim != 2 or indices_pairs.shape[1] != 2:
        raise ValueError(
            f"indices_pairs 必须是形状为 (M, 2) 的数组，但得到形状 {indices_pairs.shape}。"
            f"indices_pairs must have shape (M, 2), but got shape {indices_pairs.shape}."
        )

    n_atoms = coords.shape[0]
    max_idx = np.max(indices_pairs)
    if max_idx >= n_atoms:
        raise ValueError(
            f"索引 {max_idx} 超出坐标数组范围（共 {n_atoms} 个原子）。"
            f"Index {max_idx} exceeds coordinate array range ({n_atoms} atoms total)."
        )

    if np.min(indices_pairs) < 0:
        raise ValueError(
            "索引不能为负值。"
            "Indices cannot be negative."
        )

    i_indices = indices_pairs[:, 0]
    j_indices = indices_pairs[:, 1]

    diff = coords[j_indices] - coords[i_indices]
    return np.sqrt(np.sum(diff ** 2, axis=1))


def calc_plane_normal(points):
    """
    计算通过一组点的最佳拟合平面的法向量。
    Calculate the normal vector of the best-fit plane through a set of points.

    使用奇异值分解 (SVD) 方法拟合平面，法向量对应最小奇异值的右奇异向量。
    Uses Singular Value Decomposition (SVD) to fit the plane. The normal vector
    corresponds to the right singular vector of the smallest singular value.

    Parameters
    ----------
    points : array_like, shape (N, 3)
        平面上的点集，至少需要 3 个不共线的点。
        Points on the plane, at least 3 non-collinear points required.

    Returns
    -------
    np.ndarray, shape (3,)
        单位法向量。
        Unit normal vector.

    Raises
    ------
    ValueError
        如果点数少于 3 个，或所有点共线。
        If fewer than 3 points, or all points are collinear.

    Examples
    --------
    >>> pts = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]])
    >>> calc_plane_normal(pts)
    array([ 0.,  0., -1.])
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points 必须是形状为 (N, 3) 的数组，但得到形状 {points.shape}。"
            f"points must have shape (N, 3), but got shape {points.shape}."
        )

    if points.shape[0] < 3:
        raise ValueError(
            f"至少需要 3 个点来拟合平面，但只提供了 {points.shape[0]} 个点。"
            f"At least 3 points are needed to fit a plane, but only {points.shape[0]} provided."
        )

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # SVD 分解 / SVD decomposition
    _, s, vh = np.linalg.svd(centered, full_matrices=False)

    # 共面是正常的（最小奇异值为零），需要检测共线（两个奇异值为零）
    # Coplanar is normal (smallest singular value is zero), need to detect collinear (two singular values zero)
    # 对于 3D 中的点，如果两个最小奇异值都为零，说明共线
    # For points in 3D, if the two smallest singular values are both zero, they are collinear
    n_points = points.shape[0]
    if n_points >= 3:
        # 共线检测：第二小的奇异值也为零
        # Collinear detection: second smallest singular value is also zero
        if len(s) >= 2 and s[-2] < 1e-12:
            raise ValueError(
                "所有点共线，无法确定唯一的平面法向量。"
                "All points are collinear, cannot determine a unique plane normal."
            )

    normal = vh[-1, :]
    # 确保为单位向量 / Ensure unit vector
    normal = normal / np.linalg.norm(normal)
    return normal


def calc_angle_between_planes(normal1, normal2):
    """
    计算两个平面之间的夹角。
    Calculate the angle between two planes.

    Parameters
    ----------
    normal1 : array_like, shape (3,)
        第一个平面的法向量。
        Normal vector of the first plane.
    normal2 : array_like, shape (3,)
        第二个平面的法向量。
        Normal vector of the second plane.

    Returns
    -------
    float
        两平面之间的夹角，单位为度，范围 [0, 90]。
        Angle between the two planes in degrees, range [0, 90].

    Examples
    --------
    >>> calc_angle_between_planes([0, 0, 1], [1, 0, 0])
    90.0
    >>> calc_angle_between_planes([0, 0, 1], [0, 0, -1])
    0.0
    """
    normal1 = np.asarray(normal1, dtype=np.float64)
    normal2 = np.asarray(normal2, dtype=np.float64)

    for name, arr in [("normal1", normal1), ("normal2", normal2)]:
        if arr.shape != (3,):
            raise ValueError(
                f"{name} 必须是形状为 (3,) 的数组，但得到形状 {arr.shape}。"
                f"{name} must have shape (3,), but got shape {arr.shape}."
            )

    norm1 = np.linalg.norm(normal1)
    norm2 = np.linalg.norm(normal2)

    if norm1 < 1e-12 or norm2 < 1e-12:
        raise ValueError(
            "法向量不能为零向量。"
            "Normal vectors cannot be zero vectors."
        )

    cos_angle = np.abs(np.dot(normal1, normal2)) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, 0.0, 1.0)

    # 平面夹角取锐角（0-90度）
    # Plane angle is the acute angle (0-90 degrees)
    return np.degrees(np.arccos(cos_angle))


def calc_center_of_mass(coords, masses):
    """
    计算质心。
    Calculate the center of mass.

    Parameters
    ----------
    coords : array_like, shape (N, 3)
        原子的笛卡尔坐标。
        Cartesian coordinates of atoms.
    masses : array_like, shape (N,)
        每个原子的质量。
        Mass of each atom.

    Returns
    -------
    np.ndarray, shape (3,)
        质心坐标。
        Center of mass coordinates.

    Examples
    --------
    >>> coords = np.array([[0,0,0], [1,0,0]])
    >>> masses = np.array([1.0, 1.0])
    >>> calc_center_of_mass(coords, masses)
    array([0.5, 0., 0.])
    """
    coords = np.asarray(coords, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords 必须是形状为 (N, 3) 的数组，但得到形状 {coords.shape}。"
            f"coords must have shape (N, 3), but got shape {coords.shape}."
        )

    if masses.ndim != 1:
        raise ValueError(
            f"masses 必须是 1D 数组，但得到维度 {masses.ndim}。"
            f"masses must be a 1D array, but got ndim {masses.ndim}."
        )

    n_atoms = coords.shape[0]
    if masses.shape[0] != n_atoms:
        raise ValueError(
            f"coords 和 masses 的长度不匹配：coords 有 {n_atoms} 个原子，"
            f"masses 有 {masses.shape[0]} 个值。"
            f"coords and masses length mismatch: coords has {n_atoms} atoms, "
            f"masses has {masses.shape[0]} values."
        )

    total_mass = np.sum(masses)
    if total_mass <= 0:
        raise ValueError(
            f"总质量必须为正数，但得到 {total_mass}。"
            f"Total mass must be positive, but got {total_mass}."
        )

    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    return com


def calc_rmsd(coords1, coords2):
    """
    计算两组坐标之间的均方根偏差 (RMSD)。
    Calculate the Root Mean Square Deviation (RMSD) between two coordinate sets.

    在计算 RMSD 之前，会对两组坐标进行 Kabsch 对齐（最优旋转+平移），
    以消除刚体旋转和平移的影响。
    Before computing RMSD, the two coordinate sets are aligned using the
    Kabsch algorithm (optimal rotation + translation) to remove the effects
    of rigid body rotation and translation.

    Parameters
    ----------
    coords1 : array_like, shape (N, 3)
        第一组坐标。
        First coordinate set.
    coords2 : array_like, shape (N, 3)
        第二组坐标。
        Second coordinate set.

    Returns
    -------
    float
        RMSD 值（Å）。
        RMSD value in Angstroms.

    Raises
    ------
    ValueError
        如果两组坐标的形状不匹配，或原子数少于 3 个。

    Examples
    --------
    >>> c1 = np.array([[0,0,0], [1,0,0], [0,1,0]])
    >>> c2 = np.array([[0,0,0], [1,0,0], [0,1,0]])
    >>> calc_rmsd(c1, c2)
    0.0
    """
    coords1 = np.asarray(coords1, dtype=np.float64)
    coords2 = np.asarray(coords2, dtype=np.float64)

    if coords1.ndim != 2 or coords1.shape[1] != 3:
        raise ValueError(
            f"coords1 必须是形状为 (N, 3) 的数组，但得到形状 {coords1.shape}。"
            f"coords1 must have shape (N, 3), but got shape {coords1.shape}."
        )

    if coords2.ndim != 2 or coords2.shape[1] != 3:
        raise ValueError(
            f"coords2 必须是形状为 (N, 3) 的数组，但得到形状 {coords2.shape}。"
            f"coords2 must have shape (N, 3), but got shape {coords2.shape}."
        )

    if coords1.shape != coords2.shape:
        raise ValueError(
            f"两组坐标的形状必须相同，但得到 {coords1.shape} 和 {coords2.shape}。"
            f"Coordinate sets must have the same shape, but got {coords1.shape} and {coords2.shape}."
        )

    n = coords1.shape[0]
    if n < 3:
        raise ValueError(
            f"至少需要 3 个原子来计算 RMSD，但只提供了 {n} 个。"
            f"At least 3 atoms are needed to compute RMSD, but only {n} provided."
        )

    # 平移到质心 / Translate to centroids
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)

    centered1 = coords1 - centroid1
    centered2 = coords2 - centroid2

    # Kabsch 算法：寻找最优旋转矩阵
    # Kabsch algorithm: find optimal rotation matrix
    # 计算协方差矩阵 / Compute covariance matrix
    H = centered1.T @ centered2

    # SVD 分解 / SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # 确保右手坐标系 / Ensure right-handed coordinate system
    d = np.linalg.det(U @ Vt)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = 1.0 if d >= 0 else -1.0

    # 最优旋转矩阵 / Optimal rotation matrix
    R = U @ sign_matrix @ Vt

    # 旋转 centered2 / Rotate centered2
    rotated2 = (R @ centered2.T).T

    # 计算 RMSD / Compute RMSD
    diff = centered1 - rotated2
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    return rmsd
