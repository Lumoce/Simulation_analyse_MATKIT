"""
POSCAR / CONTCAR 解析器
=======================

读写 VASP POSCAR 和 CONTCAR 格式文件。

功能:
- 读取 POSCAR/CONTCAR 文件，返回包含结构信息的字典
- 写入 POSCAR 文件
- 自动检测元素名称行（有无元素名称均可）
- 同时支持 Direct（分数坐标）和 Cartesian（笛卡尔坐标）
- 纯 Python + NumPy 实现，无 pymatgen 依赖

POSCAR 文件格式说明:
    第1行: 注释行
    第2行: 缩放因子
    第3-5行: 晶格向量 (3x3)
    第6行: 元素名称（可选，若为数字行则跳过）
    第7行: 各元素原子数
    第8行: [Direct/Cartesian/kartesian/Direct]（坐标类型）
    第9行起: 原子坐标 (Nx3)
"""

import os
import re
import numpy as np


def expand_atomic_elements(elements, n_atoms):
    """
    Expand POSCAR species counts into one element label per atom.

    Parameters
    ----------
    elements : list of str
        Species labels in POSCAR order.
    n_atoms : dict
        Species counts keyed by element label.

    Returns
    -------
    list of str
        Element label for every atom in coordinate order.
    """
    atom_elements = []
    for elem in elements:
        if elem not in n_atoms:
            raise ValueError(f"元素 '{elem}' 不在 n_atoms 字典中")
        atom_elements.extend([elem] * int(n_atoms[elem]))
    return atom_elements


def _is_element_line(line):
    """
    判断某行是否为元素名称行。

    元素名称行由字母组成（如 "Si O"），而原子数行由纯数字组成（如 "2 4"）。
    通过正则匹配判断该行是否包含至少一个非数字 token。

    Parameters
    ----------
    line : str
        待判断的文本行

    Returns
    -------
    bool
        若为元素名称行返回 True，否则返回 False
    """
    tokens = line.split()
    # 空行不算元素行
    if len(tokens) == 0:
        return False
    # 如果所有 token 都是数字（含小数），则不是元素行
    for token in tokens:
        try:
            float(token)
        except ValueError:
            # 只要有一个 token 不是数字，就认为是元素行
            return True
    return False


def read_poscar(filepath):
    """
    读取 POSCAR 或 CONTCAR 文件。

    自动检测文件中是否包含元素名称行，并支持 Direct 和 Cartesian 两种坐标格式。
    返回的字典中同时包含分数坐标和笛卡尔坐标，方便后续使用。

    Parameters
    ----------
    filepath : str or os.PathLike
        POSCAR/CONTCAR 文件路径

    Returns
    -------
    dict
        包含以下键的字典:
        - 'comment' (str): 注释行内容
        - 'scale' (float): 缩放因子
        - 'lattice' (np.ndarray, shape=(3,3)): 晶格向量矩阵，单位 Angstrom
        - 'elements' (list of str): 元素符号列表
        - 'n_atoms' (dict): 各元素原子数，键为元素符号，值为原子数
        - 'total_atoms' (int): 总原子数
        - 'coords_frac' (np.ndarray, shape=(N,3)): 分数坐标
        - 'coords' (np.ndarray, shape=(N,3)): 笛卡尔坐标（Angstrom）
        - 'is_direct' (bool): 原始文件中坐标是否为 Direct 格式

    Raises
    ------
    FileNotFoundError
        文件不存在
    ValueError
        文件格式不正确或解析失败

    Examples
    --------
    >>> data = read_poscar("POSCAR")
    >>> print(data["comment"])
    >>> print(data["lattice"])  # 3x3 晶格矩阵
    >>> print(data["coords"])    # Nx3 笛卡尔坐标
    """
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"POSCAR 文件不存在: {filepath}")

    with open(filepath, "r") as f:
        lines = f.readlines()

    # 去除每行末尾的换行符和空白
    lines = [line.rstrip() for line in lines]

    # 至少需要 8 行（无 Selective Dynamics 和无空行的情况）
    if len(lines) < 8:
        raise ValueError(
            f"POSCAR 文件行数不足（至少需要8行），当前仅有 {len(lines)} 行: {filepath}"
        )

    # ---- 第1行: 注释 ----
    comment = lines[0].strip()

    # ---- 第2行: 缩放因子 ----
    try:
        scale = float(lines[1].strip())
    except ValueError:
        raise ValueError(
            f"无法解析缩放因子（第2行）: '{lines[1].strip()}'"
        )

    # ---- 第3-5行: 晶格向量 ----
    lattice = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        tokens = lines[2 + i].split()
        if len(tokens) < 3:
            raise ValueError(
                f"晶格向量第 {i+1} 行数据不足: '{lines[2 + i]}'"
            )
        try:
            lattice[i] = [float(t) for t in tokens[:3]]
        except ValueError:
            raise ValueError(
                f"无法解析晶格向量第 {i+1} 行: '{lines[2 + i]}'"
            )

    # 应用缩放因子
    lattice = lattice * scale

    # ---- 第6行: 判断是元素名称行还是原子数行 ----
    line_idx = 5  # 第6行（0-based index）

    if _is_element_line(lines[line_idx]):
        # 有元素名称行
        elements = lines[line_idx].split()
        line_idx += 1  # 移动到原子数行
    else:
        # 无元素名称行，使用占位符
        elements = None

    # ---- 原子数行 ----
    if line_idx >= len(lines):
        raise ValueError("文件在原子数行之前结束")

    n_atoms_tokens = lines[line_idx].split()
    n_atoms_list = []
    try:
        for t in n_atoms_tokens:
            n_atoms_list.append(int(t))
    except ValueError:
        raise ValueError(
            f"无法解析原子数行: '{lines[line_idx]}'"
        )

    total_atoms = sum(n_atoms_list)

    # 如果没有元素名称，生成占位元素名
    if elements is None:
        elements = [f"Elem{i+1}" for i in range(len(n_atoms_list))]

    if len(elements) != len(n_atoms_list):
        raise ValueError(
            f"元素名称数量 ({len(elements)}) 与原子数行数量 ({len(n_atoms_list)}) 不匹配"
        )

    # 构建 n_atoms 字典
    n_atoms = {}
    for elem, count in zip(elements, n_atoms_list):
        n_atoms[elem] = count
    atom_elements = expand_atomic_elements(elements, n_atoms)

    # ---- 可选的 Selective Dynamics 行 ----
    line_idx += 1
    if line_idx >= len(lines):
        raise ValueError("文件在坐标类型行之前结束")

    # 检查是否有 Selective Dynamics 行
    selective_line = lines[line_idx].strip().lower()
    if selective_line.startswith("selective"):
        line_idx += 1  # 跳过 Selective Dynamics 行

    # ---- 坐标类型行 ----
    if line_idx >= len(lines):
        raise ValueError("文件在坐标类型行之前结束")

    coord_line = lines[line_idx].strip().lower()
    if coord_line.startswith("d") or coord_line.startswith("direct"):
        is_direct = True
    elif coord_line.startswith("c") or coord_line.startswith("k") or \
         coord_line.startswith("cartesian") or coord_line.startswith("kartesian"):
        is_direct = False
    else:
        # 默认为 Direct
        is_direct = True

    # ---- 读取原子坐标 ----
    line_idx += 1
    coords_raw = []
    for i in range(total_atoms):
        if line_idx + i >= len(lines):
            raise ValueError(
                f"期望 {total_atoms} 个原子的坐标，但文件在第 {i+1} 个原子处结束"
            )
        tokens = lines[line_idx + i].split()
        if len(tokens) < 3:
            raise ValueError(
                f"原子坐标第 {i+1} 行数据不足: '{lines[line_idx + i]}'"
            )
        try:
            coords_raw.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])
        except ValueError:
            raise ValueError(
                f"无法解析原子坐标第 {i+1} 行: '{lines[line_idx + i]}'"
            )

    coords_raw = np.array(coords_raw, dtype=np.float64)

    # ---- 坐标转换 ----
    if is_direct:
        # Direct -> Cartesian: coords_cart = frac @ lattice
        coords_frac = coords_raw.copy()
        coords = coords_frac @ lattice
    else:
        # Cartesian -> Fractional: frac = cart @ inv(lattice)
        coords = coords_raw.copy()
        coords_frac = coords @ np.linalg.inv(lattice)

    return {
        "comment": comment,
        "scale": scale,
        "lattice": lattice,
        "elements": elements,
        "n_atoms": n_atoms,
        "atom_elements": atom_elements,
        "total_atoms": total_atoms,
        "coords_frac": coords_frac,
        "coords": coords,
        "is_direct": is_direct,
    }


def write_poscar(filepath, data, coord_type="direct"):
    """
    将结构数据写入 POSCAR 文件。

    Parameters
    ----------
    filepath : str or os.PathLike
        输出文件路径
    data : dict
        结构数据字典，应包含以下键:
        - 'comment' (str): 注释行（可选，默认 "POSCAR"）
        - 'lattice' (np.ndarray, shape=(3,3)): 晶格向量矩阵
        - 'elements' (list of str): 元素符号列表
        - 'n_atoms' (dict): 各元素原子数
        - 'coords_frac' (np.ndarray) 或 'coords' (np.ndarray): 坐标数组
        - 'is_direct' (bool): 原始坐标类型（可选）
    coord_type : str, optional
        写入时的坐标类型，'direct' 或 'cartesian'，默认 'direct'

    Raises
    ------
    ValueError
        输入数据不完整或格式不正确

    Examples
    --------
    >>> data = read_poscar("CONTCAR")
    >>> write_poscar("POSCAR_new", data, coord_type="direct")
    """
    filepath = str(filepath)

    # ---- 输入验证 ----
    required_keys = ["lattice", "elements", "n_atoms"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"写入 POSCAR 缺少必需字段: '{key}'")

    lattice = np.asarray(data["lattice"], dtype=np.float64)
    if lattice.shape != (3, 3):
        raise ValueError(f"晶格向量矩阵形状应为 (3,3)，当前为 {lattice.shape}")

    elements = data["elements"]
    n_atoms = data["n_atoms"]

    # 验证 elements 和 n_atoms 一致性
    n_atoms_list = []
    for elem in elements:
        if elem not in n_atoms:
            raise ValueError(
                f"元素 '{elem}' 不在 n_atoms 字典中"
            )
        n_atoms_list.append(n_atoms[elem])

    total_atoms = sum(n_atoms_list)
    comment = data.get("comment", "POSCAR")

    # ---- 确定坐标 ----
    coord_type_lower = coord_type.strip().lower()
    if coord_type_lower.startswith("d"):
        use_direct = True
    elif coord_type_lower.startswith("c") or coord_type_lower.startswith("k"):
        use_direct = False
    else:
        raise ValueError(
            f"不支持的坐标类型: '{coord_type}'，应为 'direct' 或 'cartesian'"
        )

    if use_direct:
        if "coords_frac" in data:
            coords = np.asarray(data["coords_frac"], dtype=np.float64)
        elif "coords" in data:
            # 从笛卡尔坐标转换为分数坐标
            coords_cart = np.asarray(data["coords"], dtype=np.float64)
            coords = coords_cart @ np.linalg.inv(lattice)
        else:
            raise ValueError("缺少坐标数据，需提供 'coords_frac' 或 'coords'")
    else:
        if "coords" in data:
            coords = np.asarray(data["coords"], dtype=np.float64)
        elif "coords_frac" in data:
            # 从分数坐标转换为笛卡尔坐标
            coords_frac = np.asarray(data["coords_frac"], dtype=np.float64)
            coords = coords_frac @ lattice
        else:
            raise ValueError("缺少坐标数据，需提供 'coords' 或 'coords_frac'")

    if coords.shape[0] != total_atoms:
        raise ValueError(
            f"坐标数量 ({coords.shape[0]}) 与总原子数 ({total_atoms}) 不匹配"
        )

    # ---- 写入文件 ----
    coord_label = "Direct" if use_direct else "Cartesian"

    with open(filepath, "w") as f:
        # 第1行: 注释
        f.write(f"{comment}\n")

        # 第2行: 缩放因子（默认1.0，因为 lattice 已经包含缩放）
        f.write("1.0\n")

        # 第3-5行: 晶格向量
        for i in range(3):
            f.write(f"  {lattice[i, 0]:18.10f} {lattice[i, 1]:18.10f} {lattice[i, 2]:18.10f}\n")

        # 第6行: 元素名称
        f.write("  " + "  ".join(elements) + "\n")

        # 第7行: 各元素原子数
        f.write("  " + "  ".join(str(n) for n in n_atoms_list) + "\n")

        # 第8行: 坐标类型
        f.write(f"{coord_label}\n")

        # 第9行起: 原子坐标
        for i in range(total_atoms):
            f.write(f"  {coords[i, 0]:18.14f} {coords[i, 1]:18.14f} {coords[i, 2]:18.14f}\n")
