"""
CHGCAR 解析器
=============

解析 VASP CHGCAR (电荷密度) 文件。

功能:
- 读取 CHGCAR 文件中的结构信息和电荷密度数据
- 处理 VASP 的列优先（column-major）数据布局
- 支持读取总电荷密度和增强电荷密度（augmentation charges）
- 可选择跳过增强电荷密度以节省内存

CHGCAR 文件格式说明:
    前半部分与 POSCAR 格式相同（注释、缩放因子、晶格、元素、原子数、坐标）
    紧接着一行: 空行
    然后一行: NGX NGY NGZ（网格尺寸）
    然后是电荷密度数据，按列优先顺序排列（每行5个数值）
    可选的增强电荷密度部分以空行分隔

注意: VASP 使用列优先（Fortran 风格）存储网格数据，即第一个索引变化最快。
      读取后需要转置为行优先（C 风格）的 NumPy 数组。
"""

import os
import numpy as np

# 复用 POSCAR 解析器中的结构读取逻辑
from matkit.parsers.poscar import read_poscar


def _read_grid_header(line):
    """
    从文本行解析网格尺寸。

    Parameters
    ----------
    line : str
        包含 NGX NGY NGZ 的文本行

    Returns
    -------
    tuple of int
        (NGX, NGY, NGZ) 网格尺寸

    Raises
    ------
    ValueError
        无法解析网格尺寸
    """
    tokens = line.split()
    if len(tokens) < 3:
        raise ValueError(f"网格尺寸行数据不足: '{line}'")
    try:
        ngx = int(tokens[0])
        ngy = int(tokens[1])
        ngz = int(tokens[2])
    except ValueError:
        raise ValueError(f"无法解析网格尺寸: '{line}'")
    return ngx, ngy, ngz


def _read_charge_data(f, total_points):
    """
    从文件对象中读取电荷密度数据。

    VASP 将电荷密度数据按每行5个数值写入，总共 total_points 个数据点。
    数据按列优先顺序排列。

    Parameters
    ----------
    f : file object
        已打开的文件对象
    total_points : int
        需要读取的数据点总数

    Returns
    -------
    np.ndarray
        一维数组，包含按列优先顺序排列的电荷密度数据
    """
    data = []
    remaining = total_points

    while remaining > 0:
        line = f.readline()
        if not line:
            # 文件结束但数据未读完
            raise ValueError(
                f"文件在读取电荷密度数据时意外结束，"
                f"已读取 {len(data)}/{total_points} 个数据点"
            )

        line = line.strip()
        if not line:
            continue  # 跳过空行

        tokens = line.split()
        for token in tokens:
            try:
                val = float(token)
                data.append(val)
                remaining -= 1
                if remaining <= 0:
                    break
            except ValueError:
                # 跳过非数字 token（理论上不应出现）
                continue

    return np.array(data, dtype=np.float64)


def read_chgcar(filepath, read_augmentation=True):
    """
    读取 VASP CHGCAR 文件。

    文件前半部分为 POSCAR 格式的结构信息，后半部分为电荷密度网格数据。
    VASP 使用列优先（column-major/Fortran 风格）存储网格数据，
    本函数会自动将其转置为行优先（row-major/C 风格）的 NumPy 数组。

    Parameters
    ----------
    filepath : str or os.PathLike
        CHGCAR 文件路径
    read_augmentation : bool, optional
        是否读取增强电荷密度（PAW 部分的 augmentation charges）。
        对于大体系，跳过此部分可显著减少内存使用。默认 True。

    Returns
    -------
    dict
        包含以下键的字典:
        - 'structure' (dict): 结构信息，格式与 read_poscar() 返回值相同
        - 'dim' (tuple of int): 网格尺寸 (NGX, NGY, NGZ)
        - 'total_charge' (np.ndarray, shape=(NGX, NGY, NGZ)):
          总电荷密度三维数组（已转为行优先）
        - 'augmentation' (np.ndarray or None):
          增强电荷密度三维数组，若无此数据则为 None

    Raises
    ------
    FileNotFoundError
        文件不存在
    ValueError
        文件格式不正确或数据不完整

    Notes
    -----
    - CHGCAR 文件可能非常大（数百MB到数GB），读取时注意内存
    - 列优先转行优先: VASP 写入顺序为 x 变化最快，即 data[x, y, z] 按
      x 内层循环写入。读取后 reshape 并转轴:
      array_3d = data_1d.reshape(NGX, NGY, NGZ, order='F')

    Examples
    --------
    >>> result = read_chgcar("CHGCAR")
    >>> print(f"网格尺寸: {result['dim']}")
    >>> print(f"总电荷密度形状: {result['total_charge'].shape}")
    >>> # 跳过增强电荷密度以节省内存
    >>> result = read_chgcar("CHGCAR", read_augmentation=False)
    """
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CHGCAR 文件不存在: {filepath}")

    # ---- 读取结构信息（使用 POSCAR 解析器）----
    structure = read_poscar(filepath)

    # ---- 定位电荷密度数据起始位置 ----
    # 需要重新打开文件，跳过结构部分
    with open(filepath, "r") as f:
        lines = f.readlines()

    # 计算结构部分占用的行数
    # 注释(1) + 缩放(1) + 晶格(3) + 元素名(0或1) + 原子数(1) +
    # 可选 Selective Dynamics(0或1) + 坐标类型(1) + 坐标(total_atoms)
    total_atoms = structure["total_atoms"]

    # 重新逐行扫描，确定结构部分的行数
    line_idx = 0

    # 第1行: 注释
    line_idx += 1
    # 第2行: 缩放因子
    line_idx += 1
    # 第3-5行: 晶格向量
    line_idx += 3

    # 判断元素名称行
    from matkit.parsers.poscar import _is_element_line
    if line_idx < len(lines) and _is_element_line(lines[line_idx]):
        line_idx += 1

    # 原子数行
    line_idx += 1

    # Selective Dynamics 行
    if line_idx < len(lines):
        test_line = lines[line_idx].strip().lower()
        if test_line.startswith("selective"):
            line_idx += 1

    # 坐标类型行
    line_idx += 1

    # 原子坐标行
    line_idx += total_atoms

    # 跳过可能的空行，找到网格尺寸行
    while line_idx < len(lines) and lines[line_idx].strip() == "":
        line_idx += 1

    if line_idx >= len(lines):
        raise ValueError("CHGCAR 文件中未找到网格尺寸行")

    # ---- 读取网格尺寸 ----
    dim = _read_grid_header(lines[line_idx])
    ngx, ngy, ngz = dim
    line_idx += 1

    total_grid_points = ngx * ngy * ngz

    # ---- 读取电荷密度数据 ----
    # 从文件中读取，使用高效方式
    with open(filepath, "r") as f:
        # 跳过到数据起始位置
        for _ in range(line_idx):
            f.readline()

        # 读取总电荷密度
        charge_data = _read_charge_data(f, total_grid_points)

    # 将一维数据重塑为三维数组（列优先 -> 行优先）
    # VASP 使用 Fortran 顺序（列优先），第一个索引变化最快
    total_charge = charge_data.reshape((ngx, ngy, ngz), order='F')

    # ---- 读取增强电荷密度（可选）----
    augmentation = None
    if read_augmentation:
        # 增强电荷密度紧跟在总电荷密度之后
        # 两者之间可能有空行
        with open(filepath, "r") as f:
            # 跳过到总电荷密度数据之后
            for _ in range(line_idx):
                f.readline()

            # 跳过总电荷密度数据
            remaining = total_grid_points
            while remaining > 0:
                line = f.readline()
                if not line:
                    break
                tokens = line.split()
                remaining -= len(tokens)

            # 跳过空行
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    # 这可能是增强电荷密度的网格尺寸行
                    # 检查是否是网格尺寸
                    tokens = line.strip().split()
                    if len(tokens) == 3:
                        try:
                            aug_ngx = int(tokens[0])
                            aug_ngy = int(tokens[1])
                            aug_ngz = int(tokens[2])
                            # 验证网格尺寸是否与主网格一致
                            if (aug_ngx, aug_ngy, aug_ngz) == (ngx, ngy, ngz):
                                aug_total = aug_ngx * aug_ngy * aug_ngz
                                aug_data = _read_charge_data(f, aug_total)
                                augmentation = aug_data.reshape(
                                    (aug_ngx, aug_ngy, aug_ngz), order='F'
                                )
                        except ValueError:
                            pass
                    break
                # 继续读取空行

    return {
        "structure": structure,
        "dim": dim,
        "total_charge": total_charge,
        "augmentation": augmentation,
    }
