"""
Bader ACF.dat 解析器
=====================

解析 Bader 电荷分析程序输出的 ACF.dat 文件。

ACF.dat 文件格式说明:
    前几行为标题信息:
    #   X  Y  Z  CHARGE  MIN DIST  ATOMIC VOL
    #   -----------------------------------------------------------------
    然后是每个原子的数据行:
        1  0.000  0.000  0.000  4.5678  0.1234  56.7890
        2  ...
    最后几行为汇总信息:
    #   NUMBER OF ELECTRONS IN VACUUM:  0.0000
    #   ------------------------------
    #   TOTAL CHARGE IN VOLUME:  XX.XXXX

功能:
- 解析每个原子的 Bader 电荷、最小距离、原子体积
- 提取总电荷和价电子数
- 返回结构化的字典数据
"""

import os
import re
import numpy as np


def read_acf(filepath):
    """
    读取 Bader 电荷分析的 ACF.dat 文件。

    Parameters
    ----------
    filepath : str or os.PathLike
        ACF.dat 文件路径

    Returns
    -------
    dict
        包含以下键的字典:
        - 'charges' (list of float): 每个原子的 Bader 电荷（价电子数）
        - 'volumes' (list of float): 每个原子的 Bader 体积 (Angstrom^3)
        - 'min_distances' (list of float): 每个原子到 Bader 表面的最小距离
        - 'coordinates' (np.ndarray, shape=(N,3)): 原子坐标 (Angstrom)
        - 'n_atoms' (int): 原子总数
        - 'valence_electrons' (float or None): 价电子总数（从参考电荷计算）
        - 'total_charge' (float or None): 体积内总电荷

    Raises
    ------
    FileNotFoundError
        文件不存在
    ValueError
        文件格式不正确或解析失败

    Examples
    --------
    >>> result = read_acf("ACF.dat")
    >>> print(f"原子数: {result['n_atoms']}")
    >>> print(f"平均 Bader 电荷: {np.mean(result['charges']):.4f}")
    >>> print(f"总电荷: {result['total_charge']}")
    """
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"ACF.dat 文件不存在: {filepath}")

    # ---- 初始化结果 ----
    charges = []
    volumes = []
    min_distances = []
    coordinates = []
    valence_electrons = None
    total_charge = None

    # ---- 正则表达式 ----
    # 数据行: 以数字开头的行，包含坐标、电荷、最小距离、体积
    # 格式: "   1   0.000000   0.000000   0.000000   4.5678   0.1234   56.7890"
    re_data_line = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"
        r"\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"
    )
    # 参考电荷行（部分 ACF.dat 包含）
    re_reference = re.compile(
        r"REFERENCE\s+CHARGE\s*:\s*([+-]?\d+\.\d+)",
        re.IGNORECASE
    )
    # 真空中的电子数
    re_vacuum = re.compile(
        r"NUMBER\s+OF\s+ELECTRONS\s+IN\s+VACUUM\s*:\s*([+-]?\d+\.\d+)",
        re.IGNORECASE
    )
    # 体积内总电荷
    re_total = re.compile(
        r"TOTAL\s+CHARGE\s+IN\s+VOLUME\s*:\s*([+-]?\d+\.\d+)",
        re.IGNORECASE
    )

    # ---- 逐行读取 ----
    with open(filepath, "r") as f:
        for line in f:
            # 尝试匹配数据行
            m = re_data_line.match(line)
            if m:
                # 提取坐标
                x = float(m.group(2))
                y = float(m.group(3))
                z = float(m.group(4))
                coordinates.append([x, y, z])

                # 提取电荷
                charge = float(m.group(5))
                charges.append(charge)

                # 提取最小距离
                min_dist = float(m.group(6))
                min_distances.append(min_dist)

                # 提取体积
                vol = float(m.group(7))
                volumes.append(vol)

            # 尝试匹配参考电荷
            m = re_reference.search(line)
            if m:
                valence_electrons = float(m.group(1))

            # 尝试匹配真空中的电子数（可用于计算总电荷）
            m = re_vacuum.search(line)
            if m:
                vacuum_electrons = float(m.group(1))

            # 尝试匹配体积内总电荷
            m = re_total.search(line)
            if m:
                total_charge = float(m.group(1))

    # ---- 验证 ----
    n_atoms = len(charges)
    if n_atoms == 0:
        raise ValueError(
            f"ACF.dat 文件中未找到有效的原子数据行: {filepath}"
        )

    # 验证各列表长度一致
    if len(volumes) != n_atoms:
        raise ValueError(
            f"体积数据数量 ({len(volumes)}) 与电荷数据数量 ({n_atoms}) 不匹配"
        )
    if len(min_distances) != n_atoms:
        raise ValueError(
            f"最小距离数据数量 ({len(min_distances)}) 与电荷数据数量 ({n_atoms}) 不匹配"
        )
    if len(coordinates) != n_atoms:
        raise ValueError(
            f"坐标数据数量 ({len(coordinates)}) 与电荷数据数量 ({n_atoms}) 不匹配"
        )

    # 如果没有从文件中直接读取到价电子数，尝试从参考电荷推断
    if valence_electrons is None and total_charge is not None:
        # 某些版本的 ACF.dat 不直接给出价电子数
        # 但可以从参考电荷和真空电荷计算
        pass

    return {
        "charges": charges,
        "volumes": volumes,
        "min_distances": min_distances,
        "coordinates": np.array(coordinates, dtype=np.float64),
        "n_atoms": n_atoms,
        "valence_electrons": valence_electrons,
        "total_charge": total_charge,
    }
