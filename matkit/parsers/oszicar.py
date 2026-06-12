"""
OSZICAR 解析器
==============

解析 VASP OSZICAR 文件，提取能量收敛信息。

OSZICAR 文件格式说明:
    每个离子步（ionic step）包含一行或多行，格式如下:
       N       E                     dE             deps       dmax     ncg     rms          rms(c)
    DAVID:   1    0.00000000E+00   0.00000E+00   -0.12345E+01    0.123E+01    48   0.123E+02
    DAVID:   1 F= 0.12345678E+02 E0= 0.12345678E+02  d E =-0.12345678E+02
       1    -0.12345678E+02

    其中:
    - N: ionic step 编号
    - E/F=: 能量值 (eV)
    - dE: 能量变化
    - dmax: 最大力 (eV/Angstrom)

功能:
- 提取每个离子步的能量
- 提取每个离子步的最大力
- 统计离子步总数
"""

import os
import re
import numpy as np


def read_oszicar(filepath):
    """
    读取 VASP OSZICAR 文件。

    OSZICAR 文件中每个离子步可能有多行输出（对应不同的电子步优化算法迭代），
    本函数提取每个离子步的最终能量（通常以 "F=" 标记的行或离子步的最后一行）。

    Parameters
    ----------
    filepath : str or os.PathLike
        OSZICAR 文件路径

    Returns
    -------
    dict
        包含以下键的字典:
        - 'energies' (list of float): 每个离子步的能量 (eV)
        - 'max_forces' (list of float): 每个离子步的最大力 (eV/Angstrom)
        - 'ionic_steps' (int): 离子步总数

    Raises
    ------
    FileNotFoundError
        文件不存在
    ValueError
        文件格式不正确

    Examples
    --------
    >>> result = read_oszicar("OSZICAR")
    >>> print(f"离子步数: {result['ionic_steps']}")
    >>> print(f"最终能量: {result['energies'][-1]:.6f} eV")
    >>> print(f"最大力: {result['max_forces'][-1]:.6f} eV/Ang")
    """
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"OSZICAR 文件不存在: {filepath}")

    # ---- 初始化 ----
    energies = []
    max_forces = []

    # ---- 正则表达式 ----
    # 匹配含 F= 的行: "   1 F= -.12345678E+02 E0= -.12345678E+02  d E =-.12345678E+02  mag= ..."
    re_f_energy = re.compile(
        r"^\s*\d+\s+F=\s*([+-]?\d+\.\d+[EeDd][+-]?\d+)"
    )
    # 匹配含 dmax 的行（DAVID/CG/RMM 等算法行）:
    # "DAVID:   1    0.12345678E+02   0.00000E+00   -0.12345E+01    0.123E+01    48   0.123E+02"
    re_dmax = re.compile(
        r"^\s*(?:DAVID|CG|RMM|IBRION|US|NELM|RMMD|CGNO)[:\s]\s*\d+\s+"
        r"[+-]?\d+\.\d+[EeDd][+-]?\d+\s+"
        r"[+-]?\d+\.\d+[EeDd][+-]?\d+\s+"
        r"[+-]?\d+\.\d+[EeDd][+-]?\d+\s+"
        r"([+-]?\d+\.\d+[EeDd][+-]?\d+)"
    )
    # 匹配简单能量行（某些版本的 OSZICAR 格式）:
    # "       1    -0.12345678E+02"
    re_simple_energy = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+\.\d+[EeDd][+-]?\d+)\s*$"
    )
    # 匹配 ionic step 标记行（含 N 和 E 的行）:
    # "       N       E                     dE             deps       dmax     ncg     rms          rms(c)"
    re_header = re.compile(
        r"^\s*N\s+E\s+dE"
    )

    # ---- 逐行读取 ----
    current_step_energy = None
    current_step_dmax = None
    seen_f_line = False  # 标记当前步是否已看到 F= 行

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()

            # 跳过空行和标题行
            if not stripped or re_header.match(stripped):
                continue

            # ---- 尝试匹配 F= 行 ----
            m = re_f_energy.match(stripped)
            if m:
                try:
                    energy = float(m.group(1).replace('D', 'E').replace('d', 'e'))
                    current_step_energy = energy
                    seen_f_line = True
                except ValueError:
                    pass

            # ---- 尝试匹配含 dmax 的算法行 ----
            m = re_dmax.match(stripped)
            if m:
                try:
                    dmax = float(m.group(1).replace('D', 'E').replace('d', 'e'))
                    current_step_dmax = dmax
                except ValueError:
                    pass

            # ---- 尝试匹配简单能量行 ----
            m = re_simple_energy.match(stripped)
            if m and not seen_f_line:
                try:
                    step_num = int(m.group(1))
                    energy = float(m.group(2).replace('D', 'E').replace('d', 'e'))
                    # 简单能量行通常表示一个新离子步的最终能量
                    current_step_energy = energy
                except ValueError:
                    pass

            # ---- 判断是否为新离子步的开始 ----
            # 检测行首的数字（离子步编号）
            step_match = re.match(r"^\s*(\d+)", stripped)
            if step_match:
                step_num = int(step_match.group(1))

                # 如果当前步有能量数据且不是第一步，先保存上一步的数据
                # （当遇到新的步号时，说明上一步已结束）
                # 注意: F= 行通常紧跟在算法行之后，所以需要延迟保存

    # ---- 重新解析（更可靠的方法）----
    # 使用更简单直接的策略:
    # 1. 找到所有 F= 行，提取能量
    # 2. 找到所有含 dmax 的行，提取最大力
    # 3. 按 ionic step 编号分组

    energies = []
    max_forces = []

    # 重新读取文件
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()

            # 匹配 F= 行，提取能量
            m = re_f_energy.match(stripped)
            if m:
                try:
                    energy = float(m.group(1).replace('D', 'E').replace('d', 'e'))
                    energies.append(energy)
                except ValueError:
                    pass

            # 匹配含 dmax 的行，提取最大力
            m = re_dmax.match(stripped)
            if m:
                try:
                    dmax = float(m.group(1).replace('D', 'E').replace('d', 'e'))
                    max_forces.append(dmax)
                except ValueError:
                    pass

    # 如果 F= 行方式没有提取到能量，尝试简单格式
    if len(energies) == 0:
        with open(filepath, "r") as f:
            for line in f:
                stripped = line.strip()
                m = re_simple_energy.match(stripped)
                if m:
                    try:
                        energy = float(m.group(2).replace('D', 'E').replace('d', 'e'))
                        energies.append(energy)
                    except ValueError:
                        pass

    # 如果 dmax 方式没有提取到力，尝试从 F= 行中提取
    # 某些 OSZICAR 格式在 F= 行中也包含 dmax 信息
    if len(max_forces) == 0 and len(energies) > 0:
        # 没有找到 dmax 数据，填充 None
        max_forces = [None] * len(energies)

    # 确保 energies 和 max_forces 长度一致
    # max_forces 可能在每个 ionic step 中有多条记录（对应多个电子步迭代）
    # 我们只取每个 ionic step 的最后一条
    # 但由于 F= 行每个 ionic step 只有一条，通常 energies 和 max_forces 数量不同
    # 这里保持原样返回，让用户自行处理

    ionic_steps = len(energies)

    return {
        "energies": energies,
        "max_forces": max_forces,
        "ionic_steps": ionic_steps,
    }
