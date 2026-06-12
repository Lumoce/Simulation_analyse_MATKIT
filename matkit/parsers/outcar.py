"""
OUTCAR 解析器
=============

解析 VASP OUTCAR 输出文件，提取能量、力、磁矩等关键信息。

功能:
- 提取最终总能量 (TOTEN)
- 提取每一步离子步的能量
- 提取费米能级、电子数、总磁化强度
- 提取力 (TOTAL-FORCE)
- 提取体积和压力
- 判断是否收敛
- 逐行读取，高效处理大文件

注意: OUTCAR 文件通常较大（数百MB到数GB），本解析器采用逐行读取策略，
避免一次性加载整个文件到内存。
"""

import os
import re
import numpy as np


def read_outcar(filepath):
    """
    读取 VASP OUTCAR 文件并提取关键信息。

    逐行扫描文件，使用正则表达式匹配关键数据块。
    对于大文件（>1GB），此方法内存效率较高。

    Parameters
    ----------
    filepath : str or os.PathLike
        OUTCAR 文件路径

    Returns
    -------
    dict
        包含以下键的字典:
        - 'final_energy' (float or None): 最终总能量 (eV)，对应最后一行 free energy TOTEN
        - 'energies_per_ionic_step' (list of float): 每个离子步的能量 (eV)
        - 'efermi' (float or None): 费米能级 (eV)
        - 'nelect' (float or None): 总价电子数
        - 'total_magnetization' (float or None): 总磁化强度 (mu_B)
        - 'forces' (np.ndarray or None, shape=(N,3)): 最后一步的力 (eV/Angstrom)
        - 'is_converged' (bool): 是否达到收敛精度
        - 'n_ionic_steps' (int): 离子步总数
        - 'volume' (float or None): 晶胞体积 (Angstrom^3)
        - 'pressures' (list of float): 各离子步的压力 (kB)

    Raises
    ------
    FileNotFoundError
        文件不存在
    ValueError
        文件格式不正确

    Examples
    --------
    >>> result = read_outcar("OUTCAR")
    >>> print(f"最终能量: {result['final_energy']:.6f} eV")
    >>> print(f"收敛: {result['is_converged']}")
    >>> print(f"离子步数: {result['n_ionic_steps']}")
    """
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"OUTCAR 文件不存在: {filepath}")

    # ---- 初始化结果字典 ----
    result = {
        "final_energy": None,
        "energies_per_ionic_step": [],
        "efermi": None,
        "nelect": None,
        "total_magnetization": None,
        "forces": None,
        "is_converged": False,
        "n_ionic_steps": 0,
        "volume": None,
        "pressures": [],
    }

    # ---- 正则表达式预编译 ----
    # 自由能 TOTEN: " free  energy   TOTEN  =        -12.345678 eV"
    re_toten = re.compile(
        r"\s*free\s+energy\s+TOTEN\s*=\s*([+-]?\d+\.\d+)\s*eV"
    )
    # 费米能级: " E-fermi :  5.1234"
    re_efermi = re.compile(
        r"\s*E-fermi\s*:\s*([+-]?\d+\.\d+)"
    )
    # 总电子数: " NELECT  =  24.0000"
    re_nelect = re.compile(
        r"\s*NELECT\s*=\s*([+-]?\d+\.\d+)"
    )
    # 总磁化强度: " total magnetization"
    re_mag = re.compile(
        r"\s*magnetization\s*\(x\)\s*=\s*([+-]?\d+\.\d+)"
    )
    # 体积: " volume of cell :  123.45"
    re_volume = re.compile(
        r"\s*volume of cell\s*:\s*([+-]?\d+\.\d+)"
    )
    # 压力: " external pressure"
    re_pressure = re.compile(
        r"\s*external pressure\s*=\s*([+-]?\d+\.\d+)\s*kB"
    )
    # 收敛: "reached required accuracy"
    re_converged = re.compile(
        r"reached required accuracy"
    )
    # 力块的开始: " TOTAL-FORCE (eV/Angst)"
    re_force_start = re.compile(
        r"\s*TOTAL-FORCE\s*\(eV/Angst\)"
    )
    # 力块的结束: "---" 行
    re_force_end = re.compile(
        r"^\s*-{5,}"
    )

    # ---- 逐行读取 ----
    in_force_block = False
    force_lines = []
    last_energy = None
    force_header_seen = False  # 标记是否已跳过力块的标题分隔线

    with open(filepath, "r") as f:
        for line in f:
            # ---- 能量 ----
            m = re_toten.search(line)
            if m:
                energy = float(m.group(1))
                result["energies_per_ionic_step"].append(energy)
                last_energy = energy
                result["n_ionic_steps"] = len(result["energies_per_ionic_step"])

            # ---- 费米能级 ----
            m = re_efermi.search(line)
            if m:
                result["efermi"] = float(m.group(1))

            # ---- 总电子数 ----
            m = re_nelect.search(line)
            if m:
                result["nelect"] = float(m.group(1))

            # ---- 总磁化强度 ----
            m = re_mag.search(line)
            if m:
                result["total_magnetization"] = float(m.group(1))

            # ---- 体积 ----
            m = re_volume.search(line)
            if m:
                result["volume"] = float(m.group(1))

            # ---- 压力 ----
            m = re_pressure.search(line)
            if m:
                result["pressures"].append(float(m.group(1)))

            # ---- 收敛判断 ----
            if re_converged.search(line):
                result["is_converged"] = True

            # ---- 力块解析 ----
            if re_force_start.search(line):
                in_force_block = True
                force_lines = []
                force_header_seen = False  # 重置标志
                continue

            if in_force_block:
                # 检查是否为分隔线
                if re_force_end.search(line):
                    if not force_header_seen:
                        # 第一个分隔线是标题和数据的分界，跳过
                        force_header_seen = True
                        continue
                    else:
                        # 第二个分隔线表示力块结束
                        in_force_block = False
                        # 解析收集到的力行
                        if force_lines:
                            try:
                                forces = []
                                for fl in force_lines:
                                    tokens = fl.split()
                                    # 力行格式:
                                    # 格式1: "     1       0.123456    0.234567    0.345678"
                                    # 格式2: "  0.00000  0.00000  0.00000   0.010000   0.020000   0.030000"
                                    if len(tokens) >= 4:
                                        # 判断第一个 token 是否为整数索引
                                        try:
                                            int(tokens[0])
                                            # 格式1: 索引 fx fy fz
                                            forces.append([
                                                float(tokens[1]),
                                                float(tokens[2]),
                                                float(tokens[3]),
                                            ])
                                        except ValueError:
                                            # 格式2: x y z fx fy fz
                                            forces.append([
                                                float(tokens[3]),
                                                float(tokens[4]),
                                                float(tokens[5]),
                                            ])
                                if forces:
                                    result["forces"] = np.array(forces, dtype=np.float64)
                            except (ValueError, IndexError):
                                pass  # 力行解析失败，跳过
                        force_lines = []
                        continue

                # 跳过力块中的标题行和空行
                stripped = line.strip()
                if stripped:
                    # 尝试解析为力数据行
                    tokens = stripped.split()
                    if len(tokens) >= 4:
                        try:
                            float(tokens[0])  # 测试第一个 token 是否为数字
                            force_lines.append(stripped)
                        except ValueError:
                            pass  # 非数据行，跳过

    # ---- 设置最终能量 ----
    result["final_energy"] = last_energy

    return result
