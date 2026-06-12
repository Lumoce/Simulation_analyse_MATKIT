"""
Bader电荷分析模块 / Bader Charge Analysis Module
==================================================

提供Bader电荷分析结果的解析、比较和报告导出功能。
Bader分析基于Henkelman课题组开发的算法，将空间中的电荷密度
分配给最近的原子，从而获得每个原子的积分电荷。
Provides parsing, comparison, and report export for Bader charge analysis results.
Bader analysis is based on the algorithm developed by Henkelman's group,
which partitions the charge density in space to the nearest atom,
yielding the integrated charge for each atom.

依赖 / Dependencies:
    - numpy (数据处理)
    - csv (报告导出)

使用示例 / Usage Example:
    >>> from matkit.charge import parse_bader_results, export_bader_report
    >>> results = parse_bader_results("ACF.dat", "POSCAR")
    >>> export_bader_report(results, "bader_report.csv")
"""

import csv
import os
import warnings
from collections import defaultdict

import numpy as np


def parse_bader_results(acf_path, poscar_path=None):
    """
    解析Bader分析结果 / Parse Bader charge analysis results.

    读取ACF.dat文件（Bader程序的输出），提取每个原子的Bader电荷、
    电荷转移量和体积信息。可选地与POSCAR文件匹配以获取元素信息。
    Reads the ACF.dat file (output of the Bader program), extracting
    per-atom Bader charges, charge transfer, and volume information.
    Optionally matches with POSCAR to get element information.

    参数 / Parameters:
        acf_path (str): ACF.dat文件路径。
            Path to the ACF.dat file.
        poscar_path (str, optional): POSCAR文件路径，用于获取元素符号。
            Path to POSCAR file for element symbols. Default is None.

    返回 / Returns:
        dict: 包含以下键的字典:
            - 'charges' (list of float): 每个原子的Bader电荷（e）
              Per-atom Bader charges (e)
            - 'charge_transfer' (list of float): 每个原子的电荷转移量（e），
              正值表示得电子，负值表示失电子
              Per-atom charge transfer (e), positive=gain, negative=loss
            - 'volumes' (list of float): 每个原子的Bader体积（Å³）
              Per-atom Bader volumes (Å³)
            - 'min_dist' (list of float): 每个原子到Bader表面的最短距离（Å）
              Per-atom minimum distance to Bader surface (Å)
            - 'elements' (list of str or None): 每个原子的元素符号（如果提供了POSCAR）
              Per-atom element symbols (if POSCAR provided)
            - 'n_atoms' (int): 原子总数
              Total number of atoms

    异常 / Raises:
        FileNotFoundError: 如果ACF.dat文件不存在。
            If ACF.dat file does not exist.
        ValueError: 如果ACF.dat文件格式不正确。
            If ACF.dat file format is incorrect.

    示例 / Example:
        >>> results = parse_bader_results("ACF.dat", "POSCAR")
        >>> print(f"Total atoms: {results['n_atoms']}")
        >>> for i, (e, ct) in enumerate(zip(results['elements'], results['charge_transfer'])):
        ...     print(f"  {e}: {ct:+.4f} e")
    """
    if not os.path.isfile(acf_path):
        raise FileNotFoundError(
            f"ACF.dat文件不存在: '{acf_path}' / "
            f"ACF.dat file not found: '{acf_path}'"
        )

    # 解析ACF.dat文件 / Parse ACF.dat file
    charges = []
    charge_transfer = []
    volumes = []
    min_dist = []

    try:
        with open(acf_path, "r") as f:
            lines = f.readlines()
    except IOError as e:
        raise IOError(
            f"无法读取ACF.dat文件: {e} / "
            f"Failed to read ACF.dat file: {e}"
        )

    # ACF.dat格式:
    # 第1行: 标题 / Header
    # 第2行: 列标题 / Column headers
    # 第3行及之后: 数据行 / Data lines
    #   X Y Z CHARGE MIN DIST VOLUME
    #
    # 最后两行是统计信息 / Last two lines are statistics
    #   #  NUMBER  X  Y  Z  CHARGE  MIN DIST  VOLUME
    #   -------------------------- ...

    # 跳过前两行标题 / Skip first two header lines
    data_start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and "VOLUME" in stripped.upper():
            data_start = i + 1
            break

    if data_start == -1:
        # 尝试另一种格式: 直接从第3行开始是数据
        # Try alternate format: data starts from line 3
        data_start = 2

    # 解析数据行 / Parse data lines
    for i in range(data_start, len(lines)):
        line = lines[i].strip()

        # 跳过空行和分隔线 / Skip empty lines and separator lines
        if not line or line.startswith("---") or line.startswith("==="):
            continue

        # 跳过以#开头的行 / Skip lines starting with #
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 6:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            charge = float(parts[3])
            min_d = float(parts[4])
            vol = float(parts[5])

            charges.append(charge)
            min_dist.append(min_d)
            volumes.append(vol)
        except (ValueError, IndexError):
            # 可能是统计行或格式异常，跳过 / May be statistics line or format issue, skip
            continue

    if len(charges) == 0:
        raise ValueError(
            f"ACF.dat文件 '{acf_path}' 中未找到有效的电荷数据。"
            f"请检查文件格式。 / "
            f"No valid charge data found in ACF.dat file '{acf_path}'. "
            f"Please check the file format."
        )

    # 计算电荷转移量 / Calculate charge transfer
    # 默认使用中性原子参考值（需要用户提供价电子数来精确计算）
    # 这里先存储原始电荷，电荷转移需要参考值
    # Default: use neutral atom reference (user needs to provide valence for precise calculation)
    # Store raw charges first; charge transfer needs reference values
    charge_transfer = [0.0] * len(charges)

    # 尝试从POSCAR获取元素信息 / Try to get element info from POSCAR
    elements = [None] * len(charges)
    valence_electrons = {}

    if poscar_path is not None:
        if not os.path.isfile(poscar_path):
            warnings.warn(
                f"POSCAR文件不存在: '{poscar_path}'，将不匹配元素信息。 / "
                f"POSCAR file not found: '{poscar_path}', "
                f"element matching will be skipped."
            )
        else:
            try:
                elements, valence_electrons = _parse_poscar_for_bader(poscar_path, len(charges))
            except Exception as e:
                warnings.warn(
                    f"解析POSCAR文件失败: {e}，将不匹配元素信息。 / "
                    f"Failed to parse POSCAR: {e}, element matching will be skipped."
                )

            # 如果有价电子信息，计算电荷转移
            # If valence electron info available, calculate charge transfer
            if valence_electrons:
                for i, elem in enumerate(elements):
                    if elem is not None and elem in valence_electrons:
                        charge_transfer[i] = valence_electrons[elem] - charges[i]

    return {
        "charges": charges,
        "charge_transfer": charge_transfer,
        "volumes": volumes,
        "min_dist": min_dist,
        "elements": elements,
        "n_atoms": len(charges),
    }


def _parse_poscar_for_bader(poscar_path, n_atoms):
    """
    从POSCAR文件中提取元素信息和原子数量 / Extract element info from POSCAR.

    这是内部辅助函数，用于解析POSCAR文件获取元素符号列表。
    This is an internal helper function to parse POSCAR for element symbols.

    参数 / Parameters:
        poscar_path (str): POSCAR文件路径。
            Path to POSCAR file.
        n_atoms (int): 原子总数（用于验证）。
            Total number of atoms (for validation).

    返回 / Returns:
        tuple: (elements_list, valence_dict)
            - elements_list: 元素符号列表
            - valence_dict: 元素到价电子数的映射（使用常见VASP POTCAR参考值）
    """
    # VASP常见价电子配置（POTCAR参考值）
    # Common VASP valence electron configurations (POTCAR reference values)
    # 这些是VASP推荐值，可能与实际使用的POTCAR不同
    # These are VASP recommended values; may differ from actual POTCAR used
    default_valence = {
        "H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5,
        "O": 6, "F": 7, "Ne": 8, "Na": 1, "Mg": 2, "Al": 3, "Si": 4,
        "P": 5, "S": 6, "Cl": 7, "Ar": 8, "K": 1, "Ca": 2, "Sc": 3,
        "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
        "Cu": 11, "Zn": 12, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7,
        "Kr": 8, "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6,
        "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 12, "In": 3,
        "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Xe": 8, "Cs": 1, "Ba": 2,
        "La": 11, "Ce": 12, "Pr": 12, "Nd": 12, "Pm": 12, "Sm": 12,
        "Eu": 12, "Gd": 12, "Tb": 12, "Dy": 12, "Ho": 12, "Er": 12,
        "Tm": 12, "Yb": 12, "Lu": 12, "Hf": 4, "Ta": 5, "W": 6, "Re": 7,
        "Os": 8, "Ir": 9, "Pt": 10, "Au": 11, "Hg": 12, "Tl": 3, "Pb": 4,
        "Bi": 5, "Po": 6, "At": 7, "Rn": 8,
    }

    elements = []
    with open(poscar_path, "r") as f:
        lines = f.readlines()

    # POSCAR格式:
    # 第1行: 注释 / Comment
    # 第2行: 缩放因子 / Scaling factor
    # 第3-5行: 晶格矢量 / Lattice vectors
    # 第6行: 元素符号 / Element symbols
    # 第7行: 每种元素的原子数 / Atom counts per element
    element_line = lines[5].strip().split()
    count_line = lines[6].strip().split()

    try:
        counts = [int(c) for c in count_line]
    except ValueError:
        raise ValueError(
            f"无法解析POSCAR中的原子数量行: '{lines[6].strip()}' / "
            f"Cannot parse atom count line in POSCAR: '{lines[6].strip()}'"
        )

    total = sum(counts)
    if total != n_atoms:
        warnings.warn(
            f"POSCAR中的原子数 ({total}) 与ACF.dat中的原子数 ({n_atoms}) 不匹配。"
            f"结果可能不正确。 / "
            f"Atom count in POSCAR ({total}) does not match ACF.dat ({n_atoms}). "
            f"Results may be incorrect."
        )

    for elem, count in zip(element_line, counts):
        elements.extend([elem] * count)

    return elements, default_valence


def compare_bader_charges(acf_clean, acf_adsorbed, atom_indices):
    """
    比较清洁表面和吸附表面的Bader电荷 / Compare Bader charges between clean and adsorbed surfaces.

    对于指定的原子索引，计算吸附前后的Bader电荷差异，
    从而确定吸附引起的电荷转移。
    For specified atom indices, computes the Bader charge difference
    before and after adsorption, determining charge transfer induced
    by adsorption.

    参数 / Parameters:
        acf_clean (dict): 清洁表面的Bader分析结果（parse_bader_results的返回值）。
            Bader analysis results of the clean surface (output of parse_bader_results).
        acf_adsorbed (dict): 吸附表面的Bader分析结果。
            Bader analysis results of the adsorbed surface.
        atom_indices (list of int): 要比较的原子索引列表（在清洁表面中的索引）。
            List of atom indices to compare (indices in the clean surface).

    返回 / Returns:
        dict: 包含以下键的字典:
            - 'atom_indices' (list of int): 比较的原子索引
              Compared atom indices
            - 'charges_clean' (list of float): 清洁表面的Bader电荷
              Bader charges on clean surface
            - 'charges_adsorbed' (list of float): 吸附表面的Bader电荷
              Bader charges on adsorbed surface
            - 'charge_diff' (list of float): 电荷差异（adsorbed - clean）
              Charge difference (adsorbed - clean)
            - 'elements' (list of str or None): 对应的元素符号
              Corresponding element symbols

    异常 / Raises:
        ValueError: 如果输入无效或索引超出范围。
            If inputs are invalid or indices are out of range.

    示例 / Example:
        >>> clean = parse_bader_results("clean/ACF.dat", "clean/POSCAR")
        >>> ads = parse_bader_results("ads/ACF.dat", "ads/POSCAR")
        >>> result = compare_bader_charges(clean, ads, [0, 1, 2, 3])
        >>> for idx, diff in zip(result['atom_indices'], result['charge_diff']):
        ...     print(f"Atom {idx}: ΔQ = {diff:+.4f} e")
    """
    if not isinstance(acf_clean, dict) or "charges" not in acf_clean:
        raise ValueError(
            "acf_clean 必须是parse_bader_results的返回值字典 / "
            "acf_clean must be a dict returned by parse_bader_results"
        )

    if not isinstance(acf_adsorbed, dict) or "charges" not in acf_adsorbed:
        raise ValueError(
            "acf_adsorbed 必须是parse_bader_results的返回值字典 / "
            "acf_adsorbed must be a dict returned by parse_bader_results"
        )

    n_clean = acf_clean["n_atoms"]
    n_ads = acf_adsorbed["n_atoms"]

    # 验证索引范围 / Validate indices
    for idx in atom_indices:
        if idx < 0 or idx >= n_clean:
            raise ValueError(
                f"原子索引 {idx} 超出清洁表面范围 [0, {n_clean - 1}] / "
                f"Atom index {idx} is out of range for clean surface [0, {n_clean - 1}]"
            )
        if idx >= n_ads:
            warnings.warn(
                f"原子索引 {idx} 超出吸附表面范围 [0, {n_ads - 1}]，"
                f"可能因为吸附体系增加了原子。请确认索引对应关系。 / "
                f"Atom index {idx} is out of range for adsorbed surface "
                f"[0, {n_ads - 1}]. This may be because the adsorbed system "
                f"has additional atoms. Please verify index correspondence."
            )

    charges_clean = []
    charges_adsorbed = []
    charge_diff = []
    elements = []

    for idx in atom_indices:
        q_clean = acf_clean["charges"][idx]
        charges_clean.append(q_clean)

        if idx < n_ads:
            q_ads = acf_adsorbed["charges"][idx]
            charges_adsorbed.append(q_ads)
            charge_diff.append(q_ads - q_clean)
        else:
            charges_adsorbed.append(float("nan"))
            charge_diff.append(float("nan"))

        # 获取元素信息 / Get element info
        elem_clean = acf_clean.get("elements", [None] * n_clean)
        elem = elem_clean[idx] if idx < len(elem_clean) else None
        elements.append(elem)

    return {
        "atom_indices": atom_indices,
        "charges_clean": charges_clean,
        "charges_adsorbed": charges_adsorbed,
        "charge_diff": charge_diff,
        "elements": elements,
    }


def summarize_bader_by_element(charges, elements):
    """
    按元素汇总Bader电荷 / Summarize Bader charges by element.

    将Bader电荷按元素类型分组，计算每组的平均值、标准差和计数。
    Groups Bader charges by element type, computing mean, standard
    deviation, and count for each group.

    参数 / Parameters:
        charges (list of float): 每个原子的Bader电荷列表。
            List of per-atom Bader charges.
        elements (list of str): 每个原子的元素符号列表。
            List of per-atom element symbols.

    返回 / Returns:
        dict: 元素到统计信息的映射字典，每个元素包含:
            - 'mean' (float): 平均Bader电荷
            - 'std' (float): 标准差
            - 'min' (float): 最小值
            - 'max' (float): 最大值
            - 'count' (int): 原子数量
            Element -> statistics dict with mean, std, min, max, count.

    异常 / Raises:
        ValueError: 如果输入长度不匹配或为空。
            If input lengths do not match or are empty.

    示例 / Example:
        >>> summary = summarize_bader_by_element(results['charges'], results['elements'])
        >>> for elem, stats in summary.items():
        ...     print(f"{elem}: {stats['mean']:.3f} ± {stats['std']:.3f} e (n={stats['count']})")
    """
    if len(charges) != len(elements):
        raise ValueError(
            f"charges长度 ({len(charges)}) 与elements长度 ({len(elements)}) 不匹配 / "
            f"Length of charges ({len(charges)}) does not match length of elements ({len(elements)})"
        )

    if len(charges) == 0:
        raise ValueError(
            "charges和elements不能为空 / charges and elements cannot be empty"
        )

    charges = np.asarray(charges, dtype=float)
    grouped = defaultdict(list)

    for elem, charge in zip(elements, charges):
        if elem is not None:
            grouped[elem].append(charge)

    summary = {}
    for elem, charge_list in sorted(grouped.items()):
        arr = np.array(charge_list)
        summary[elem] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr),
        }

    return summary


def export_bader_report(results, output_path):
    """
    导出Bader分析结果为CSV文件 / Export Bader analysis results to CSV file.

    将Bader电荷分析结果导出为结构化的CSV文件，方便后续处理和绘图。
    Exports Bader charge analysis results to a structured CSV file
    for further processing and plotting.

    参数 / Parameters:
        results (dict): parse_bader_results的返回值字典。
            Dict returned by parse_bader_results.
        output_path (str): 输出CSV文件路径。
            Output CSV file path.

    返回 / Returns:
        str: 输出文件的绝对路径。
            Absolute path of the output file.

    异常 / Raises:
        ValueError: 如果results缺少必需的键。
            If results is missing required keys.
        IOError: 如果文件写入失败。
            If file writing fails.

    示例 / Example:
        >>> results = parse_bader_results("ACF.dat", "POSCAR")
        >>> path = export_bader_report(results, "bader_report.csv")
        >>> print(f"Report saved to: {path}")
    """
    required_keys = ["charges", "volumes", "min_dist", "n_atoms"]
    for key in required_keys:
        if key not in results:
            raise ValueError(
                f"results缺少必需的键 '{key}' / "
                f"results is missing required key '{key}'"
            )

    n_atoms = results["n_atoms"]
    elements = results.get("elements", [None] * n_atoms)
    charge_transfer = results.get("charge_transfer", [None] * n_atoms)

    output_path = os.path.abspath(output_path)

    # 确保输出目录存在 / Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # 写入表头 / Write header
            header = ["Index", "Element", "Bader_Charge_e", "Charge_Transfer_e",
                      "Min_Dist_A", "Volume_A3"]
            writer.writerow(header)

            # 写入数据 / Write data
            for i in range(n_atoms):
                elem = elements[i] if i < len(elements) else ""
                ct = charge_transfer[i] if i < len(charge_transfer) else ""
                row = [
                    i,
                    elem,
                    f"{results['charges'][i]:.6f}",
                    f"{ct:.6f}" if ct is not None else "",
                    f"{results['min_dist'][i]:.6f}",
                    f"{results['volumes'][i]:.6f}",
                ]
                writer.writerow(row)

    except IOError as e:
        raise IOError(
            f"无法写入CSV文件 '{output_path}': {e} / "
            f"Failed to write CSV file '{output_path}': {e}"
        )

    return output_path
