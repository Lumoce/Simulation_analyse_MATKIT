"""
matkit.energy.adsorption_energy - 吸附能计算
Adsorption energy calculation for MatKit

提供吸附能的计算功能，支持单个计算和批量计算。
Provides adsorption energy calculation, supporting single and batch computation.
"""

import numpy as np
import os


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
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"OUTCAR 文件不存在：{filepath}。"
            f"OUTCAR file not found: {filepath}."
        )

    energy = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if 'free  energy   TOTEN' in line or 'energy  without entropy' in line:
                parts = line.split('=')
                if len(parts) >= 2:
                    try:
                        energy = float(parts[-1].strip().split()[0])
                    except (ValueError, IndexError):
                        continue

    if energy is None:
        raise ValueError(
            f"无法从 OUTCAR 文件中解析总能量：{filepath}。"
            f"Cannot parse total energy from OUTCAR file: {filepath}."
        )

    return energy


def calc_adsorption_energy(E_slab_ads, E_slab_clean, E_adsorbate, n_adsorbate=1):
    """
    计算吸附能。
    Calculate adsorption energy.

    公式：E_ads = E_slab+ads - E_slab_clean - n * E_adsorbate

    负值表示吸附过程放热（有利），正值表示吸热（不利）。
    Formula: E_ads = E_slab+ads - E_slab_clean - n * E_adsorbate

    Negative values indicate exothermic (favorable) adsorption,
    positive values indicate endothermic (unfavorable) adsorption.

    Parameters
    ----------
    E_slab_ads : float
        吸附后板模型的总能量（eV）。
        Total energy of the adsorbed slab model in eV.
    E_slab_clean : float
        清洁板模型的总能量（eV）。
        Total energy of the clean slab model in eV.
    E_adsorbate : float
        单个吸附物分子的能量（eV）。
        Energy of a single adsorbate molecule in eV.
    n_adsorbate : int, optional
        吸附物分子数量，默认为 1。
        Number of adsorbate molecules, default 1.

    Returns
    -------
    dict
        吸附能计算结果：
        Adsorption energy calculation result:
        - 'adsorption_energy_eV': float - 总吸附能（eV）/ Total adsorption energy in eV
        - 'adsorption_energy_per_molecule': float - 每个分子的吸附能（eV）/ Adsorption energy per molecule in eV
        - 'E_slab_ads': float - 吸附后板模型能量 / Adsorbed slab energy
        - 'E_slab_clean': float - 清洁板模型能量 / Clean slab energy
        - 'E_adsorbate_total': float - 吸附物总能量 / Total adsorbate energy
        - 'n_adsorbate': int - 吸附物数量 / Number of adsorbates

    Raises
    ------
    ValueError
        如果输入验证失败。
        If input validation fails.

    Examples
    --------
    >>> result = calc_adsorption_energy(-25.0, -20.0, -3.0)
    >>> result['adsorption_energy_eV']
    -2.0
    """
    # 输入验证 / Input validation
    for name, val in [('E_slab_ads', E_slab_ads),
                       ('E_slab_clean', E_slab_clean),
                       ('E_adsorbate', E_adsorbate)]:
        if not isinstance(val, (int, float)):
            raise ValueError(
                f"{name} 必须是数值类型，但得到 {type(val).__name__}。"
                f"{name} must be a numeric type, but got {type(val).__name__}."
            )

    if not isinstance(n_adsorbate, (int, np.integer)) or n_adsorbate <= 0:
        raise ValueError(
            f"n_adsorbate 必须是正整数，但得到 {n_adsorbate}。"
            f"n_adsorbate must be a positive integer, but got {n_adsorbate}."
        )

    E_slab_ads = float(E_slab_ads)
    E_slab_clean = float(E_slab_clean)
    E_adsorbate = float(E_adsorbate)
    n_adsorbate = int(n_adsorbate)

    E_adsorbate_total = n_adsorbate * E_adsorbate

    # 吸附能 / Adsorption energy
    adsorption_energy_eV = E_slab_ads - E_slab_clean - E_adsorbate_total
    adsorption_energy_per_molecule = adsorption_energy_eV / n_adsorbate

    return {
        'adsorption_energy_eV': adsorption_energy_eV,
        'adsorption_energy_per_molecule': adsorption_energy_per_molecule,
        'E_slab_ads': E_slab_ads,
        'E_slab_clean': E_slab_clean,
        'E_adsorbate_total': E_adsorbate_total,
        'n_adsorbate': n_adsorbate,
    }


def calc_adsorption_energies_batch(file_pairs):
    """
    批量计算吸附能。
    Batch calculation of adsorption energies.

    Parameters
    ----------
    file_pairs : list of dict
        文件路径对列表，每个字典包含：
        List of file path pairs, each dict contains:
        - 'slab_ads': str - 吸附后板模型的 OUTCAR 路径 / Path to adsorbed slab OUTCAR
        - 'slab_clean': str - 清洁板模型的 OUTCAR 路径 / Path to clean slab OUTCAR
        - 'adsorbate': str - 吸附物的 OUTCAR 路径 / Path to adsorbate OUTCAR
        - 'n_adsorbate': int, optional - 吸附物数量，默认 1 / Number of adsorbates, default 1
        - 'label': str, optional - 标签名称 / Label name

    Returns
    -------
    list of dict
        吸附能计算结果列表，每个字典包含 calc_adsorption_energy 的结果
        以及额外的文件信息：
        List of adsorption energy results, each dict contains the result of
        calc_adsorption_energy plus additional file info:
        - 'label': str or None - 标签 / Label
        - 'slab_ads_file': str - 吸附后 OUTCAR 路径 / Adsorbed OUTCAR path
        - 'slab_clean_file': str - 清洁 OUTCAR 路径 / Clean OUTCAR path
        - 'adsorbate_file': str - 吸附物 OUTCAR 路径 / Adsorbate OUTCAR path
        - 以及 calc_adsorption_energy 返回的所有键
        - And all keys returned by calc_adsorption_energy

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
    >>> file_pairs = [
    ...     {'slab_ads': 'SO4/Cu111/OUTCAR', 'slab_clean': 'Cu111/OUTCAR',
    ...      'adsorbate': 'SO4/OUTCAR', 'label': 'SO4@Cu111'},
    ...     {'slab_ads': 'SO4/Cu100/OUTCAR', 'slab_clean': 'Cu100/OUTCAR',
    ...      'adsorbate': 'SO4/OUTCAR', 'label': 'SO4@Cu100'},
    ... ]
    >>> results = calc_adsorption_energies_batch(file_pairs)
    """
    if not isinstance(file_pairs, list):
        raise ValueError(
            "file_pairs 必须是列表。"
            "file_pairs must be a list."
        )

    if len(file_pairs) == 0:
        return []

    results = []
    for i, pair in enumerate(file_pairs):
        if not isinstance(pair, dict):
            raise ValueError(
                f"file_pairs[{i}] 必须是字典，但得到 {type(pair).__name__}。"
                f"file_pairs[{i}] must be a dict, but got {type(pair).__name__}."
            )

        required_keys = {'slab_ads', 'slab_clean', 'adsorbate'}
        missing = required_keys - set(pair.keys())
        if missing:
            raise ValueError(
                f"file_pairs[{i}] 缺少必要的键：{missing}。"
                f"file_pairs[{i}] is missing required keys: {missing}."
            )

        slab_ads_path = pair['slab_ads']
        slab_clean_path = pair['slab_clean']
        adsorbate_path = pair['adsorbate']

        for path_name, path in [('slab_ads', slab_ads_path),
                                 ('slab_clean', slab_clean_path),
                                 ('adsorbate', adsorbate_path)]:
            if not isinstance(path, str) or not path.strip():
                raise ValueError(
                    f"file_pairs[{i}]['{path_name}'] 必须是非空字符串路径。"
                    f"file_pairs[{i}]['{path_name}'] must be a non-empty string path."
                )

        n_adsorbate = pair.get('n_adsorbate', 1)
        label = pair.get('label', None)

        # 读取能量 / Read energies
        E_slab_ads = _read_outcar_energy(slab_ads_path)
        E_slab_clean = _read_outcar_energy(slab_clean_path)
        E_adsorbate = _read_outcar_energy(adsorbate_path)

        # 计算吸附能 / Calculate adsorption energy
        result = calc_adsorption_energy(
            E_slab_ads=E_slab_ads,
            E_slab_clean=E_slab_clean,
            E_adsorbate=E_adsorbate,
            n_adsorbate=n_adsorbate,
        )

        # 添加文件信息 / Add file info
        result['label'] = label
        result['slab_ads_file'] = slab_ads_path
        result['slab_clean_file'] = slab_clean_path
        result['adsorbate_file'] = adsorbate_path

        results.append(result)

    return results
