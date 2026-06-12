"""
matkit.energy.doping_energy - 掺杂能（缺陷形成能）计算
Doping energy (defect formation energy) calculation for MatKit

提供缺陷形成能的计算功能，支持单个计算和多配置批量计算。
Provides defect formation energy calculation, supporting single and
multi-configuration batch computation.
"""

import numpy as np


def calc_doping_energy(E_doped, E_pristine, n_dopant, mu_dopant, mu_host,
                       charge_state=0, efermi=0, correction=0):
    """
    计算掺杂缺陷形成能。
    Calculate defect formation energy for doping.

    公式：
    E_f = E_doped - E_pristine - n_dopant * mu_dopant + n_host * mu_host + q * E_fermi + E_corr

    其中：
    - E_doped: 掺杂体系总能量
    - E_pristine: 未掺杂（原始）体系总能量
    - n_dopant: 掺入的杂质原子数
    - mu_dopant: 杂质原子的化学势
    - n_host: 被替换的宿主原子数（通常等于 n_dopant）
    - mu_host: 宿主原子的化学势
    - q: 电荷态
    - E_fermi: 费米能级（相对于价带顶）
    - E_corr: 电荷校正能（如 Makov-Payne 校正）

    Formula:
    E_f = E_doped - E_pristine - n_dopant * mu_dopant + n_host * mu_host + q * E_fermi + E_corr

    Where:
    - E_doped: Total energy of the doped system
    - E_pristine: Total energy of the pristine (undoped) system
    - n_dopant: Number of dopant atoms introduced
    - mu_dopant: Chemical potential of the dopant atom
    - n_host: Number of host atoms replaced (usually equal to n_dopant)
    - mu_host: Chemical potential of the host atom
    - q: Charge state
    - E_fermi: Fermi level (relative to valence band maximum)
    - E_corr: Charge correction energy (e.g., Makov-Payne correction)

    Parameters
    ----------
    E_doped : float
        掺杂体系总能量（eV）。
        Total energy of the doped system in eV.
    E_pristine : float
        未掺杂体系总能量（eV）。
        Total energy of the pristine system in eV.
    n_dopant : int
        掺入的杂质原子数。
        Number of dopant atoms introduced.
    mu_dopant : float
        杂质原子的化学势（eV）。
        Chemical potential of the dopant atom in eV.
    mu_host : float
        宿主原子的化学势（eV）。
        Chemical potential of the host atom in eV.
    charge_state : int, optional
        缺陷电荷态，默认为 0（中性）。
        Defect charge state, default 0 (neutral).
    efermi : float, optional
        费米能级（eV），相对于价带顶，默认为 0。
        Fermi level in eV, relative to valence band maximum, default 0.
    correction : float, optional
        电荷校正能（eV），默认为 0。
        Charge correction energy in eV, default 0.

    Returns
    -------
    dict
        缺陷形成能计算结果：
        Defect formation energy calculation result:
        - 'formation_energy_eV': float - 缺陷形成能（eV）/ Defect formation energy in eV
        - 'E_doped': float - 掺杂体系能量 / Doped system energy
        - 'E_pristine': float - 未掺杂体系能量 / Pristine system energy
        - 'n_dopant': int - 杂质原子数 / Number of dopant atoms
        - 'n_host': int - 宿主原子数 / Number of host atoms
        - 'mu_dopant': float - 杂质化学势 / Dopant chemical potential
        - 'mu_host': float - 宿主化学势 / Host chemical potential
        - 'chemical_potential_term': float - 化学势项 / Chemical potential term
        - 'charge_state': int - 电荷态 / Charge state
        - 'efermi': float - 费米能级 / Fermi level
        - 'charge_correction_term': float - 电荷校正项 / Charge correction term
        - 'correction': float - 校正能量 / Correction energy
        - 'energy_diff': float - 能量差 (E_doped - E_pristine) / Energy difference

    Raises
    ------
    ValueError
        如果输入验证失败。
        If input validation fails.

    Examples
    --------
    >>> result = calc_doping_energy(-100.0, -98.0, 1, -5.0, -4.0)
    >>> result['formation_energy_eV']
    -3.0
    >>> result = calc_doping_energy(-100.0, -98.0, 1, -5.0, -4.0,
    ...                              charge_state=1, efermi=0.5)
    """
    # 输入验证 / Input validation
    for name, val in [('E_doped', E_doped), ('E_pristine', E_pristine),
                       ('mu_dopant', mu_dopant), ('mu_host', mu_host),
                       ('efermi', efermi), ('correction', correction)]:
        if not isinstance(val, (int, float)):
            raise ValueError(
                f"{name} 必须是数值类型，但得到 {type(val).__name__}。"
                f"{name} must be a numeric type, but got {type(val).__name__}."
            )

    for name, val in [('n_dopant', n_dopant), ('charge_state', charge_state)]:
        if not isinstance(val, (int, np.integer)):
            raise ValueError(
                f"{name} 必须是整数，但得到 {type(val).__name__}。"
                f"{name} must be an integer, but got {type(val).__name__}."
            )

    if n_dopant <= 0:
        raise ValueError(
            f"n_dopant 必须是正整数，但得到 {n_dopant}。"
            f"n_dopant must be a positive integer, but got {n_dopant}."
        )

    # 转换为标准类型 / Convert to standard types
    E_doped = float(E_doped)
    E_pristine = float(E_pristine)
    n_dopant = int(n_dopant)
    n_host = n_dopant  # 被替换的宿主原子数通常等于掺入的杂质数
    mu_dopant = float(mu_dopant)
    mu_host = float(mu_host)
    charge_state = int(charge_state)
    efermi = float(efermi)
    correction = float(correction)

    # 能量差 / Energy difference
    energy_diff = E_doped - E_pristine

    # 化学势项 / Chemical potential term
    chemical_potential_term = -n_dopant * mu_dopant + n_host * mu_host

    # 电荷校正项 / Charge correction term
    charge_correction_term = charge_state * efermi + correction

    # 缺陷形成能 / Defect formation energy
    formation_energy = energy_diff + chemical_potential_term + charge_correction_term

    return {
        'formation_energy_eV': formation_energy,
        'E_doped': E_doped,
        'E_pristine': E_pristine,
        'n_dopant': n_dopant,
        'n_host': n_host,
        'mu_dopant': mu_dopant,
        'mu_host': mu_host,
        'chemical_potential_term': chemical_potential_term,
        'charge_state': charge_state,
        'efermi': efermi,
        'charge_correction_term': charge_correction_term,
        'correction': correction,
        'energy_diff': energy_diff,
    }


def calc_doping_energies_table(configurations):
    """
    计算多种电荷态和化学势组合下的掺杂能表格。
    Calculate doping energies for multiple charge states and chemical potentials.

    Parameters
    ----------
    configurations : list of dict
        配置列表，每个字典包含：
        Configuration list, each dict contains:
        - 'E_doped': float - 掺杂体系能量（eV）/ Doped system energy in eV
        - 'E_pristine': float - 未掺杂体系能量（eV）/ Pristine system energy in eV
        - 'n_dopant': int - 杂质原子数 / Number of dopant atoms
        - 'mu_dopant': float or list - 杂质化学势（eV），单个值或列表 / Dopant chemical potential, single or list
        - 'mu_host': float or list - 宿主化学势（eV），单个值或列表 / Host chemical potential, single or list
        - 'charge_states': list of int, optional - 电荷态列表，默认 [0] / Charge state list, default [0]
        - 'efermi_values': list of float, optional - 费米能级列表，默认 [0] / Fermi level list, default [0]
        - 'correction': float, optional - 校正能量，默认 0 / Correction energy, default 0
        - 'label': str, optional - 配置标签 / Configuration label

    Returns
    -------
    list of dict
        计算结果列表，每个字典包含 calc_doping_energy 的结果
        以及额外的配置信息：
        Calculation result list, each dict contains the result of
        calc_doping_energy plus additional configuration info:
        - 'label': str or None - 配置标签 / Configuration label
        - 'config_index': int - 配置索引 / Configuration index
        - 以及 calc_doping_energy 返回的所有键
        - And all keys returned by calc_doping_energy

    Raises
    ------
    ValueError
        如果输入验证失败。
        If input validation fails.

    Examples
    --------
    >>> configurations = [{
    ...     'E_doped': -100.0, 'E_pristine': -98.0, 'n_dopant': 1,
    ...     'mu_dopant': [-5.0, -4.5], 'mu_host': -4.0,
    ...     'charge_states': [0, 1, -1], 'efermi_values': [0, 0.5, 1.0],
    ...     'label': 'N-doped ZnO',
    ... }]
    >>> results = calc_doping_energies_table(configurations)
    """
    if not isinstance(configurations, list):
        raise ValueError(
            "configurations 必须是列表。"
            "configurations must be a list."
        )

    if len(configurations) == 0:
        return []

    results = []
    result_index = 0

    for config_idx, config in enumerate(configurations):
        if not isinstance(config, dict):
            raise ValueError(
                f"configurations[{config_idx}] 必须是字典，但得到 {type(config).__name__}。"
                f"configurations[{config_idx}] must be a dict, but got {type(config).__name__}."
            )

        # 验证必要参数 / Validate required parameters
        required_keys = {'E_doped', 'E_pristine', 'n_dopant', 'mu_dopant', 'mu_host'}
        missing = required_keys - set(config.keys())
        if missing:
            raise ValueError(
                f"configurations[{config_idx}] 缺少必要的键：{missing}。"
                f"configurations[{config_idx}] is missing required keys: {missing}."
            )

        E_doped = config['E_doped']
        E_pristine = config['E_pristine']
        n_dopant = config['n_dopant']
        mu_dopant = config['mu_dopant']
        mu_host = config['mu_host']
        charge_states = config.get('charge_states', [0])
        efermi_values = config.get('efermi_values', [0])
        correction = config.get('correction', 0)
        label = config.get('label', None)

        # 标准化为列表 / Normalize to lists
        if not isinstance(mu_dopant, list):
            mu_dopant = [mu_dopant]
        if not isinstance(mu_host, list):
            mu_host = [mu_host]
        if not isinstance(charge_states, list):
            charge_states = [charge_states]
        if not isinstance(efermi_values, list):
            efermi_values = [efermi_values]

        # 验证列表内容 / Validate list contents
        if len(mu_dopant) == 0:
            raise ValueError(
                f"configurations[{config_idx}] 的 mu_dopant 列表不能为空。"
                f"configurations[{config_idx}] mu_dopant list cannot be empty."
            )
        if len(mu_host) == 0:
            raise ValueError(
                f"configurations[{config_idx}] 的 mu_host 列表不能为空。"
                f"configurations[{config_idx}] mu_host list cannot be empty."
            )
        if len(charge_states) == 0:
            raise ValueError(
                f"configurations[{config_idx}] 的 charge_states 列表不能为空。"
                f"configurations[{config_idx}] charge_states list cannot be empty."
            )
        if len(efermi_values) == 0:
            raise ValueError(
                f"configurations[{config_idx}] 的 efermi_values 列表不能为空。"
                f"configurations[{config_idx}] efermi_values list cannot be empty."
            )

        # 遍历所有组合 / Iterate over all combinations
        for mu_d in mu_dopant:
            for mu_h in mu_host:
                for q in charge_states:
                    for ef in efermi_values:
                        result = calc_doping_energy(
                            E_doped=E_doped,
                            E_pristine=E_pristine,
                            n_dopant=n_dopant,
                            mu_dopant=mu_d,
                            mu_host=mu_h,
                            charge_state=q,
                            efermi=ef,
                            correction=correction,
                        )

                        result['label'] = label
                        result['config_index'] = config_idx
                        result['result_index'] = result_index
                        results.append(result)
                        result_index += 1

    return results
