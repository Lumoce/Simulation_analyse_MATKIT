"""
预构建的提示词模板 / Pre-built Prompt Templates
=================================================

为MatKitAI助手提供结构化的提示词模板，用于不同场景下的分析任务。
Provides structured prompt templates for the MatKitAI assistant,
tailored for different analysis scenarios.

每个模板是一个函数，接受相关参数并返回格式化的提示词字符串。
Each template is a function that takes relevant parameters and returns
a formatted prompt string.

使用示例 / Usage Example:
    >>> from matkit.ai.prompts import ENERGY_ANALYSIS_PROMPT
    >>> prompt = ENERGY_ANALYSIS_PROMPT(
    ...     system="Cu(111) surface with CO adsorption",
    ...     energy=-123.456,
    ...     energy_ref=-120.789,
    ...     details="PBE functional, 500 eV cutoff, 6x6x1 k-mesh"
    ... )
"""

import json


def ENERGY_ANALYSIS_PROMPT(system="", energy=0.0, energy_ref=None,
                           adsorption_energy=None, details=""):
    """
    能量分析提示词模板 / Energy analysis prompt template.

    用于解释DFT能量计算结果，包括总能量、吸附能等。
    Template for interpreting DFT energy calculation results,
    including total energy, adsorption energy, etc.

    参数 / Parameters:
        system (str): 体系描述。
            System description.
        energy (float): 总能量（eV）。
            Total energy (eV).
        energy_ref (float, optional): 参考能量（eV）。
            Reference energy (eV).
        adsorption_energy (float, optional): 吸附能（eV）。
            Adsorption energy (eV).
        details (str): 计算细节。
            Calculation details.

    返回 / Returns:
        str: 格式化的提示词字符串。
            Formatted prompt string.
    """
    data = {
        "system": system,
        "total_energy_eV": energy,
    }
    if energy_ref is not None:
        data["reference_energy_eV"] = energy_ref
        data["energy_difference_eV"] = energy - energy_ref
    if adsorption_energy is not None:
        data["adsorption_energy_eV"] = adsorption_energy
    if details:
        data["calculation_details"] = details

    data_str = json.dumps(data, indent=2, ensure_ascii=False)

    return (
        "请分析以下DFT能量计算结果，并用用户使用的语言进行解释。\n"
        "Please analyze the following DFT energy calculation results "
        "and explain them in the user's language.\n\n"
        f"## 计算数据 / Calculation Data\n\n"
        f"体系 / System: {system}\n\n"
        f"```json\n{data_str}\n```\n\n"
        "请提供以下分析:\n"
        "Please provide the following analysis:\n"
        "1. 能量数据的物理意义 / Physical meaning of the energy data\n"
        "2. 与典型参考值的比较 / Comparison with typical reference values\n"
        "3. 结果的可靠性评估 / Reliability assessment of the results\n"
        "4. 可能的改进建议 / Possible improvement suggestions\n"
    )


def STRUCTURE_ANALYSIS_PROMPT(system="", lattice_params=None, atoms=None,
                              bond_lengths=None, angles=None, details=""):
    """
    结构分析提示词模板 / Structure analysis prompt template.

    用于解释结构优化结果，包括晶格参数、键长、键角等。
    Template for interpreting structural optimization results,
    including lattice parameters, bond lengths, bond angles, etc.

    参数 / Parameters:
        system (str): 体系描述。
            System description.
        lattice_params (dict, optional): 晶格参数 {a, b, c, alpha, beta, gamma}。
            Lattice parameters {a, b, c, alpha, beta, gamma}.
        atoms (list, optional): 原子信息列表。
            List of atom information.
        bond_lengths (dict, optional): 键长信息。
            Bond length information.
        angles (dict, optional): 键角信息。
            Bond angle information.
        details (str): 计算细节。
            Calculation details.

    返回 / Returns:
        str: 格式化的提示词字符串。
            Formatted prompt string.
    """
    data = {"system": system}
    if lattice_params:
        data["lattice_parameters"] = lattice_params
    if atoms:
        data["atoms"] = atoms
    if bond_lengths:
        data["bond_lengths"] = bond_lengths
    if angles:
        data["angles"] = angles
    if details:
        data["calculation_details"] = details

    data_str = json.dumps(data, indent=2, ensure_ascii=False)

    return (
        "请分析以下晶体结构数据，并用用户使用的语言进行解释。\n"
        "Please analyze the following crystal structure data "
        "and explain them in the user's language.\n\n"
        f"## 结构数据 / Structure Data\n\n"
        f"体系 / System: {system}\n\n"
        f"```json\n{data_str}\n```\n\n"
        "请提供以下分析:\n"
        "Please provide the following analysis:\n"
        "1. 结构特征描述 / Description of structural features\n"
        "2. 与实验值或文献值的比较 / Comparison with experimental or literature values\n"
        "3. 结构稳定性评估 / Structural stability assessment\n"
        "4. 可能的对称性分析 / Possible symmetry analysis\n"
    )


def CHARGE_ANALYSIS_PROMPT(system="", method="Bader", charge_data=None,
                           charge_transfer=None, diff_charge_stats=None,
                           details=""):
    """
    电荷分析提示词模板 / Charge analysis prompt template.

    用于解释电荷分析结果，包括Bader电荷、差分电荷密度等。
    Template for interpreting charge analysis results,
    including Bader charges, differential charge density, etc.

    参数 / Parameters:
        system (str): 体系描述。
            System description.
        method (str): 分析方法（如 "Bader", "Hirshfeld", "Mulliken"）。
            Analysis method (e.g., "Bader", "Hirshfeld", "Mulliken").
        charge_data (dict, optional): 电荷数据。
            Charge data.
        charge_transfer (dict, optional): 电荷转移数据。
            Charge transfer data.
        diff_charge_stats (dict, optional): 差分电荷密度统计信息。
            Differential charge density statistics.
        details (str): 计算细节。
            Calculation details.

    返回 / Returns:
        str: 格式化的提示词字符串。
            Formatted prompt string.
    """
    data = {
        "system": system,
        "analysis_method": method,
    }
    if charge_data:
        data["charge_data"] = charge_data
    if charge_transfer:
        data["charge_transfer"] = charge_transfer
    if diff_charge_stats:
        data["diff_charge_stats"] = diff_charge_stats
    if details:
        data["calculation_details"] = details

    data_str = json.dumps(data, indent=2, ensure_ascii=False)

    return (
        "请分析以下电荷分析结果，并用用户使用的语言进行解释。\n"
        "Please analyze the following charge analysis results "
        "and explain them in the user's language.\n\n"
        f"## 电荷分析数据 / Charge Analysis Data\n\n"
        f"体系 / System: {system}\n"
        f"方法 / Method: {method}\n\n"
        f"```json\n{data_str}\n```\n\n"
        "请提供以下分析:\n"
        "Please provide the following analysis:\n"
        "1. 电荷分布特征 / Charge distribution characteristics\n"
        "2. 电荷转移方向和量级 / Charge transfer direction and magnitude\n"
        "3. 化学键性质分析（共价/离子/金属） / Chemical bonding analysis (covalent/ionic/metallic)\n"
        "4. 与类似体系的比较 / Comparison with similar systems\n"
        "5. 对催化活性或物理性质的影响 / Impact on catalytic activity or physical properties\n"
    )


def NEXT_STEPS_PROMPT(current_results="", available_tools="", research_goal=""):
    """
    下一步建议提示词模板 / Next steps suggestion prompt template.

    根据当前计算结果，建议后续的计算或分析步骤。
    Suggests subsequent calculation or analysis steps based on
    current results.

    参数 / Parameters:
        current_results (str): 当前结果的描述。
            Description of current results.
        available_tools (str): 可用工具列表。
            List of available tools.
        research_goal (str): 研究目标。
            Research goal.

    返回 / Returns:
        str: 格式化的提示词字符串。
            Formatted prompt string.
    """
    return (
        "基于以下当前计算结果和研究目标，请建议下一步的计算或分析工作。\n"
        "Based on the following current results and research goal, "
        "please suggest next calculation or analysis steps.\n\n"
        f"## 研究目标 / Research Goal\n{research_goal}\n\n"
        f"## 当前结果 / Current Results\n{current_results}\n\n"
        f"## 可用工具 / Available Tools\n{available_tools}\n\n"
        "请提供:\n"
        "Please provide:\n"
        "1. 优先级排序的下一步建议（按重要性排列）\n"
        "   Prioritized next-step suggestions (ordered by importance)\n"
        "2. 每个建议的理由和预期结果\n"
        "   Rationale and expected outcome for each suggestion\n"
        "3. 需要注意的潜在问题\n"
        "   Potential issues to watch out for\n"
        "4. 估计的计算资源需求\n"
        "   Estimated computational resource requirements\n"
    )


def INCAR_GENERATION_PROMPT(research_goal="", system_info="", functional="PBE",
                            previous_params=None, constraints=""):
    """
    INCAR参数生成提示词模板 / INCAR parameter generation prompt template.

    根据研究目标和体系信息，生成VASP INCAR参数建议。
    Generates VASP INCAR parameter suggestions based on research
    goals and system information.

    参数 / Parameters:
        research_goal (str): 研究目标描述。
            Research goal description.
        system_info (str): 体系信息（元素、结构类型等）。
            System information (elements, structure type, etc.).
        functional (str): 交换关联泛函。
            Exchange-correlation functional.
        previous_params (dict, optional): 之前使用的INCAR参数。
            Previously used INCAR parameters.
        constraints (str): 计算资源约束（如内存、核数限制）。
            Computational resource constraints (e.g., memory, core limits).

    返回 / Returns:
        str: 格式化的提示词字符串。
            Formatted prompt string.
    """
    data = {
        "research_goal": research_goal,
        "system_info": system_info,
        "functional": functional,
    }
    if previous_params:
        data["previous_INCAR_parameters"] = previous_params
    if constraints:
        data["constraints"] = constraints

    data_str = json.dumps(data, indent=2, ensure_ascii=False)

    return (
        "请根据以下信息生成VASP INCAR参数建议，并用用户使用的语言进行解释。\n"
        "Please generate VASP INCAR parameter suggestions based on "
        "the following information and explain in the user's language.\n\n"
        f"```json\n{data_str}\n```\n\n"
        "请提供:\n"
        "Please provide:\n"
        "1. 完整的INCAR参数列表（包含注释说明每个参数的作用）\n"
        "   Complete INCAR parameter list (with comments explaining each parameter)\n"
        "2. 每个参数的推荐值和选择理由\n"
        "   Recommended value and rationale for each parameter\n"
        "3. 参数之间的依赖关系和注意事项\n"
        "   Parameter interdependencies and caveats\n"
        "4. 针对该体系的特殊建议\n"
        "   Special suggestions for this particular system\n"
        "5. 收敛性测试建议（ENCUT, KPOINTS等）\n"
        "   Convergence testing suggestions (ENCUT, KPOINTS, etc.)\n"
        "请以VASP INCAR格式输出参数（KEY = VALUE），并在每个参数后添加注释行。\n"
        "Output parameters in VASP INCAR format (KEY = VALUE) with comment lines.\n"
    )
