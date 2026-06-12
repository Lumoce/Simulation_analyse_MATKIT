"""
MatKit - 综合性 VASP 后处理与材料分析工具包
=================================================

MatKit - Comprehensive VASP Post-Processing and Materials Analysis Toolkit

提供 VASP DFT 计算的完整后处理分析工具链，包括：
Provides a complete post-processing analysis toolchain for VASP DFT calculations:

  - matkit.parsers: VASP 输入/输出文件解析器 (POSCAR, OUTCAR, CHGCAR, ACF.dat, OSZICAR)
  - matkit.structure: 结构分析 (几何计算、分子识别、表面分析)
  - matkit.energy: 能量计算 (表面能、吸附能、掺杂能/缺陷形成能)
  - matkit.charge: 电荷分析 (差分电荷密度、Bader 电荷分析、平面平均)
  - matkit.analysis: 高层工作流 (SO4/Cu2O 吸附、通用吸附几何、建议)
  - matkit.ai: AI 助手 (基于大语言模型的结果分析、参数建议、现象解释)
  - matkit.cli: 命令行接口 (python -m matkit <command> [options])

使用示例 / Usage Example:
    >>> from matkit.parsers import read_poscar, read_outcar
    >>> from matkit.energy import calc_surface_energy
    >>> from matkit.charge import calc_diff_charge_density
    >>> from matkit.ai import MatKitAI

命令行使用 / Command-line usage:
    python -m matkit energy surface --slab OUTCAR_slab --bulk OUTCAR_bulk --n-bulk 2
    python -m matkit structure info POSCAR
    python -m matkit charge diff --total CHGCAR_total --slab CHGCAR_slab --ads CHGCAR_ads
    python -m matkit ai analyze --context "SO4 adsorption on Cu2O" --results results.json
"""

__version__ = "0.2.0"
__author__ = "MatKit Team"

from matkit import parsers
from matkit import structure
from matkit import energy
from matkit import charge
from matkit import analysis
from matkit import ai

__all__ = [
    "parsers",
    "structure",
    "energy",
    "charge",
    "analysis",
    "ai",
]
