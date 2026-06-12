"""
MatKit I/O 解析器模块
=====================

提供 VASP 输入/输出文件的纯 Python + NumPy 解析器，不依赖 pymatgen。

支持的文件格式:
- POSCAR / CONTCAR: 晶体结构文件
- OUTCAR: VASP 计算输出文件
- CHGCAR: 电荷密度文件
- ACF.dat: Bader 电荷分析结果文件
- OSZICAR: 能量收敛信息文件

所有解析器返回纯 Python 字典和 NumPy 数组，不使用自定义类。
"""

from matkit.parsers.poscar import expand_atomic_elements, read_poscar, write_poscar
from matkit.parsers.outcar import read_outcar
from matkit.parsers.chgcar import read_chgcar
from matkit.parsers.bader_parser import read_acf
from matkit.parsers.oszicar import read_oszicar

__all__ = [
    "read_poscar",
    "write_poscar",
    "expand_atomic_elements",
    "read_outcar",
    "read_chgcar",
    "read_acf",
    "read_oszicar",
]
