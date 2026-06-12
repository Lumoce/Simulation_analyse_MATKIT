"""
MatKit 电荷分析模块 / MatKit Charge Analysis Module
=====================================================

提供差分电荷密度分析、Bader电荷分析等电荷相关计算工具。
Provides differential charge density analysis, Bader charge analysis,
and other charge-related computational tools.

公共接口 / Public API:
    - calc_diff_charge_density: 计算差分电荷密度
    - planar_average: 计算平面平均电荷密度
    - macroscopic_average: 计算宏观平均电荷密度
    - integrate_charge_in_region: 在指定区域积分电荷密度
    - charge_transfer_analysis: 分析电荷转移
    - export_cube: 导出Gaussian cube格式文件
    - parse_bader_results: 解析Bader分析结果
    - compare_bader_charges: 比较Bader电荷
    - summarize_bader_by_element: 按元素汇总Bader电荷
    - export_bader_report: 导出Bader分析报告
"""

from matkit.charge.charge_density import (
    calc_diff_charge_density,
    planar_average,
    macroscopic_average,
    integrate_charge_in_region,
    charge_transfer_analysis,
    export_cube,
)

from matkit.charge.bader_analysis import (
    parse_bader_results,
    compare_bader_charges,
    summarize_bader_by_element,
    export_bader_report,
)

__all__ = [
    # 电荷密度分析 / Charge density analysis
    "calc_diff_charge_density",
    "planar_average",
    "macroscopic_average",
    "integrate_charge_in_region",
    "charge_transfer_analysis",
    "export_cube",
    # Bader分析 / Bader analysis
    "parse_bader_results",
    "compare_bader_charges",
    "summarize_bader_by_element",
    "export_bader_report",
]
