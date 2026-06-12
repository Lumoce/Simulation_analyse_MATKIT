"""
MatKit CLI 模块 / MatKit CLI Module
=====================================

提供基于 argparse 的命令行接口，支持所有 MatKit 功能模块。
Provides argparse-based command-line interface for all MatKit modules.

使用方式 / Usage:
    python -m matkit <command> [options]

可用命令 / Available commands:
    energy      - 能量计算 (表面能、吸附能、掺杂能)
    structure   - 结构分析 (几何、分子识别、表面分析)
    charge      - 电荷分析 (差分电荷密度、Bader 分析)
    adsorption  - 吸附分析 (吸附物识别、键长、表面变化)
    ai          - AI 助手 (结果分析、建议、INCAR 生成)
"""

from matkit.cli.main import main

__all__ = ["main"]
