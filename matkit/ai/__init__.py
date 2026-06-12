"""
MatKit AI助手模块 / MatKit AI Assistant Module
================================================

提供基于大语言模型的计算材料科学AI助手，支持结果分析、
参数建议、现象解释等功能。
Provides an LLM-powered computational materials science AI assistant,
supporting result analysis, parameter suggestions, and phenomenon explanation.

公共接口 / Public API:
    - MatKitAI: AI助手主类
"""

from matkit.ai.assistant import MatKitAI

__all__ = [
    "MatKitAI",
]
