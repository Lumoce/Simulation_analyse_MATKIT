"""
MatKit 入口点 / MatKit Entry Point
===================================

支持通过 python -m matkit 方式运行 CLI。

使用方式 / Usage:
    python -m matkit --help
    python -m matkit energy surface --slab OUTCAR_slab --bulk OUTCAR_bulk --n-bulk 2
"""

from matkit.cli.main import main

if __name__ == "__main__":
    main()
