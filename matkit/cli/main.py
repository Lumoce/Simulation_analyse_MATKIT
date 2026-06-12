"""
MatKit CLI 主入口 / MatKit CLI Main Entry Point
=================================================

基于 argparse 的命令行接口，提供所有 MatKit 功能的命令行访问。
argparse-based CLI providing command-line access to all MatKit functionality.

使用方式 / Usage:
    python -m matkit <command> [subcommand] [options]

命令列表 / Commands:
    energy      能量计算 (表面能、吸附能、掺杂能)
    structure   结构分析 (几何、分子识别、表面分析)
    charge      电荷分析 (差分电荷密度、Bader分析)
    adsorption  吸附分析 (吸附物识别、键长、表面变化)
    ai          AI助手 (结果分析、建议、INCAR生成)
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


# ============================================================================
# ANSI 颜色工具 / ANSI Color Utilities
# ============================================================================

class Colors:
    """ANSI 终端颜色代码 / ANSI terminal color codes."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"

    @staticmethod
    def disable():
        """禁用所有颜色输出 / Disable all color output."""
        Colors.RESET = ""
        Colors.BOLD = ""
        Colors.RED = ""
        Colors.GREEN = ""
        Colors.YELLOW = ""
        Colors.BLUE = ""
        Colors.MAGENTA = ""
        Colors.CYAN = ""
        Colors.WHITE = ""
        Colors.BG_BLUE = ""
        Colors.BG_GREEN = ""


def color_header(text):
    """生成带颜色的标题 / Generate colored header."""
    return f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}"


def color_success(text):
    """生成成功信息 / Generate success message."""
    return f"{Colors.GREEN}{text}{Colors.RESET}"


def color_error(text):
    """生成错误信息 / Generate error message."""
    return f"{Colors.RED}{text}{Colors.RESET}"


def color_warning(text):
    """生成警告信息 / Generate warning message."""
    return f"{Colors.YELLOW}{text}{Colors.RESET}"


def color_value(text):
    """生成数值高亮 / Generate highlighted value."""
    return f"{Colors.BOLD}{Colors.GREEN}{text}{Colors.RESET}"


def color_key(text):
    """生成键名高亮 / Generate highlighted key."""
    return f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}"


# ============================================================================
# 格式化输出工具 / Formatted Output Utilities
# ============================================================================

def print_banner():
    """打印 MatKit 横幅 / Print MatKit banner."""
    version = __import__("matkit").__version__
    print(f"{Colors.BOLD}{Colors.CYAN}"
          f"╔══════════════════════════════════════════════════╗\n"
          f"║  MatKit v{version:<5} - VASP Post-Processing Toolkit   ║\n"
          f"║  综合性 VASP 后处理与材料分析工具包              ║\n"
          f"╚══════════════════════════════════════════════════╝"
          f"{Colors.RESET}")
    print()


def print_table(headers, rows, col_widths=None):
    """
    打印格式化表格 / Print formatted table.

    Parameters
    ----------
    headers : list of str
        表头列表 / List of column headers.
    rows : list of list
        数据行列表 / List of data rows.
    col_widths : list of int, optional
        列宽列表，自动计算 / Column widths, auto-calculated if None.
    """
    if not rows:
        return

    n_cols = len(headers)

    if col_widths is None:
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                if i < n_cols:
                    col_widths[i] = max(col_widths[i], len(str(val)))

    # 表头分隔线 / Header separator
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(sep)

    # 表头 / Header
    header_parts = []
    for i, h in enumerate(headers):
        header_parts.append(f" {str(h):<{col_widths[i]}} ")
    print("|" + "|".join(header_parts) + "|")
    print(sep)

    # 数据行 / Data rows
    for row in rows:
        row_parts = []
        for i in range(n_cols):
            val = str(row[i]) if i < len(row) else ""
            row_parts.append(f" {val:<{col_widths[i]}} ")
        print("|" + "|".join(row_parts) + "|")

    print(sep)


def print_kv(key, value, indent=2):
    """打印键值对 / Print key-value pair."""
    print(f"{' ' * indent}{color_key(key + ':')} {color_value(str(value))}")


def save_csv(filepath, headers, rows):
    """
    保存数据到 CSV 文件 / Save data to CSV file.

    Parameters
    ----------
    filepath : str
        输出文件路径 / Output file path.
    headers : list of str
        表头列表 / List of column headers.
    rows : list of list
        数据行列表 / List of data rows.
    """
    filepath = os.path.abspath(filepath)
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

    print(f"\n{color_success('CSV 已保存 / CSV saved:')} {filepath}")


def save_json(filepath, data):
    """
    保存数据到 JSON 文件 / Save data to JSON file.

    Parameters
    ----------
    filepath : str
        输出文件路径 / Output file path.
    data : dict or list
        要保存的数据 / Data to save.
    """
    filepath = os.path.abspath(filepath)
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\n{color_success('JSON 已保存 / JSON saved:')} {filepath}")


def _json_default(obj):
    """JSON 序列化默认处理器 / JSON serialization default handler."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def handle_error(msg):
    """处理错误并退出 / Handle error and exit."""
    print(f"\n{color_error('错误 / Error:')} {msg}", file=sys.stderr)
    sys.exit(1)


def parse_pairs_string(pairs_str):
    """
    解析原子对字符串 / Parse atom pair string.

    格式: "1,2;3,4;5,6" -> [[1,2], [3,4], [5,6]]

    Parameters
    ----------
    pairs_str : str
        原子对字符串 / Atom pair string.

    Returns
    -------
    list of list of int
        原子索引对列表 / List of atom index pairs.
    """
    pairs = []
    for group in pairs_str.split(";"):
        group = group.strip()
        if not group:
            continue
        parts = group.split(",")
        if len(parts) != 2:
            raise ValueError(f"无效的原子对格式: '{group}'，应为 'i,j'")
        pairs.append([int(parts[0].strip()), int(parts[1].strip())])
    return pairs


def parse_cutoffs_string(cutoffs_str):
    """
    解析截断值字符串 / Parse cutoffs string.

    格式: "O-Cu:2.5,S-O:2.0" -> {('O','Cu'): 2.5, ('S','O'): 2.0}

    Parameters
    ----------
    cutoffs_str : str
        截断值字符串 / Cutoffs string.

    Returns
    -------
    dict
        元素对到截断值的映射 / Element pair to cutoff mapping.
    """
    cutoffs = {}
    for item in cutoffs_str.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"无效的截断值格式: '{item}'，应为 'Elem1-Elem2:value'")
        pair = parts[0].strip().split("-")
        if len(pair) != 2:
            raise ValueError(f"无效的元素对格式: '{parts[0]}'，应为 'Elem1-Elem2'")
        cutoffs[(pair[0].strip(), pair[1].strip())] = float(parts[1].strip())
    return cutoffs


def parse_indices_string(indices_str):
    """
    解析索引字符串 / Parse indices string.

    格式: "1,2,3,4,5" -> [1, 2, 3, 4, 5]

    Parameters
    ----------
    indices_str : str
        索引字符串 / Indices string.

    Returns
    -------
    list of int
        索引列表 / List of indices.
    """
    return [int(x.strip()) for x in indices_str.split(",") if x.strip()]


def atom_elements_from_poscar_data(data):
    """Return one element label per atom from parsed POSCAR data."""
    if "atom_elements" in data:
        return list(data["atom_elements"])
    elements = []
    for elem, count in data["n_atoms"].items():
        elements.extend([elem] * count)
    return elements


# ============================================================================
# 子命令: energy
# ============================================================================

def cmd_energy(args):
    """能量计算子命令 / Energy calculation subcommand."""
    if args.subcommand == "surface":
        _cmd_energy_surface(args)
    elif args.subcommand == "surface-batch":
        _cmd_energy_surface_batch(args)
    elif args.subcommand == "adsorption":
        _cmd_energy_adsorption(args)
    elif args.subcommand == "doping":
        _cmd_energy_doping(args)
    else:
        handle_error(f"未知的 energy 子命令: '{args.subcommand}'")


def _cmd_energy_surface(args):
    """
    表面能计算 / Surface energy calculation.

    默认从单质库读取元素化学势，计算不按面积归一的表面过剩能。
    如果显式提供 --bulk，则保留旧的 bulk/area 表面能计算方式。
    """
    if args.bulk:
        _cmd_energy_surface_legacy(args)
        return

    from matkit.energy import calc_surface_excess_energy_from_directory

    print(color_header(">>> 表面过剩能计算 / Surface Excess Energy Calculation"))
    print(f"  板模型路径:      {args.slab}")
    print(f"  单质库路径:      {args.reference_db}")
    print(f"  表面数量:        {args.n_surfaces}")
    print()

    try:
        result = calc_surface_excess_energy_from_directory(
            slab_path=args.slab,
            simple_substance_db=args.reference_db,
            n_surfaces=args.n_surfaces,
        )

        print(color_header("=== 表面过剩能结果 / Surface Excess Energy Results ==="))
        print()
        print_kv("Case", result["label"])
        if result["task_number"] is not None:
            print_kv("Task number", result["task_number"])
        print_kv("E_slab", f"{result['E_slab']:.6f} eV")
        print_kv("Reference energy", f"{result['reference_energy_eV']:.6f} eV")
        print_kv("Excess energy", f"{result['excess_energy_eV']:.6f} eV")
        print_kv(
            "Surface excess energy",
            f"{result['surface_excess_energy_eV_per_surface']:.6f} eV/surface",
        )
        print()
        print(color_header("元素参考项 / Element Reference Terms"))
        rows = [
            [
                term["element"],
                term["count"],
                f"{term['mu_eV_per_atom']:.6f}",
                f"{term['reference_energy_eV']:.6f}",
            ]
            for term in result["reference_terms"]
        ]
        print_table(
            ["Element", "Count", "mu (eV/atom)", "n*mu (eV)"],
            rows,
        )

        if args.csv:
            headers = ["Property", "Value", "Unit"]
            rows = [
                ["label", result["label"], ""],
                ["task_number", result["task_number"] or "", ""],
                ["E_slab", f"{result['E_slab']:.6f}", "eV"],
                ["reference_energy", f"{result['reference_energy_eV']:.6f}", "eV"],
                ["excess_energy", f"{result['excess_energy_eV']:.6f}", "eV"],
                [
                    "surface_excess_energy",
                    f"{result['surface_excess_energy_eV_per_surface']:.6f}",
                    "eV/surface",
                ],
                ["n_surfaces", str(result["n_surfaces"]), ""],
            ]
            for term in result["reference_terms"]:
                rows.append([
                    f"mu_{term['element']}",
                    f"{term['mu_eV_per_atom']:.6f}",
                    "eV/atom",
                ])
                rows.append([
                    f"count_{term['element']}",
                    str(term["count"]),
                    "",
                ])
            save_csv(args.csv, headers, rows)

        if args.json:
            save_json(args.json, result)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_energy_surface_legacy(args):
    """
    Legacy area-normalised surface energy calculation.
    """
    from matkit.parsers import read_poscar
    from matkit.energy import calc_surface_energy, calc_surface_energies_from_files

    if args.n_bulk is None:
        handle_error("使用 --bulk 旧表面能模式时必须同时提供 --n-bulk")

    print(color_header(">>> 传统表面能计算 / Legacy Surface Energy Calculation"))
    print(f"  板模型 OUTCAR: {args.slab}")
    print(f"  体相 OUTCAR:   {args.bulk}")
    print(f"  体相原子数:   {args.n_bulk}")
    print()

    try:
        # 读取能量 / Read energies
        energy_data = calc_surface_energies_from_files(
            slab_outcar=args.slab,
            bulk_outcar=args.bulk,
            n_bulk_atoms=args.n_bulk,
        )

        E_slab = energy_data['E_slab']
        E_bulk_per_atom = energy_data['E_bulk_per_atom']

        # 尝试从 POSCAR 读取板模型原子数和表面积
        # Try to read slab atom count and surface area from POSCAR
        slab_poscar = args.slab.replace("OUTCAR", "POSCAR").replace("outcar", "POSCAR")
        n_atoms_slab = None
        surface_area = None

        if os.path.isfile(slab_poscar):
            try:
                poscar_data = read_poscar(slab_poscar)
                n_atoms_slab = poscar_data['total_atoms']
                lattice = poscar_data['lattice']
                from matkit.structure import calc_surface_area
                area_result = calc_surface_area(lattice)
                surface_area = area_result['surface_area_A2']
                print(f"  从 POSCAR 读取: {slab_poscar}")
                print(f"  板模型原子数: {n_atoms_slab}")
                print(f"  表面积: {surface_area:.4f} A^2")
            except Exception as e:
                print(f"  {color_warning(f'POSCAR 解析失败，无法计算表面积: {e}')}")
        else:
            print(f"  {color_warning(f'未找到 POSCAR 文件 ({slab_poscar})，无法自动计算表面积')}")

        print()

        # 如果有表面积，计算完整表面能 / If surface area available, compute full surface energy
        if n_atoms_slab is not None and surface_area is not None:
            result = calc_surface_energy(
                E_slab=E_slab,
                E_bulk_per_atom=E_bulk_per_atom,
                n_atoms_slab=n_atoms_slab,
                surface_area=surface_area,
            )

            print(color_header("=== 表面能结果 / Surface Energy Results ==="))
            print()
            print_kv("E_slab (板模型能量)", f"{E_slab:.6f} eV")
            print_kv("E_bulk (体相总能量)", f"{energy_data['E_bulk']:.6f} eV")
            print_kv("E_bulk/atom (体相每原子能量)", f"{E_bulk_per_atom:.6f} eV")
            print_kv("N_atoms (板模型原子数)", n_atoms_slab)
            print_kv("Surface area (表面积)", f"{surface_area:.4f} A^2")
            print()
            print_kv("Excess energy (多余能量)", f"{result['excess_energy']:.6f} eV")
            print_kv("Surface energy", f"{result['surface_energy_eV_A2']:.6f} eV/A^2")
            print_kv("Surface energy", f"{result['surface_energy_J_m2']:.4f} J/m^2")

            # CSV 导出 / CSV export
            if args.csv:
                headers = ["Property", "Value", "Unit"]
                rows = [
                    ["E_slab", f"{E_slab:.6f}", "eV"],
                    ["E_bulk", f"{energy_data['E_bulk']:.6f}", "eV"],
                    ["E_bulk_per_atom", f"{E_bulk_per_atom:.6f}", "eV"],
                    ["N_atoms_slab", str(n_atoms_slab), ""],
                    ["Surface_area", f"{surface_area:.4f}", "A^2"],
                    ["Excess_energy", f"{result['excess_energy']:.6f}", "eV"],
                    ["Surface_energy_eV_A2", f"{result['surface_energy_eV_A2']:.6f}", "eV/A^2"],
                    ["Surface_energy_J_m2", f"{result['surface_energy_J_m2']:.4f}", "J/m^2"],
                ]
                save_csv(args.csv, headers, rows)

            # JSON 导出 / JSON export
            if args.json:
                save_json(args.json, result)
        else:
            # 仅显示能量部分 / Only show energy part
            print(color_header("=== 能量数据 / Energy Data ==="))
            print()
            print_kv("E_slab (板模型能量)", f"{E_slab:.6f} eV")
            print_kv("E_bulk (体相总能量)", f"{energy_data['E_bulk']:.6f} eV")
            print_kv("E_bulk/atom (体相每原子能量)", f"{E_bulk_per_atom:.6f} eV")
            print()
            print(color_warning("提示: 需要提供 POSCAR 文件以计算完整表面能"))
            print(color_warning("Tip: POSCAR file needed for full surface energy calculation"))

            if args.csv:
                headers = ["Property", "Value", "Unit"]
                rows = [
                    ["E_slab", f"{E_slab:.6f}", "eV"],
                    ["E_bulk", f"{energy_data['E_bulk']:.6f}", "eV"],
                    ["E_bulk_per_atom", f"{E_bulk_per_atom:.6f}", "eV"],
                ]
                save_csv(args.csv, headers, rows)

            if args.json:
                save_json(args.json, energy_data)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_energy_surface_batch(args):
    """
    Batch surface excess energy calculation from a directory tree.
    """
    from matkit.energy import calc_surface_excess_energies_batch

    print(color_header(">>> 批量表面过剩能计算 / Batch Surface Excess Energy Calculation"))
    print(f"  根目录:          {args.root}")
    print(f"  单质库路径:      {args.reference_db}")
    print(f"  表面数量:        {args.n_surfaces}")
    print()

    try:
        results = calc_surface_excess_energies_batch(
            root_dir=args.root,
            simple_substance_db=args.reference_db,
            n_surfaces=args.n_surfaces,
            strict=args.strict,
        )

        ok_count = sum(1 for row in results if row.get("status") == "ok")
        error_count = len(results) - ok_count
        print_kv("Total cases", len(results))
        print_kv("OK", ok_count)
        print_kv("Errors", error_count)
        print()

        table_rows = []
        for row in results:
            if row.get("status") == "ok":
                table_rows.append([
                    row["label"],
                    row.get("task_number") or "",
                    row.get("task_suffix") or "",
                    "ok",
                    f"{row['surface_excess_energy_eV_per_surface']:.6f}",
                    f"{row['excess_energy_eV']:.6f}",
                ])
            else:
                table_rows.append([
                    row["label"],
                    row.get("task_number") or "",
                    row.get("task_suffix") or "",
                    "error",
                    "",
                    row.get("error", ""),
                ])

        if table_rows:
            print_table(
                ["Label", "Task", "Suffix", "Status", "E/surface", "Excess/Error"],
                table_rows,
            )
        else:
            print(color_warning("未发现包含 POSCAR 和能量文件的计算目录"))

        if args.csv:
            headers = [
                "label",
                "task_number",
                "task_suffix",
                "status",
                "surface_excess_energy_eV_per_surface",
                "excess_energy_eV",
                "E_slab",
                "reference_energy_eV",
                "n_surfaces",
                "composition",
                "slab_dir",
                "energy_file",
                "structure_file",
                "error",
            ]
            csv_rows = []
            for row in results:
                csv_rows.append([
                    row.get("label", ""),
                    row.get("task_number") or "",
                    row.get("task_suffix") or "",
                    row.get("status", ""),
                    row.get("surface_excess_energy_eV_per_surface", ""),
                    row.get("excess_energy_eV", ""),
                    row.get("E_slab", ""),
                    row.get("reference_energy_eV", ""),
                    row.get("n_surfaces", ""),
                    json.dumps(row.get("composition", {}), ensure_ascii=False),
                    row.get("slab_dir", ""),
                    row.get("energy_file", ""),
                    row.get("structure_file", ""),
                    row.get("error", ""),
                ])
            save_csv(args.csv, headers, csv_rows)

        if args.json:
            save_json(args.json, results)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_energy_adsorption(args):
    """
    吸附能计算 / Adsorption energy calculation.
    """
    from matkit.energy import calc_adsorption_energy

    print(color_header(">>> 吸附能计算 / Adsorption Energy Calculation"))
    print(f"  吸附体系 OUTCAR: {args.system}")
    print(f"  清洁表面 OUTCAR: {args.slab}")
    print(f"  吸附物 OUTCAR:   {args.adsorbate}")
    print(f"  吸附物数量:     {args.n_ads}")
    print()

    try:
        # 读取能量 / Read energies
        from matkit.parsers import read_outcar

        system_data = read_outcar(args.system)
        slab_data = read_outcar(args.slab)
        ads_data = read_outcar(args.adsorbate)

        E_slab_ads = system_data['final_energy']
        E_slab_clean = slab_data['final_energy']
        E_adsorbate = ads_data['final_energy']

        if E_slab_ads is None:
            handle_error(f"无法从 {args.system} 中解析总能量")
        if E_slab_clean is None:
            handle_error(f"无法从 {args.slab} 中解析总能量")
        if E_adsorbate is None:
            handle_error(f"无法从 {args.adsorbate} 中解析总能量")

        result = calc_adsorption_energy(
            E_slab_ads=E_slab_ads,
            E_slab_clean=E_slab_clean,
            E_adsorbate=E_adsorbate,
            n_adsorbate=args.n_ads,
        )

        print(color_header("=== 吸附能结果 / Adsorption Energy Results ==="))
        print()
        print_kv("E_slab+ads (吸附体系能量)", f"{E_slab_ads:.6f} eV")
        print_kv("E_slab_clean (清洁表面能量)", f"{E_slab_clean:.6f} eV")
        print_kv("E_adsorbate (吸附物能量)", f"{E_adsorbate:.6f} eV")
        print_kv("n_adsorbate (吸附物数量)", args.n_ads)
        print_kv("E_adsorbate_total (吸附物总能量)", f"{result['E_adsorbate_total']:.6f} eV")
        print()
        print_kv("E_ads (总吸附能)", f"{result['adsorption_energy_eV']:.6f} eV")
        print_kv("E_ads/molecule (每分子吸附能)", f"{result['adsorption_energy_per_molecule']:.6f} eV")

        if result['adsorption_energy_eV'] < 0:
            print(f"\n  {color_success('吸附过程放热 (有利) / Exothermic (favorable)')}")
        else:
            print(f"\n  {color_warning('吸附过程吸热 (不利) / Endothermic (unfavorable)')}")

        if args.csv:
            headers = ["Property", "Value", "Unit"]
            rows = [
                ["E_slab_ads", f"{E_slab_ads:.6f}", "eV"],
                ["E_slab_clean", f"{E_slab_clean:.6f}", "eV"],
                ["E_adsorbate", f"{E_adsorbate:.6f}", "eV"],
                ["n_adsorbate", str(args.n_ads), ""],
                ["E_ads_total", f"{result['adsorption_energy_eV']:.6f}", "eV"],
                ["E_ads_per_molecule", f"{result['adsorption_energy_per_molecule']:.6f}", "eV"],
            ]
            save_csv(args.csv, headers, rows)

        if args.json:
            save_json(args.json, result)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_energy_doping(args):
    """
    掺杂能 (缺陷形成能) 计算 / Doping energy (defect formation energy) calculation.
    """
    from matkit.parsers import read_outcar
    from matkit.energy import calc_doping_energy

    print(color_header(">>> 掺杂能计算 / Doping Energy Calculation"))
    print(f"  掺杂体系 OUTCAR: {args.doped}")
    print(f"  未掺杂 OUTCAR:   {args.pristine}")
    print(f"  掺杂原子数:      {args.n_dopant}")
    print(f"  掺杂原子化学势:  {args.mu_dopant} eV")
    print(f"  宿主原子化学势:  {args.mu_host} eV")
    print(f"  电荷态:          {args.charge}")
    print(f"  费米能级:        {args.efermi} eV")
    print()

    try:
        doped_data = read_outcar(args.doped)
        pristine_data = read_outcar(args.pristine)

        E_doped = doped_data['final_energy']
        E_pristine = pristine_data['final_energy']

        if E_doped is None:
            handle_error(f"无法从 {args.doped} 中解析总能量")
        if E_pristine is None:
            handle_error(f"无法从 {args.pristine} 中解析总能量")

        result = calc_doping_energy(
            E_doped=E_doped,
            E_pristine=E_pristine,
            n_dopant=args.n_dopant,
            mu_dopant=args.mu_dopant,
            mu_host=args.mu_host,
            charge_state=args.charge,
            efermi=args.efermi,
        )

        print(color_header("=== 掺杂能结果 / Doping Energy Results ==="))
        print()
        print_kv("E_doped (掺杂体系能量)", f"{E_doped:.6f} eV")
        print_kv("E_pristine (未掺杂能量)", f"{E_pristine:.6f} eV")
        print_kv("Energy diff (E_doped - E_pristine)", f"{result['energy_diff']:.6f} eV")
        print_kv("n_dopant (掺杂原子数)", args.n_dopant)
        print_kv("mu_dopant (掺杂原子化学势)", f"{args.mu_dopant} eV")
        print_kv("mu_host (宿主原子化学势)", f"{args.mu_host} eV")
        print_kv("Chemical potential term", f"{result['chemical_potential_term']:.6f} eV")
        print_kv("Charge state (电荷态)", args.charge)
        print_kv("E_fermi (费米能级)", f"{args.efermi} eV")
        print_kv("Charge correction term", f"{result['charge_correction_term']:.6f} eV")
        print()
        print_kv("E_f (缺陷形成能)", f"{result['formation_energy_eV']:.6f} eV")

        if args.csv:
            headers = ["Property", "Value", "Unit"]
            rows = [
                ["E_doped", f"{E_doped:.6f}", "eV"],
                ["E_pristine", f"{E_pristine:.6f}", "eV"],
                ["Energy_diff", f"{result['energy_diff']:.6f}", "eV"],
                ["n_dopant", str(args.n_dopant), ""],
                ["mu_dopant", f"{args.mu_dopant}", "eV"],
                ["mu_host", f"{args.mu_host}", "eV"],
                ["Chemical_potential_term", f"{result['chemical_potential_term']:.6f}", "eV"],
                ["Charge_state", str(args.charge), ""],
                ["E_fermi", f"{args.efermi}", "eV"],
                ["Charge_correction_term", f"{result['charge_correction_term']:.6f}", "eV"],
                ["Formation_energy", f"{result['formation_energy_eV']:.6f}", "eV"],
            ]
            save_csv(args.csv, headers, rows)

        if args.json:
            save_json(args.json, result)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def add_energy_subparser(subparsers):
    """添加 energy 子命令解析器 / Add energy subcommand parser."""
    parser = subparsers.add_parser(
        "energy",
        help="能量计算 (表面能、吸附能、掺杂能)",
        description="能量计算模块 / Energy calculation module",
    )
    subparsers_energy = parser.add_subparsers(dest="subcommand", help="能量计算子命令")

    # energy surface
    p_surface = subparsers_energy.add_parser(
        "surface",
        help="计算表面能 / Calculate surface energy",
        description=(
            "默认从单质库读取元素化学势，计算不按面积归一的表面过剩能；"
            "提供 --bulk 时使用旧的面积归一表面能公式"
        ),
    )
    p_surface.add_argument("--slab", required=True, help="板模型计算目录或 OUTCAR/log 文件路径 / Slab calculation dir or energy file")
    p_surface.add_argument("--bulk", help="旧模式: 体相 OUTCAR 文件路径 / Legacy mode bulk OUTCAR path")
    p_surface.add_argument("--n-bulk", type=int, help="旧模式: 体相计算中的原子数 / Legacy number of bulk atoms")
    p_surface.add_argument("--reference-db", default="simple_substance_database", help="单质库路径 (默认: simple_substance_database) / Simple-substance reference database")
    p_surface.add_argument("--n-surfaces", type=int, default=2, help="表面数量 (默认: 2) / Number of exposed surfaces")
    p_surface.add_argument("--csv", help="导出 CSV 文件路径 / CSV output path")
    p_surface.add_argument("--json", help="导出 JSON 文件路径 / JSON output path")

    # energy surface-batch
    p_surface_batch = subparsers_energy.add_parser(
        "surface-batch",
        help="批量计算表面过剩能 / Batch-calculate surface excess energies",
        description="扫描根目录下所有包含 POSCAR 和 OUTCAR/OSZICAR/log 的计算文件夹",
    )
    p_surface_batch.add_argument("root", help="表面能计算结果根目录，如 Surface_energy / Surface calculation root")
    p_surface_batch.add_argument("--reference-db", default="simple_substance_database", help="单质库路径 (默认: simple_substance_database) / Simple-substance reference database")
    p_surface_batch.add_argument("--n-surfaces", type=int, default=2, help="表面数量 (默认: 2) / Number of exposed surfaces")
    p_surface_batch.add_argument("--strict", action="store_true", help="遇到单个错误时立即停止 / Stop on the first case error")
    p_surface_batch.add_argument("--csv", help="导出 CSV 文件路径 / CSV output path")
    p_surface_batch.add_argument("--json", help="导出 JSON 文件路径 / JSON output path")

    # energy adsorption
    p_ads = subparsers_energy.add_parser(
        "adsorption",
        help="计算吸附能 / Calculate adsorption energy",
        description="从 OUTCAR 文件计算吸附能 / Calculate adsorption energy from OUTCAR files",
    )
    p_ads.add_argument("--system", required=True, help="吸附体系 OUTCAR 文件路径 / Adsorbed system OUTCAR path")
    p_ads.add_argument("--slab", required=True, help="清洁表面 OUTCAR 文件路径 / Clean slab OUTCAR path")
    p_ads.add_argument("--adsorbate", required=True, help="吸附物 OUTCAR 文件路径 / Adsorbate OUTCAR path")
    p_ads.add_argument("--n-ads", type=int, default=1, help="吸附物分子数量 (默认: 1) / Number of adsorbate molecules (default: 1)")
    p_ads.add_argument("--csv", help="导出 CSV 文件路径 / CSV output path")
    p_ads.add_argument("--json", help="导出 JSON 文件路径 / JSON output path")

    # energy doping
    p_doping = subparsers_energy.add_parser(
        "doping",
        help="计算掺杂能 (缺陷形成能) / Calculate doping energy (defect formation energy)",
        description="计算掺杂缺陷形成能 / Calculate doping defect formation energy",
    )
    p_doping.add_argument("--doped", required=True, help="掺杂体系 OUTCAR 文件路径 / Doped system OUTCAR path")
    p_doping.add_argument("--pristine", required=True, help="未掺杂体系 OUTCAR 文件路径 / Pristine system OUTCAR path")
    p_doping.add_argument("--n-dopant", type=int, required=True, help="掺入的杂质原子数 / Number of dopant atoms")
    p_doping.add_argument("--mu-dopant", type=float, required=True, help="杂质原子的化学势 (eV) / Dopant chemical potential (eV)")
    p_doping.add_argument("--mu-host", type=float, required=True, help="宿主原子的化学势 (eV) / Host chemical potential (eV)")
    p_doping.add_argument("--charge", type=int, default=0, help="缺陷电荷态 (默认: 0) / Defect charge state (default: 0)")
    p_doping.add_argument("--efermi", type=float, default=0.0, help="费米能级 (eV, 默认: 0.0) / Fermi level (eV, default: 0.0)")
    p_doping.add_argument("--csv", help="导出 CSV 文件路径 / CSV output path")
    p_doping.add_argument("--json", help="导出 JSON 文件路径 / JSON output path")


# ============================================================================
# 子命令: structure
# ============================================================================

def cmd_structure(args):
    """结构分析子命令 / Structure analysis subcommand."""
    if args.subcommand == "info":
        _cmd_structure_info(args)
    elif args.subcommand == "geometry":
        _cmd_structure_geometry(args)
    elif args.subcommand == "molecules":
        _cmd_structure_molecules(args)
    elif args.subcommand == "surface":
        _cmd_structure_surface(args)
    elif args.subcommand == "relaxation":
        _cmd_structure_relaxation(args)
    else:
        handle_error(f"未知的 structure 子命令: '{args.subcommand}'")


def _cmd_structure_info(args):
    """
    结构信息摘要 / Structure information summary.
    """
    from matkit.parsers import read_poscar

    print(color_header(">>> 结构信息 / Structure Info"))
    print(f"  文件: {args.poscar}")
    print()

    try:
        data = read_poscar(args.poscar)

        print(color_header("=== 基本信息 / Basic Information ==="))
        print()
        print_kv("Comment (注释)", data['comment'])
        print_kv("Scale (缩放因子)", data['scale'])
        print_kv("Total atoms (总原子数)", data['total_atoms'])
        print_kv("Elements (元素)", ", ".join(data['elements']))
        print_kv("Atom counts (原子数)", str(data['n_atoms']))
        print_kv("Coord type (坐标类型)", "Direct" if data['is_direct'] else "Cartesian")

        print()
        print(color_header("=== 晶格向量 / Lattice Vectors ==="))
        print()
        lattice = data['lattice']
        for i, label in enumerate(["a", "b", "c"]):
            vec = lattice[i]
            length = np.linalg.norm(vec)
            print(f"  {label}: [{vec[0]:10.6f}, {vec[1]:10.6f}, {vec[2]:10.6f}]  |{label}| = {length:.6f} A")

        a_len = np.linalg.norm(lattice[0])
        b_len = np.linalg.norm(lattice[1])
        c_len = np.linalg.norm(lattice[2])
        alpha = np.degrees(np.arccos(np.dot(lattice[1], lattice[2]) / (b_len * c_len)))
        beta = np.degrees(np.arccos(np.dot(lattice[0], lattice[2]) / (a_len * c_len)))
        gamma = np.degrees(np.arccos(np.dot(lattice[0], lattice[1]) / (a_len * b_len)))

        print()
        print_kv("a", f"{a_len:.6f} A")
        print_kv("b", f"{b_len:.6f} A")
        print_kv("c", f"{c_len:.6f} A")
        print_kv("alpha", f"{alpha:.2f} deg")
        print_kv("beta", f"{beta:.2f} deg")
        print_kv("gamma", f"{gamma:.2f} deg")

        volume = abs(np.linalg.det(lattice))
        print_kv("Volume (体积)", f"{volume:.4f} A^3")

        # 元素统计 / Element statistics
        print()
        print(color_header("=== 元素统计 / Element Statistics ==="))
        print()
        elem_rows = []
        for elem in data['elements']:
            count = data['n_atoms'][elem]
            elem_rows.append([elem, str(count), f"{count / data['total_atoms'] * 100:.1f}%"])
        print_table(["Element", "Count", "Fraction"], elem_rows)

        # 坐标范围 / Coordinate ranges
        coords = data['coords']
        print()
        print(color_header("=== 坐标范围 / Coordinate Ranges (Angstrom) ==="))
        print()
        for dim, label in enumerate(["X", "Y", "Z"]):
            vals = coords[:, dim]
            print_kv(f"{label} range", f"[{vals.min():.6f}, {vals.max():.6f}]")

        # 输出到文件 / Output to file
        if args.output:
            filepath = os.path.abspath(args.output)
            output_dir = os.path.dirname(filepath)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"MatKit Structure Info Summary\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"File: {args.poscar}\n")
                f.write(f"Comment: {data['comment']}\n")
                f.write(f"Total atoms: {data['total_atoms']}\n")
                f.write(f"Elements: {', '.join(data['elements'])}\n")
                f.write(f"Atom counts: {data['n_atoms']}\n\n")
                f.write(f"Lattice parameters:\n")
                f.write(f"  a = {a_len:.6f} A\n")
                f.write(f"  b = {b_len:.6f} A\n")
                f.write(f"  c = {c_len:.6f} A\n")
                f.write(f"  alpha = {alpha:.2f} deg\n")
                f.write(f"  beta = {beta:.2f} deg\n")
                f.write(f"  gamma = {gamma:.2f} deg\n")
                f.write(f"  Volume = {volume:.4f} A^3\n")
            print(f"\n{color_success('摘要已保存 / Summary saved:')} {filepath}")

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_structure_geometry(args):
    """
    几何计算 (键长、键角) / Geometry calculation (bond lengths, angles).
    """
    from matkit.parsers import read_poscar
    from matkit.structure import calc_distance, calc_angle

    print(color_header(">>> 几何计算 / Geometry Calculation"))
    print(f"  POSCAR: {args.poscar}")
    print()

    try:
        data = read_poscar(args.poscar)
        coords = data['coords']
        atom_elements = atom_elements_from_poscar_data(data)

        # 解析原子对 / Parse atom pairs
        pairs = parse_pairs_string(args.pairs)

        print(color_header("=== 键长 / Bond Lengths ==="))
        print()
        bond_rows = []
        bond_results = []
        for pair in pairs:
            i, j = pair
            if i >= data['total_atoms'] or j >= data['total_atoms']:
                msg = f'索引 {i},{j} 超出范围 (共 {data["total_atoms"]} 个原子)'
                print(f"  {color_warning(msg)}")
                continue
            dist = calc_distance(coords[i], coords[j])
            elem_i = atom_elements[i] if i < len(atom_elements) else "?"
            elem_j = atom_elements[j] if j < len(atom_elements) else "?"
            bond_rows.append([str(i), elem_i, str(j), elem_j, f"{dist:.6f}"])
            bond_results.append({"pair": [i, j], "elements": [elem_i, elem_j], "distance": dist})

        print_table(["Index i", "Elem", "Index j", "Elem", "Distance (A)"], bond_rows)

        # 键角计算 / Angle calculation
        if args.angles:
            print()
            print(color_header("=== 键角 / Bond Angles ==="))
            print()
            angle_groups = args.angles.split(";")
            angle_rows = []
            for group in angle_groups:
                group = group.strip()
                if not group:
                    continue
                indices = [int(x.strip()) for x in group.split(",")]
                if len(indices) != 3:
                    msg = f'无效的角度格式: "{group}"，应为 "i,j,k"'
                    print(f"  {color_warning(msg)}")
                    continue
                i, j, k = indices
                if max(i, j, k) >= data['total_atoms']:
                    msg = f'索引超出范围 (共 {data["total_atoms"]} 个原子)'
                    print(f"  {color_warning(msg)}")
                    continue
                angle = calc_angle(coords[i], coords[j], coords[k])
                elems = [atom_elements[idx] if idx < len(atom_elements) else "?" for idx in [i, j, k]]
                angle_rows.append([f"{i}({elems[0]})", f"{j}({elems[1]})", f"{k}({elems[2]})", f"{angle:.4f}"])

            print_table(["Atom i", "Atom j (vertex)", "Atom k", "Angle (deg)"], angle_rows)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_structure_molecules(args):
    """
    分子识别 / Molecule identification.
    """
    from matkit.parsers import read_poscar
    from matkit.structure import identify_molecules, match_molecule

    print(color_header(">>> 分子识别 / Molecule Identification"))
    print(f"  POSCAR: {args.poscar}")
    print(f"  键长截断值: {args.cutoffs}")
    if args.formula:
        print(f"  目标化学式: {args.formula}")
    print()

    try:
        data = read_poscar(args.poscar)
        cutoffs = parse_cutoffs_string(args.cutoffs)

        elements = atom_elements_from_poscar_data(data)

        molecules = identify_molecules(elements, data['coords'], cutoffs)

        print(color_header(f"=== 识别到 {len(molecules)} 个分子 / {len(molecules)} Molecule(s) Found ==="))
        print()

        mol_rows = []
        for i, mol in enumerate(molecules):
            mol_rows.append([
                str(i),
                mol['formula'],
                str(len(mol['indices'])),
                str(mol['indices'][:10]) + ("..." if len(mol['indices']) > 10 else ""),
            ])
        print_table(["#", "Formula", "N_atoms", "Indices"], mol_rows)

        # 化学式匹配 / Formula matching
        if args.formula:
            print()
            print(color_header(f"=== 匹配化学式: {args.formula} ==="))
            print()
            matched = match_molecule(molecules, args.formula)
            if matched:
                print_kv("匹配成功 / Match found", f"分子 #{molecules.index(matched)}")
                print_kv("化学式 / Formula", matched['formula'])
                print_kv("原子数 / N_atoms", len(matched['indices']))
                print_kv("原子索引 / Indices", str(matched['indices']))
                # 计算质心 / Calculate center of mass
                com = np.mean(matched['coords'], axis=0)
                print_kv("质心 / Center of mass", f"[{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}]")
            else:
                print(f"  {color_warning(f'未找到匹配化学式 {args.formula} 的分子')}")

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_structure_surface(args):
    """
    表面层分析 / Surface layer analysis.
    """
    from matkit.parsers import read_poscar
    from matkit.structure import identify_surface_layers

    print(color_header(">>> 表面层分析 / Surface Layer Analysis"))
    print(f"  POSCAR: {args.poscar}")
    print(f"  表面元素: {args.element}")
    print(f"  z 坐标容差: {args.tolerance} A")
    print()

    try:
        data = read_poscar(args.poscar)

        elements = atom_elements_from_poscar_data(data)

        layers = identify_surface_layers(elements, data['coords'], args.element, args.tolerance)

        print(color_header(f"=== 识别到 {len(layers)} 个 {args.element} 层 / {len(layers)} {args.element} Layer(s) ==="))
        print()

        layer_rows = []
        for i, layer in enumerate(layers):
            layer_rows.append([
                str(i),
                str(len(layer['indices'])),
                f"{layer['z_avg']:.4f}",
                f"{layer['z_min']:.4f}",
                f"{layer['z_max']:.4f}",
                str(layer['indices'][:8]) + ("..." if len(layer['indices']) > 8 else ""),
            ])
        print_table(
            ["Layer", "N_atoms", "Z_avg (A)", "Z_min (A)", "Z_max (A)", "Indices"],
            layer_rows,
        )

        if len(layers) >= 2:
            print()
            print_kv("层间距 (Layer spacing)",
                     f"{layers[-1]['z_avg'] - layers[-2]['z_avg']:.4f} A")

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_structure_relaxation(args):
    """
    表面弛豫分析 / Surface relaxation analysis.
    """
    from matkit.parsers import read_poscar
    from matkit.structure import identify_surface_layers, analyze_surface_relaxation

    print(color_header(">>> 表面弛豫分析 / Surface Relaxation Analysis"))
    print(f"  清洁表面 POSCAR: {args.poscar_clean}")
    print(f"  吸附后 POSCAR:   {args.poscar_adsorbed}")
    print(f"  表面元素:        {args.element}")
    print(f"  弛豫阈值:        {args.threshold} A")
    print()

    try:
        clean_data = read_poscar(args.poscar_clean)
        ads_data = read_poscar(args.poscar_adsorbed)

        clean_elements = atom_elements_from_poscar_data(clean_data)

        # 识别表面原子 / Identify surface atoms
        layers = identify_surface_layers(clean_elements, clean_data['coords'], args.element)
        if not layers:
            handle_error(f"未找到元素 '{args.element}' 的表面层")

        # 使用最外层作为表面原子 / Use outermost layer as surface atoms
        surface_indices = layers[-1]['indices']

        print_kv("表面原子数", len(surface_indices))
        print_kv("表面层 Z_avg", f"{layers[-1]['z_avg']:.4f} A")
        print()

        # 分析弛豫 / Analyze relaxation
        result = analyze_surface_relaxation(
            clean_data['coords'],
            ads_data['coords'],
            surface_indices,
            tolerance=args.threshold,
        )

        print(color_header("=== 弛豫分析结果 / Relaxation Results ==="))
        print()
        print_kv("弛豫原子数 / N_relaxed", result['n_relaxed'])
        print_kv("未弛豫原子数 / N_unrelaxed", result['n_unrelaxed'])
        print_kv("平均位移 / Avg displacement", f"{result['avg_displacement']:.6f} A")
        print_kv("最大位移 / Max displacement", f"{result['max_displacement']:.6f} A")

        if result['relaxed_atoms']:
            print()
            print(color_header("=== 弛豫原子详情 / Relaxed Atom Details ==="))
            print()
            relaxed_rows = []
            for atom in result['relaxed_atoms']:
                elem = clean_elements[atom['index']] if atom['index'] < len(clean_elements) else "?"
                relaxed_rows.append([
                    str(atom['index']),
                    elem,
                    f"{atom['displacement']:.6f}",
                    f"{atom['dx']:.6f}",
                    f"{atom['dy']:.6f}",
                    f"{atom['dz']:.6f}",
                ])
            print_table(
                ["Index", "Element", "|d| (A)", "dx (A)", "dy (A)", "dz (A)"],
                relaxed_rows,
            )

        if args.csv:
            headers = ["Index", "Element", "Displacement", "dx", "dy", "dz"]
            rows = []
            for atom in result['all_displacements']:
                elem = clean_elements[atom['index']] if atom['index'] < len(clean_elements) else "?"
                rows.append([
                    atom['index'], elem,
                    f"{atom['displacement']:.6f}",
                    f"{atom['dx']:.6f}",
                    f"{atom['dy']:.6f}",
                    f"{atom['dz']:.6f}",
                ])
            save_csv(args.csv, headers, rows)

        if args.json:
            save_json(args.json, result)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def add_structure_subparser(subparsers):
    """添加 structure 子命令解析器 / Add structure subcommand parser."""
    parser = subparsers.add_parser(
        "structure",
        help="结构分析 (几何、分子识别、表面分析)",
        description="结构分析模块 / Structure analysis module",
    )
    subparsers_struct = parser.add_subparsers(dest="subcommand", help="结构分析子命令")

    # structure info
    p_info = subparsers_struct.add_parser(
        "info",
        help="显示结构信息摘要 / Show structure info summary",
        description="显示 POSCAR/CONTCAR 文件的结构信息摘要",
    )
    p_info.add_argument("poscar", help="POSCAR/CONTCAR 文件路径 / POSCAR/CONTCAR file path")
    p_info.add_argument("--output", help="输出摘要文件路径 / Output summary file path")

    # structure geometry
    p_geom = subparsers_struct.add_parser(
        "geometry",
        help="计算键长和键角 / Calculate bond lengths and angles",
        description="计算指定原子对之间的键长和键角",
    )
    p_geom.add_argument("poscar", help="POSCAR/CONTCAR 文件路径 / POSCAR/CONTCAR file path")
    p_geom.add_argument("--pairs", required=True, help='原子对，格式: "1,2;3,4;5,6" / Atom pairs, format: "1,2;3,4;5,6"')
    p_geom.add_argument("--angles", help='键角，格式: "1,2,3;4,5,6" / Bond angles, format: "1,2,3;4,5,6"')

    # structure molecules
    p_mol = subparsers_struct.add_parser(
        "molecules",
        help="识别分子 / Identify molecules",
        description="基于键长截断值识别体系中的分子",
    )
    p_mol.add_argument("poscar", help="POSCAR/CONTCAR 文件路径 / POSCAR/CONTCAR file path")
    p_mol.add_argument("--cutoffs", required=True, help='键长截断值，格式: "O-Cu:2.5,S-O:2.0" / Bond cutoffs, format: "O-Cu:2.5,S-O:2.0"')
    p_mol.add_argument("--formula", help="目标化学式 (如 SO4) / Target chemical formula (e.g., SO4)")

    # structure surface
    p_surf = subparsers_struct.add_parser(
        "surface",
        help="表面层分析 / Surface layer analysis",
        description="识别表面层并分析层结构",
    )
    p_surf.add_argument("poscar", help="POSCAR/CONTCAR 文件路径 / POSCAR/CONTCAR file path")
    p_surf.add_argument("--element", required=True, help="表面元素符号 (如 Cu) / Surface element symbol (e.g., Cu)")
    p_surf.add_argument("--tolerance", type=float, default=0.5, help="z 坐标聚类容差 (A, 默认: 0.5) / Z-coordinate clustering tolerance (A, default: 0.5)")

    # structure relaxation
    p_relax = subparsers_struct.add_parser(
        "relaxation",
        help="表面弛豫分析 / Surface relaxation analysis",
        description="比较清洁表面和吸附后表面的原子位移",
    )
    p_relax.add_argument("poscar_clean", help="清洁表面 POSCAR 文件路径 / Clean surface POSCAR path")
    p_relax.add_argument("poscar_adsorbed", help="吸附后 POSCAR 文件路径 / Adsorbed POSCAR path")
    p_relax.add_argument("--element", required=True, help="表面元素符号 (如 Cu) / Surface element symbol (e.g., Cu)")
    p_relax.add_argument("--threshold", type=float, default=0.1, help="弛豫判定阈值 (A, 默认: 0.1) / Relaxation threshold (A, default: 0.1)")
    p_relax.add_argument("--csv", help="导出 CSV 文件路径 / CSV output path")
    p_relax.add_argument("--json", help="导出 JSON 文件路径 / JSON output path")


# ============================================================================
# 子命令: charge
# ============================================================================

def cmd_charge(args):
    """电荷分析子命令 / Charge analysis subcommand."""
    if args.subcommand == "diff":
        _cmd_charge_diff(args)
    elif args.subcommand == "planar":
        _cmd_charge_planar(args)
    elif args.subcommand == "bader":
        _cmd_charge_bader(args)
    elif args.subcommand == "compare-bader":
        _cmd_charge_compare_bader(args)
    else:
        handle_error(f"未知的 charge 子命令: '{args.subcommand}'")


def _cmd_charge_diff(args):
    """
    差分电荷密度计算 / Differential charge density calculation.
    """
    from matkit.parsers import read_chgcar, read_poscar
    from matkit.charge import calc_diff_charge_density, export_cube, planar_average

    print(color_header(">>> 差分电荷密度计算 / Differential Charge Density"))
    print(f"  总系统 CHGCAR: {args.total}")
    print(f"  衬底 CHGCAR:   {args.slab}")
    print(f"  吸附物 CHGCAR: {args.ads}")
    print()

    try:
        chgcar_total = read_chgcar(args.total)
        chgcar_slab = read_chgcar(args.slab)
        chgcar_ads = read_chgcar(args.ads)

        print("  正在计算差分电荷密度...")
        diff_density = calc_diff_charge_density(chgcar_total, chgcar_slab, chgcar_ads)

        print()
        print(color_header("=== 差分电荷密度统计 / Diff Charge Density Statistics ==="))
        print()
        print_kv("Grid shape (网格形状)", str(diff_density.shape))
        print_kv("Max (最大值)", f"{diff_density.max():.6e} e/A^3")
        print_kv("Min (最小值)", f"{diff_density.min():.6e} e/A^3")
        print_kv("Mean (平均值)", f"{diff_density.mean():.6e} e/A^3")
        print_kv("Std (标准差)", f"{diff_density.std():.6e} e/A^3")
        print_kv("Total charge (总电荷)", f"{diff_density.sum():.6e} e")

        # Z 方向切片分析 / Z-direction slice analysis
        if args.slice_z is not None:
            print()
            print(color_header(f"=== Z 方向切片 (z = {args.slice_z}) ==="))
            print()
            # 找到最近的 z 切片 / Find nearest z slice
            nz = diff_density.shape[2]
            slice_idx = min(args.slice_z, nz - 1)
            slice_data = diff_density[:, :, slice_idx]
            print_kv("Slice index", slice_idx)
            print_kv("Slice max", f"{slice_data.max():.6e} e/A^3")
            print_kv("Slice min", f"{slice_data.min():.6e} e/A^3")

        # 导出 cube 文件 / Export cube file
        if args.cube:
            print()
            print("  正在导出 cube 文件...")
            # 尝试从 POSCAR 获取原子信息 / Try to get atom info from POSCAR
            poscar_path = args.total.replace("CHGCAR", "POSCAR").replace("chgcar", "POSCAR")
            atoms = []
            coords = []
            lattice = chgcar_total.get('lattice', None)

            if lattice is None and os.path.isfile(poscar_path):
                try:
                    poscar_data = read_poscar(poscar_path)
                    lattice = poscar_data['lattice']
                    for elem, count in poscar_data['n_atoms'].items():
                        atoms.extend([elem] * count)
                    coords = poscar_data['coords']
                except Exception:
                    pass

            if lattice is None:
                handle_error("无法获取晶格信息，请确保 CHGCAR 或 POSCAR 文件可用")

            if not atoms:
                atoms = ["X"] * int(np.prod(diff_density.shape))
                coords = np.zeros((len(atoms), 3))

            export_cube(diff_density, lattice, atoms, coords, args.cube)

        # 导出绘图数据 / Export plot data
        if args.plot:
            print()
            print("  正在导出平面平均数据...")
            planar_avg = planar_average(diff_density, axis=2)
            plot_path = os.path.abspath(args.plot)

            # 尝试用 matplotlib 绘图 / Try to plot with matplotlib
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(planar_avg, color='blue', linewidth=1.0)
                ax.set_xlabel('Grid point (z-direction)')
                ax.set_ylabel('Planar average charge density (e/A^3)')
                ax.set_title('Planar Average Differential Charge Density')
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"\n{color_success('绘图已保存 / Plot saved:')} {plot_path}")
            except ImportError:
                # 回退: 保存数据为文本 / Fallback: save data as text
                data_path = plot_path.replace('.png', '.dat')
                np.savetxt(data_path, planar_avg)
                print(f"\n{color_warning('matplotlib 未安装，数据已保存为:')} {data_path}")

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_charge_planar(args):
    """
    平面平均电荷密度 / Planar average charge density.
    """
    from matkit.parsers import read_chgcar
    from matkit.charge import planar_average

    print(color_header(">>> 平面平均电荷密度 / Planar Average Charge Density"))
    print(f"  CHGCAR: {args.chgcar}")
    print(f"  平均轴: {['X', 'Y', 'Z'][args.axis]}")
    print()

    try:
        chgcar = read_chgcar(args.chgcar)
        charge_density = chgcar['total_charge']

        avg = planar_average(charge_density, axis=args.axis)

        print(color_header("=== 平面平均统计 / Planar Average Statistics ==="))
        print()
        print_kv("Grid points (网格点数)", len(avg))
        print_kv("Max (最大值)", f"{avg.max():.6e} e/A^3")
        print_kv("Min (最小值)", f"{avg.min():.6e} e/A^3")
        print_kv("Mean (平均值)", f"{avg.mean():.6e} e/A^3")

        if args.plot:
            plot_path = os.path.abspath(args.plot)
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(avg, color='blue', linewidth=1.0)
                ax.set_xlabel(f'Grid point ({["x", "y", "z"][args.axis]}-direction)')
                ax.set_ylabel('Planar average (e/A^3)')
                ax.set_title(f'Planar Average Charge Density along {["X", "Y", "Z"][args.axis]}')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"\n{color_success('绘图已保存 / Plot saved:')} {plot_path}")
            except ImportError:
                data_path = plot_path.replace('.png', '.dat')
                np.savetxt(data_path, avg)
                print(f"\n{color_warning('matplotlib 未安装，数据已保存为:')} {data_path}")

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_charge_bader(args):
    """
    Bader 电荷分析 / Bader charge analysis.
    """
    from matkit.charge import parse_bader_results, summarize_bader_by_element, export_bader_report

    print(color_header(">>> Bader 电荷分析 / Bader Charge Analysis"))
    print(f"  ACF.dat: {args.acf}")
    if args.poscar:
        print(f"  POSCAR:  {args.poscar}")
    print()

    try:
        results = parse_bader_results(args.acf, args.poscar)

        print(color_header(f"=== Bader 分析结果 ({results['n_atoms']} 个原子) ==="))
        print()

        # 逐原子显示 / Per-atom display
        headers = ["Index", "Element", "Charge (e)", "Transfer (e)", "Min Dist (A)", "Volume (A^3)"]
        rows = []
        for i in range(results['n_atoms']):
            elem = results['elements'][i] if i < len(results['elements']) else "?"
            ct = results['charge_transfer'][i] if i < len(results['charge_transfer']) else 0.0
            rows.append([
                str(i),
                elem,
                f"{results['charges'][i]:.6f}",
                f"{ct:+.6f}",
                f"{results['min_dist'][i]:.6f}",
                f"{results['volumes'][i]:.2f}",
            ])
        print_table(headers, rows)

        # 按元素汇总 / Summary by element
        if results['elements'] and any(e is not None for e in results['elements']):
            print()
            print(color_header("=== 按元素汇总 / Summary by Element ==="))
            print()
            summary = summarize_bader_by_element(results['charges'], results['elements'])
            sum_headers = ["Element", "Mean Charge (e)", "Std", "Min", "Max", "Count"]
            sum_rows = []
            for elem, stats in sorted(summary.items()):
                sum_rows.append([
                    elem,
                    f"{stats['mean']:.6f}",
                    f"{stats['std']:.6f}",
                    f"{stats['min']:.6f}",
                    f"{stats['max']:.6f}",
                    str(stats['count']),
                ])
            print_table(sum_headers, sum_rows)

        if args.csv:
            export_bader_report(results, args.csv)

        if args.json:
            save_json(args.json, results)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_charge_compare_bader(args):
    """
    比较 Bader 电荷 / Compare Bader charges.
    """
    from matkit.charge import parse_bader_results, compare_bader_charges

    print(color_header(">>> Bader 电荷比较 / Bader Charge Comparison"))
    print(f"  清洁表面 ACF.dat: {args.acf_clean}")
    print(f"  吸附后 ACF.dat:   {args.acf_adsorbed}")
    print(f"  比较原子索引:     {args.indices}")
    if args.poscar:
        print(f"  POSCAR:          {args.poscar}")
    print()

    try:
        indices = parse_indices_string(args.indices)

        clean_results = parse_bader_results(args.acf_clean, args.poscar)
        ads_results = parse_bader_results(args.acf_adsorbed, args.poscar)

        comparison = compare_bader_charges(clean_results, ads_results, indices)

        print(color_header("=== Bader 电荷比较结果 ==="))
        print()
        headers = ["Index", "Element", "Q_clean (e)", "Q_ads (e)", "Delta_Q (e)"]
        rows = []
        for i, idx in enumerate(comparison['atom_indices']):
            elem = comparison['elements'][i] if comparison['elements'][i] else "?"
            q_clean = comparison['charges_clean'][i]
            q_ads = comparison['charges_adsorbed'][i]
            dq = comparison['charge_diff'][i]
            rows.append([
                str(idx),
                elem,
                f"{q_clean:.6f}",
                f"{q_ads:.6f}",
                f"{dq:+.6f}",
            ])
        print_table(headers, rows)

        # 电荷转移分析 / Charge transfer analysis
        print()
        total_transfer = sum(comparison['charge_diff'])
        print_kv("总电荷转移 / Total charge transfer", f"{total_transfer:+.6f} e")
        if total_transfer > 0:
            print(f"  {color_success('表面向吸附物转移电子 / Electron transfer from surface to adsorbate')}")
        elif total_transfer < 0:
            print(f"  {color_warning('吸附物向表面转移电子 / Electron transfer from adsorbate to surface')}")

        if args.json:
            save_json(args.json, comparison)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def add_charge_subparser(subparsers):
    """添加 charge 子命令解析器 / Add charge subcommand parser."""
    parser = subparsers.add_parser(
        "charge",
        help="电荷分析 (差分电荷密度、Bader 分析)",
        description="电荷分析模块 / Charge analysis module",
    )
    subparsers_charge = parser.add_subparsers(dest="subcommand", help="电荷分析子命令")

    # charge diff
    p_diff = subparsers_charge.add_parser(
        "diff",
        help="计算差分电荷密度 / Calculate differential charge density",
        description="计算差分电荷密度并可选导出 cube 文件",
    )
    p_diff.add_argument("--total", required=True, help="总系统 CHGCAR 文件路径 / Total system CHGCAR path")
    p_diff.add_argument("--slab", required=True, help="衬底 CHGCAR 文件路径 / Slab CHGCAR path")
    p_diff.add_argument("--ads", required=True, help="吸附物 CHGCAR 文件路径 / Adsorbate CHGCAR path")
    p_diff.add_argument("--cube", help="输出 cube 文件路径 / Output cube file path")
    p_diff.add_argument("--slice-z", type=int, help="显示指定 z 切片的统计 / Show statistics for z-slice")
    p_diff.add_argument("--plot", help="输出平面平均绘图 (PNG) / Output planar average plot (PNG)")

    # charge planar
    p_planar = subparsers_charge.add_parser(
        "planar",
        help="计算平面平均电荷密度 / Calculate planar average charge density",
        description="沿指定轴计算平面平均电荷密度",
    )
    p_planar.add_argument("chgcar", help="CHGCAR 文件路径 / CHGCAR file path")
    p_planar.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="平均轴方向 (0=X, 1=Y, 2=Z, 默认: 2) / Averaging axis (0=X, 1=Y, 2=Z, default: 2)")
    p_planar.add_argument("--plot", help="输出绘图 (PNG) / Output plot (PNG)")

    # charge bader
    p_bader = subparsers_charge.add_parser(
        "bader",
        help="Bader 电荷分析 / Bader charge analysis",
        description="解析 ACF.dat 文件并进行 Bader 电荷分析",
    )
    p_bader.add_argument("acf", help="ACF.dat 文件路径 / ACF.dat file path")
    p_bader.add_argument("--poscar", help="POSCAR 文件路径 (用于元素匹配) / POSCAR path (for element matching)")
    p_bader.add_argument("--csv", help="导出 CSV 文件路径 / CSV output path")
    p_bader.add_argument("--json", help="导出 JSON 文件路径 / JSON output path")

    # charge compare-bader
    p_compare = subparsers_charge.add_parser(
        "compare-bader",
        help="比较两组 Bader 电荷 / Compare two Bader charge analyses",
        description="比较清洁表面和吸附表面的 Bader 电荷差异",
    )
    p_compare.add_argument("acf_clean", help="清洁表面 ACF.dat 文件路径 / Clean surface ACF.dat path")
    p_compare.add_argument("acf_adsorbed", help="吸附后 ACF.dat 文件路径 / Adsorbed ACF.dat path")
    p_compare.add_argument("--indices", required=True, help='要比较的原子索引，格式: "1,2,3,4,5" / Atom indices to compare, format: "1,2,3,4,5"')
    p_compare.add_argument("--poscar", help="POSCAR 文件路径 / POSCAR path")
    p_compare.add_argument("--json", help="导出 JSON 文件路径 / JSON output path")


# ============================================================================
# 子命令: adsorption
# ============================================================================

def cmd_adsorption(args):
    """吸附分析子命令 / Adsorption analysis subcommand."""
    if args.subcommand == "analyze":
        _cmd_adsorption_analyze(args)
    elif args.subcommand == "geometry":
        _cmd_adsorption_geometry(args)
    elif args.subcommand == "so4-cu2o":
        _cmd_adsorption_so4_cu2o(args)
    elif args.subcommand == "bonds":
        _cmd_adsorption_bonds(args)
    elif args.subcommand == "surface-change":
        _cmd_adsorption_surface_change(args)
    else:
        handle_error(f"未知的 adsorption 子命令: '{args.subcommand}'")


def _cmd_adsorption_analyze(args):
    """
    吸附分析 - 自动识别吸附物 / Adsorption analysis - auto-identify adsorbate.
    """
    from matkit.parsers import read_poscar
    from matkit.structure import identify_molecules, find_adsorption_sites

    print(color_header(">>> 吸附分析 / Adsorption Analysis"))
    print(f"  CONTCAR: {args.contcar}")
    print(f"  键长截断值: {args.cutoff} A")
    if args.surface_element:
        print(f"  表面元素: {args.surface_element}")
    print()

    try:
        data = read_poscar(args.contcar)

        elements = atom_elements_from_poscar_data(data)

        # 使用通用截断值识别分子 / Use general cutoffs to identify molecules
        # 常见键长截断值 / Common bond cutoffs
        default_cutoffs = {
            ('O', 'H'): 1.3, ('N', 'H'): 1.2, ('C', 'H'): 1.2,
            ('O', 'O'): 1.8, ('N', 'N'): 1.6, ('C', 'C'): 1.8,
            ('C', 'O'): 1.6, ('C', 'N'): 1.6, ('S', 'O'): 2.0,
            ('S', 'S'): 2.2, ('N', 'O'): 1.6, ('O', 'S'): 2.0,
            ('O', 'Cu'): args.cutoff, ('O', 'Fe'): args.cutoff,
            ('O', 'Ni'): args.cutoff, ('O', 'Pt'): args.cutoff,
            ('O', 'Pd'): args.cutoff, ('O', 'Au'): args.cutoff,
            ('O', 'Ag'): args.cutoff, ('O', 'Zn'): args.cutoff,
            ('S', 'Cu'): args.cutoff, ('S', 'Fe'): args.cutoff,
            ('N', 'Cu'): args.cutoff, ('C', 'Cu'): args.cutoff,
            ('H', 'Cu'): args.cutoff,
        }

        molecules = identify_molecules(elements, data['coords'], default_cutoffs)

        print(color_header(f"=== 识别到 {len(molecules)} 个分子/原子团 ==="))
        print()

        # 分类: 小分子 vs 大团簇 / Classify: small molecules vs large clusters
        mol_rows = []
        for i, mol in enumerate(molecules):
            n_atoms = len(mol['indices'])
            mol_type = "adsorbate" if n_atoms <= 20 else "surface/lattice"
            mol_rows.append([
                str(i),
                mol['formula'],
                str(n_atoms),
                mol_type,
                str(mol['indices'][:8]) + ("..." if n_atoms > 8 else ""),
            ])
        print_table(["#", "Formula", "N_atoms", "Type", "Indices"], mol_rows)

        # 吸附位点分析 / Adsorption site analysis
        if args.surface_element:
            print()
            print(color_header("=== 吸附位点分析 / Adsorption Site Analysis ==="))
            print()

            surface_elements_list = [
                elem if elem == args.surface_element else None
                for elem in elements
            ]

            for i, mol in enumerate(molecules):
                if len(mol['indices']) > 20:
                    continue  # 跳过大团簇 / Skip large clusters

                sites = find_adsorption_sites(
                    mol, surface_elements_list, data['coords'], args.cutoff
                )
                if sites:
                    print_kv(f"分子 #{i} ({mol['formula']})", f"{len(sites)} 个吸附位点")
                    for site in sites:
                        neighbors_str = ", ".join(
                            f"{n['element']}@{n['surface_index']}({n['distance']:.3f}A)"
                            for n in site['surface_neighbors']
                        )
                        print(f"    原子 {site['global_index']} ({site['element']}): "
                              f"最近距离 = {site['min_distance']:.4f} A")
                        print(f"      邻近表面原子: {neighbors_str}")

        if args.csv:
            headers = ["#", "Formula", "N_atoms", "Indices"]
            rows = [[i, mol['formula'], len(mol['indices']), str(mol['indices'])]
                    for i, mol in enumerate(molecules)]
            save_csv(args.csv, headers, rows)

        if args.json:
            json_data = []
            for i, mol in enumerate(molecules):
                entry = {
                    "index": i,
                    "formula": mol['formula'],
                    "n_atoms": len(mol['indices']),
                    "indices": mol['indices'],
                    "elements": mol['elements'],
                }
                json_data.append(entry)
            save_json(args.json, json_data)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_adsorption_bonds(args):
    """
    吸附键长分析 / Adsorption bond length analysis.
    """
    from matkit.parsers import read_poscar
    from matkit.structure import identify_molecules, find_adsorption_sites

    print(color_header(">>> 吸附键长分析 / Adsorption Bond Analysis"))
    print(f"  CONTCAR: {args.contcar}")
    print(f"  键长截断值: {args.cutoff} A")
    if args.surface_element:
        print(f"  表面元素: {args.surface_element}")
    print()

    try:
        data = read_poscar(args.contcar)

        elements = atom_elements_from_poscar_data(data)

        # 构建截断值 / Build cutoffs
        cutoffs = {
            ('O', 'H'): 1.3, ('N', 'H'): 1.2, ('C', 'H'): 1.2,
            ('O', 'O'): 1.8, ('N', 'N'): 1.6, ('C', 'C'): 1.8,
            ('C', 'O'): 1.6, ('C', 'N'): 1.6, ('S', 'O'): 2.0,
            ('S', 'S'): 2.2,
        }
        if args.surface_element:
            for ads_elem in ['O', 'S', 'N', 'C', 'H']:
                cutoffs[(ads_elem, args.surface_element)] = args.cutoff

        molecules = identify_molecules(elements, data['coords'], cutoffs)

        print(color_header("=== 吸附键长 ==="))
        print()

        bond_rows = []
        for i, mol in enumerate(molecules):
            if len(mol['indices']) > 20:
                continue

            if args.surface_element:
                surface_elems = [
                    elem if elem == args.surface_element else None
                    for elem in elements
                ]
                sites = find_adsorption_sites(
                    mol, surface_elems, data['coords'], args.cutoff
                )
                for site in sites:
                    for neighbor in site['surface_neighbors']:
                        bond_rows.append([
                            str(site['global_index']),
                            site['element'],
                            str(neighbor['surface_index']),
                            neighbor['element'],
                            f"{neighbor['distance']:.6f}",
                        ])
            else:
                # 无表面元素时，显示分子内键长 / Without surface element, show intra-molecular bonds
                for mi, gi in enumerate(mol['indices']):
                    for mj, gj in enumerate(mol['indices']):
                        if gi < gj:
                            dist = np.linalg.norm(data['coords'][gi] - data['coords'][gj])
                            if dist <= args.cutoff:
                                bond_rows.append([
                                    str(gi), mol['elements'][mi],
                                    str(gj), mol['elements'][mj],
                                    f"{dist:.6f}",
                                ])

        if bond_rows:
            print_table(["Index i", "Elem", "Index j", "Elem", "Distance (A)"], bond_rows)
        else:
            print(f"  {color_warning('未找到吸附键 / No adsorption bonds found')}")

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_adsorption_surface_change(args):
    """
    表面变化分析 / Surface change analysis.
    """
    from matkit.parsers import read_poscar
    from matkit.structure import identify_surface_layers, analyze_surface_relaxation

    print(color_header(">>> 表面变化分析 / Surface Change Analysis"))
    print(f"  清洁表面 CONTCAR: {args.contcar_clean}")
    print(f"  吸附后 CONTCAR:   {args.contcar_adsorbed}")
    print(f"  表面元素:         {args.surface_element}")
    print(f"  变化阈值:         {args.threshold} A")
    print()

    try:
        clean_data = read_poscar(args.contcar_clean)
        ads_data = read_poscar(args.contcar_adsorbed)

        clean_elements = atom_elements_from_poscar_data(clean_data)

        layers = identify_surface_layers(clean_elements, clean_data['coords'], args.surface_element)
        if not layers:
            handle_error(f"未找到元素 '{args.surface_element}' 的表面层")

        # 使用最外两层 / Use outermost two layers
        surface_indices = []
        for layer in layers[-2:]:
            surface_indices.extend(layer['indices'])

        print_kv("表面原子数 (最外两层)", len(surface_indices))

        result = analyze_surface_relaxation(
            clean_data['coords'],
            ads_data['coords'],
            surface_indices,
            tolerance=args.threshold,
        )

        print()
        print(color_header("=== 表面变化结果 ==="))
        print()
        print_kv("变化原子数 / N_changed", result['n_relaxed'])
        print_kv("未变化原子数 / N_unchanged", result['n_unrelaxed'])
        print_kv("平均位移 / Avg displacement", f"{result['avg_displacement']:.6f} A")
        print_kv("最大位移 / Max displacement", f"{result['max_displacement']:.6f} A")

        if result['relaxed_atoms']:
            print()
            print(color_header("=== 变化原子详情 ==="))
            print()
            rows = []
            for atom in result['relaxed_atoms']:
                elem = clean_elements[atom['index']] if atom['index'] < len(clean_elements) else "?"
                rows.append([
                    str(atom['index']),
                    elem,
                    f"{atom['displacement']:.6f}",
                    f"{atom['dx']:.6f}",
                    f"{atom['dy']:.6f}",
                    f"{atom['dz']:.6f}",
                ])
            print_table(["Index", "Element", "|d| (A)", "dx (A)", "dy (A)", "dz (A)"], rows)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_adsorption_geometry(args):
    """
    Generic adsorbate-on-surface geometry analysis.
    """
    from matkit.analysis import analyze_adsorbate_geometry, suggest_next_analyses

    print(color_header(">>> 通用吸附几何分析 / Generic Adsorption Geometry"))
    print(f"  结构文件:       {args.contcar}")
    print(f"  吸附物化学式:   {args.formula or 'auto'}")
    print(f"  表面元素:       {args.surface_element}")
    print(f"  表面方向:       {args.surface_side}")
    print(f"  吸附截断值:     {args.surface_cutoff} A")
    print()

    try:
        result = analyze_adsorbate_geometry(
            poscar_path=args.contcar,
            adsorbate_formula=args.formula,
            surface_element=args.surface_element,
            surface_cutoff=args.surface_cutoff,
            layer_tolerance=args.layer_tolerance,
            n_surface_layers=args.surface_layers,
            surface_side=args.surface_side,
        )

        ads = result["adsorbate"]
        surface = result["surface"]
        height = result["height_stats_A"]

        print(color_header("=== 吸附物 / Adsorbate ==="))
        print()
        print_kv("Formula", ads["formula"])
        print_kv("N_atoms", ads["n_atoms"])
        print_kv("Indices (1-based)", ads["indices1"])
        print_kv("Center", [round(x, 4) for x in ads["center_A"]])
        print()

        print(color_header("=== 表面参考 / Surface Reference ==="))
        print()
        print_kv("Surface element", surface["element"])
        print_kv("Surface side", surface["side"])
        print_kv("Reference z", f"{surface['reference_plane_z_A']:.6f} A")
        print_kv("Surface atoms", len(surface["surface_indices1"]))
        print_kv("Mean |z distance|", f"{height['mean_abs_z_to_surface']:.6f} A")
        print()

        print(color_header("=== 吸附接触 / Adsorption Contacts ==="))
        print()
        contact_rows = [
            [
                str(row["adsorbate_index1"]),
                row["adsorbate_element"],
                str(row["surface_index1"]),
                row["surface_element"],
                f"{row['distance_A']:.6f}",
            ]
            for row in result["contacts"]
        ]
        if contact_rows:
            print_table(["Ads idx", "Elem", "Surf idx", "Elem", "Distance (A)"], contact_rows)
        else:
            print(f"  {color_warning('未识别到吸附接触 / No contacts detected')}")

        if args.recommend:
            print()
            print(color_header("=== 本地建议 / Local Recommendations ==="))
            print()
            for item in suggest_next_analyses(result):
                print(f"  [{item['priority']}] {item['topic']}: {item['suggestion']}")

        if args.json:
            save_json(args.json, result)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def _cmd_adsorption_so4_cu2o(args):
    """
    SO4/Cu2O adsorption geometry analysis.
    """
    from matkit.analysis import analyze_so4_cu2o, flatten_so4_metrics, suggest_next_analyses

    print(color_header(">>> SO4/Cu2O 吸附几何分析 / SO4 on Cu2O Analysis"))
    print(f"  结构文件:       {args.contcar}")
    print(f"  表面元素:       {args.surface_element}")
    print(f"  表面方向:       {args.surface_side}")
    print(f"  Cu-O 截断值:    {args.surface_cutoff} A")
    print(f"  S-O 截断值:     {args.s_o_cutoff} A")
    print()

    try:
        result = analyze_so4_cu2o(
            poscar_path=args.contcar,
            surface_element=args.surface_element,
            surface_cutoff=args.surface_cutoff,
            s_o_cutoff=args.s_o_cutoff,
            layer_tolerance=args.layer_tolerance,
            n_surface_layers=args.surface_layers,
            surface_side=args.surface_side,
        )

        metrics = result["so4_metrics"]

        print(color_header("=== 关键结果 / Key Results ==="))
        print()
        print_kv("S atom index (1-based)", metrics["s_index1"])
        print_kv("Reference surface z", f"{metrics['surface_reference_z_A']:.6f} A")
        print_kv("S z", f"{metrics['s_z_A']:.6f} A")
        print_kv("S -> surface dz", f"{metrics['s_to_surface_signed_z_A']:.6f} A")
        print_kv("|S -> surface dz|", f"{metrics['s_to_surface_abs_z_A']:.6f} A")
        print_kv("Cu-bound O atoms (1-based)", metrics["surface_bound_o_indices1"])
        print()

        print(color_header("=== 吸附 O 的 S-O 键长 / S-O Bonds of Adsorbing O ==="))
        print()
        bond_rows = []
        for row in metrics["adsorbed_s_o_bonds"]:
            nearest = row["nearest_surface_distance_A"]
            bond_rows.append([
                str(row["o_index1"]),
                str(row["s_index1"]),
                f"{row['s_o_distance_A']:.6f}",
                f"{nearest:.6f}" if nearest is not None else "-",
            ])
        if bond_rows:
            print_table(["O idx", "S idx", "S-O (A)", "nearest Cu-O (A)"], bond_rows)
        else:
            print(f"  {color_warning('未识别到吸附在 Cu 上的 SO4-O 原子')}")

        print()
        print(color_header("=== 吸附 O-S-O 夹角 / O-S-O Angles ==="))
        print()
        angle_rows = [
            [
                str(row["o1_index1"]),
                str(row["s_index1"]),
                str(row["o2_index1"]),
                f"{row['angle_deg']:.4f}",
            ]
            for row in metrics["o_s_o_angles"]
        ]
        if angle_rows:
            print_table(["O1 idx", "S idx", "O2 idx", "Angle (deg)"], angle_rows)
        else:
            print(f"  {color_warning('吸附 O 少于两个，无法计算吸附 O-S-O 夹角')}")

        if args.verbose:
            print()
            print(color_header("=== 全部 S-O 键长 / All S-O Bonds ==="))
            print()
            all_rows = [
                [
                    str(row["o_index1"]),
                    str(row["s_index1"]),
                    "yes" if row["is_surface_bound_o"] else "no",
                    f"{row['s_o_distance_A']:.6f}",
                ]
                for row in metrics["all_s_o_bonds"]
            ]
            print_table(["O idx", "S idx", "Cu-bound", "S-O (A)"], all_rows)

        for warning in result.get("warnings", []):
            print(f"\n  {color_warning('注意 / Warning:')} {warning}")

        if args.recommend:
            print()
            print(color_header("=== 本地建议 / Local Recommendations ==="))
            print()
            for item in suggest_next_analyses(result):
                print(f"  [{item['priority']}] {item['topic']}: {item['suggestion']}")

        if args.csv:
            rows = flatten_so4_metrics(result)
            save_csv(
                args.csv,
                ["metric", "atom_indices_1based", "value", "unit"],
                [[row["metric"], row["atom_indices_1based"], row["value"], row["unit"]] for row in rows],
            )

        if args.json:
            save_json(args.json, result)

    except FileNotFoundError as e:
        handle_error(str(e))
    except ValueError as e:
        handle_error(str(e))


def add_adsorption_subparser(subparsers):
    """添加 adsorption 子命令解析器 / Add adsorption subcommand parser."""
    parser = subparsers.add_parser(
        "adsorption",
        help="吸附分析 (吸附物识别、键长、表面变化)",
        description="吸附分析模块 / Adsorption analysis module",
    )
    subparsers_ads = parser.add_subparsers(dest="subcommand", help="吸附分析子命令")

    # adsorption analyze
    p_analyze = subparsers_ads.add_parser(
        "analyze",
        help="吸附分析 - 自动识别吸附物 / Adsorption analysis - auto-identify adsorbate",
        description="自动识别吸附物分子并分析吸附位点",
    )
    p_analyze.add_argument("contcar", help="CONTCAR 文件路径 / CONTCAR file path")
    p_analyze.add_argument("--cutoff", type=float, default=2.5, help="吸附键长截断值 (A, 默认: 2.5) / Adsorption bond cutoff (A, default: 2.5)")
    p_analyze.add_argument("--surface-element", help="表面元素符号 (如 Cu) / Surface element symbol (e.g., Cu)")
    p_analyze.add_argument("--csv", help="导出 CSV 文件路径 / CSV output path")
    p_analyze.add_argument("--json", help="导出 JSON 文件路径 / JSON output path")

    # adsorption geometry
    p_geometry = subparsers_ads.add_parser(
        "geometry",
        help="通用吸附几何分析 / Generic adsorbate geometry analysis",
        description="识别吸附物、表面参考层和吸附接触",
    )
    p_geometry.add_argument("contcar", help="POSCAR/CONTCAR 文件路径 / POSCAR/CONTCAR path")
    p_geometry.add_argument("--formula", help="吸附物化学式，如 SO4 或 C6H5N3 / Adsorbate formula")
    p_geometry.add_argument("--surface-element", default="Cu", help="表面元素符号 (默认: Cu) / Surface element (default: Cu)")
    p_geometry.add_argument("--surface-cutoff", type=float, default=2.5, help="吸附接触截断值 (A, 默认: 2.5) / Contact cutoff")
    p_geometry.add_argument("--layer-tolerance", type=float, default=0.5, help="表面层 z 聚类容差 (A, 默认: 0.5) / Layer tolerance")
    p_geometry.add_argument("--surface-layers", type=int, default=1, help="用于吸附接触的外层层数 (默认: 1) / Number of outer layers")
    p_geometry.add_argument("--surface-side", choices=["auto", "top", "bottom"], default="auto", help="表面方向 (默认: auto) / Surface side")
    p_geometry.add_argument("--recommend", action="store_true", help="输出本地下一步建议 / Print local next-step recommendations")
    p_geometry.add_argument("--json", help="导出 JSON 文件路径 / JSON output path")

    # adsorption so4-cu2o
    p_so4 = subparsers_ads.add_parser(
        "so4-cu2o",
        help="SO4/Cu2O 专项几何分析 / SO4 on Cu2O geometry analysis",
        description="提取吸附 O 的 S-O 键长、吸附 O-S-O 夹角和 S 到 Cu2O 表面的 z 向距离",
    )
    p_so4.add_argument("contcar", help="POSCAR/CONTCAR 文件路径 / POSCAR/CONTCAR path")
    p_so4.add_argument("--surface-element", default="Cu", help="表面配位元素 (默认: Cu) / Surface binding element")
    p_so4.add_argument("--surface-cutoff", type=float, default=2.5, help="Cu-O 吸附接触截断值 (A, 默认: 2.5) / Cu-O contact cutoff")
    p_so4.add_argument("--s-o-cutoff", type=float, default=1.9, help="SO4 内部 S-O 截断值 (A, 默认: 1.9) / Internal S-O cutoff")
    p_so4.add_argument("--layer-tolerance", type=float, default=0.5, help="Cu 表面层 z 聚类容差 (A, 默认: 0.5) / Surface layer tolerance")
    p_so4.add_argument("--surface-layers", type=int, default=1, help="用于吸附接触的外层 Cu 层数 (默认: 1) / Number of outer Cu layers")
    p_so4.add_argument("--surface-side", choices=["auto", "top", "bottom"], default="auto", help="表面方向 (默认: auto) / Surface side")
    p_so4.add_argument("--verbose", action="store_true", help="输出所有 SO4 内部 S-O 键长 / Print all S-O bonds")
    p_so4.add_argument("--recommend", action="store_true", help="输出本地下一步建议 / Print local next-step recommendations")
    p_so4.add_argument("--csv", help="导出关键指标 CSV 文件路径 / CSV output path")
    p_so4.add_argument("--json", help="导出完整 JSON 文件路径 / JSON output path")

    # adsorption bonds
    p_bonds = subparsers_ads.add_parser(
        "bonds",
        help="吸附键长分析 / Adsorption bond length analysis",
        description="分析吸附物与表面之间的键长",
    )
    p_bonds.add_argument("contcar", help="CONTCAR 文件路径 / CONTCAR file path")
    p_bonds.add_argument("--cutoff", type=float, default=2.5, help="键长截断值 (A, 默认: 2.5) / Bond cutoff (A, default: 2.5)")
    p_bonds.add_argument("--surface-element", help="表面元素符号 (如 Cu) / Surface element symbol (e.g., Cu)")

    # adsorption surface-change
    p_change = subparsers_ads.add_parser(
        "surface-change",
        help="表面变化分析 / Surface change analysis",
        description="比较吸附前后表面原子的位移",
    )
    p_change.add_argument("contcar_clean", help="清洁表面 CONTCAR 文件路径 / Clean surface CONTCAR path")
    p_change.add_argument("contcar_adsorbed", help="吸附后 CONTCAR 文件路径 / Adsorbed CONTCAR path")
    p_change.add_argument("--surface-element", required=True, help="表面元素符号 (如 Cu) / Surface element symbol (e.g., Cu)")
    p_change.add_argument("--threshold", type=float, default=0.1, help="变化判定阈值 (A, 默认: 0.1) / Change threshold (A, default: 0.1)")


# ============================================================================
# 子命令: ai
# ============================================================================

def cmd_ai(args):
    """AI 助手子命令 / AI assistant subcommand."""
    if args.subcommand == "analyze":
        _cmd_ai_analyze(args)
    elif args.subcommand == "suggest":
        _cmd_ai_suggest(args)
    elif args.subcommand == "incar":
        _cmd_ai_incar(args)
    elif args.subcommand == "explain":
        _cmd_ai_explain(args)
    else:
        handle_error(f"未知的 ai 子命令: '{args.subcommand}'")


def _cmd_ai_analyze(args):
    """
    AI 分析计算结果 / AI analyze calculation results.
    """
    from matkit.ai import MatKitAI

    print(color_header(">>> AI 结果分析 / AI Result Analysis"))
    print(f"  上下文: {args.context}")
    print(f"  结果文件: {args.results}")
    print()

    try:
        # 读取结果文件 / Read results file
        with open(args.results, 'r', encoding='utf-8') as f:
            if args.results.endswith('.json'):
                results_dict = json.load(f)
            else:
                results_dict = {"raw_content": f.read()}

        # 初始化 AI 助手 / Initialize AI assistant
        ai = MatKitAI(api_key=args.api_key, model=args.model, base_url=args.base_url)

        print("  正在调用 AI 分析...")
        print()

        analysis = ai.analyze_results(args.context, results_dict)

        print(color_header("=== AI 分析结果 ==="))
        print()
        print(analysis)

    except FileNotFoundError:
        handle_error(f"结果文件不存在: {args.results}")
    except json.JSONDecodeError as e:
        handle_error(f"JSON 解析失败: {e}")


def _cmd_ai_suggest(args):
    """
    AI 建议下一步 / AI suggest next steps.
    """
    from matkit.ai import MatKitAI

    print(color_header(">>> AI 建议下一步 / AI Suggest Next Steps"))
    print(f"  当前结果: {args.context}")
    print(f"  可用工具: {args.tools}")
    print()

    try:
        ai = MatKitAI(api_key=args.api_key, model=args.model, base_url=args.base_url)

        print("  正在调用 AI...")
        print()

        suggestions = ai.suggest_next_steps(args.context, args.tools)

        print(color_header("=== AI 建议 ==="))
        print()
        print(suggestions)

    except Exception as e:
        handle_error(f"AI 调用失败: {e}")


def _cmd_ai_incar(args):
    """
    AI 生成 INCAR 参数 / AI generate INCAR parameters.
    """
    from matkit.ai import MatKitAI

    print(color_header(">>> AI 生成 INCAR / AI Generate INCAR"))
    print(f"  研究目标: {args.goal}")
    print(f"  体系信息: {args.system}")
    print()

    try:
        ai = MatKitAI(api_key=args.api_key, model=args.model, base_url=args.base_url)

        print("  正在调用 AI...")
        print()

        incar = ai.generate_incar(args.goal, args.system)

        print(color_header("=== AI 建议的 INCAR 参数 ==="))
        print()
        print(incar)

    except Exception as e:
        handle_error(f"AI 调用失败: {e}")


def _cmd_ai_explain(args):
    """
    AI 解释现象 / AI explain phenomenon.
    """
    from matkit.ai import MatKitAI

    print(color_header(">>> AI 解释现象 / AI Explain Phenomenon"))
    print(f"  问题: {args.question}")
    if args.context:
        print(f"  背景: {args.context}")
    print()

    try:
        ai = MatKitAI(api_key=args.api_key, model=args.model, base_url=args.base_url)

        print("  正在调用 AI...")
        print()

        explanation = ai.explain_phenomenon(args.question, args.context or "")

        print(color_header("=== AI 解释 ==="))
        print()
        print(explanation)

    except Exception as e:
        handle_error(f"AI 调用失败: {e}")


def add_ai_subparser(subparsers):
    """添加 ai 子命令解析器 / Add ai subcommand parser."""
    parser = subparsers.add_parser(
        "ai",
        help="AI 助手 (分析、建议、INCAR 生成、解释)",
        description="AI 助手模块 / AI assistant module",
    )
    subparsers_ai = parser.add_subparsers(dest="subcommand", help="AI 助手子命令")

    # ai analyze
    p_analyze = subparsers_ai.add_parser(
        "analyze",
        help="AI 分析计算结果 / AI analyze calculation results",
        description="使用 AI 分析 DFT 计算结果",
    )
    p_analyze.add_argument("--context", required=True, help="计算上下文描述 / Calculation context description")
    p_analyze.add_argument("--results", required=True, help="结果文件路径 (JSON) / Results file path (JSON)")
    p_analyze.add_argument("--api-key", help="API 密钥 (或设置 MATKIT_API_KEY 环境变量) / API key (or set MATKIT_API_KEY env var)")
    p_analyze.add_argument("--model", default="deepseek-chat", help="模型名称 (默认: deepseek-chat) / Model name (default: deepseek-chat)")
    p_analyze.add_argument("--base-url", default="https://api.deepseek.com", help="OpenAI兼容 API 地址 / OpenAI-compatible API base URL")

    # ai suggest
    p_suggest = subparsers_ai.add_parser(
        "suggest",
        help="AI 建议下一步计算 / AI suggest next calculation steps",
        description="基于当前结果建议后续分析步骤",
    )
    p_suggest.add_argument("--context", required=True, help="当前结果描述 / Current results description")
    p_suggest.add_argument("--tools", required=True, help="可用工具列表 / Available tools list")
    p_suggest.add_argument("--api-key", help="API 密钥 / API key")
    p_suggest.add_argument("--model", default="deepseek-chat", help="模型名称 (默认: deepseek-chat) / Model name (default: deepseek-chat)")
    p_suggest.add_argument("--base-url", default="https://api.deepseek.com", help="OpenAI兼容 API 地址 / OpenAI-compatible API base URL")

    # ai incar
    p_incar = subparsers_ai.add_parser(
        "incar",
        help="AI 生成 INCAR 参数 / AI generate INCAR parameters",
        description="根据研究目标和体系信息生成 VASP INCAR 参数建议",
    )
    p_incar.add_argument("--goal", required=True, help="研究目标描述 / Research goal description")
    p_incar.add_argument("--system", required=True, help="体系信息 (元素组成、结构类型等) / System info")
    p_incar.add_argument("--api-key", help="API 密钥 / API key")
    p_incar.add_argument("--model", default="deepseek-chat", help="模型名称 (默认: deepseek-chat) / Model name (default: deepseek-chat)")
    p_incar.add_argument("--base-url", default="https://api.deepseek.com", help="OpenAI兼容 API 地址 / OpenAI-compatible API base URL")

    # ai explain
    p_explain = subparsers_ai.add_parser(
        "explain",
        help="AI 解释现象 / AI explain phenomenon",
        description="解释 DFT 计算中观察到的物理/化学现象",
    )
    p_explain.add_argument("--question", required=True, help="要解释的问题 / Question to explain")
    p_explain.add_argument("--context", default="", help="相关计算背景信息 / Relevant calculation context")
    p_explain.add_argument("--api-key", help="API 密钥 / API key")
    p_explain.add_argument("--model", default="deepseek-chat", help="模型名称 (默认: deepseek-chat) / Model name (default: deepseek-chat)")
    p_explain.add_argument("--base-url", default="https://api.deepseek.com", help="OpenAI兼容 API 地址 / OpenAI-compatible API base URL")


# ============================================================================
# 子命令: ui
# ============================================================================

def cmd_ui(args):
    """启动本地 Web UI / Start local Web UI."""
    from matkit.ui import run_ui_server

    run_ui_server(
        host=args.host,
        port=args.port,
        open_browser=args.open,
        quiet=args.quiet,
    )


def add_ui_subparser(subparsers):
    """添加 ui 子命令解析器 / Add UI subcommand parser."""
    parser = subparsers.add_parser(
        "ui",
        help="启动点击式 Web 界面 / Start the local web interface",
        description="启动 MatKit 本地能量分析 Web 工作台",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认: 127.0.0.1) / Host")
    parser.add_argument("--port", type=int, default=8765, help="端口 (默认: 8765) / Port")
    parser.add_argument("--open", action="store_true", help="启动后尝试打开浏览器 / Open browser after start")
    parser.add_argument("--quiet", action="store_true", help="减少 HTTP 请求日志 / Reduce request logging")


# ============================================================================
# 主入口 / Main Entry Point
# ============================================================================

def build_parser():
    """
    构建 argparse 解析器 / Build argparse parser.

    Returns
    -------
    argparse.ArgumentParser
        主解析器 / Main parser.
    """
    parser = argparse.ArgumentParser(
        prog="matkit",
        description=(
            "MatKit - 综合性 VASP 后处理与材料分析工具包\n"
            "MatKit - Comprehensive VASP Post-Processing and Materials Analysis Toolkit\n\n"
            "使用方式 / Usage:\n"
            "  python -m matkit <command> [subcommand] [options]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"MatKit v{__import__('matkit').__version__}",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="禁用彩色输出 / Disable colored output",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令 / Available commands")

    # 添加各子命令 / Add subcommands
    add_energy_subparser(subparsers)
    add_structure_subparser(subparsers)
    add_charge_subparser(subparsers)
    add_adsorption_subparser(subparsers)
    add_ai_subparser(subparsers)
    add_ui_subparser(subparsers)

    return parser


def main(argv=None):
    """
    CLI 主入口函数 / CLI main entry function.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数列表。如果为 None，使用 sys.argv。
        Command-line argument list. If None, uses sys.argv.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    # 处理 --no-color / Handle --no-color
    if hasattr(args, 'no_color') and args.no_color:
        Colors.disable()

    # 无命令时显示帮助 / Show help when no command
    if args.command is None:
        print_banner()
        parser.print_help()
        print()
        print(color_key("提示 / Tip:") + " 使用 'python -m matkit <command> -h' 查看子命令帮助")
        print(color_key("Tip:") + " Use 'python -m matkit <command> -h' for subcommand help")
        return

    # 打印横幅 / Print banner
    print_banner()

    # 分发到子命令 / Dispatch to subcommands
    try:
        if args.command == "energy":
            if not hasattr(args, 'subcommand') or args.subcommand is None:
                parser.parse_args([args.command, "-h"])
                return
            cmd_energy(args)
        elif args.command == "structure":
            if not hasattr(args, 'subcommand') or args.subcommand is None:
                parser.parse_args([args.command, "-h"])
                return
            cmd_structure(args)
        elif args.command == "charge":
            if not hasattr(args, 'subcommand') or args.subcommand is None:
                parser.parse_args([args.command, "-h"])
                return
            cmd_charge(args)
        elif args.command == "adsorption":
            if not hasattr(args, 'subcommand') or args.subcommand is None:
                parser.parse_args([args.command, "-h"])
                return
            cmd_adsorption(args)
        elif args.command == "ai":
            if not hasattr(args, 'subcommand') or args.subcommand is None:
                parser.parse_args([args.command, "-h"])
                return
            cmd_ai(args)
        elif args.command == "ui":
            cmd_ui(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print(f"\n{color_warning('已中断 / Interrupted')}")
        sys.exit(1)
    except Exception as e:
        handle_error(f"未预期的错误: {e}\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
