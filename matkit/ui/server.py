"""
Local web UI server for MatKit.

This module intentionally uses only Python's standard library so the UI works in
the existing analysis environment without adding web framework dependencies.
"""

import csv
import json
import mimetypes
import os
import re
import posixpath
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from matkit.energy import (
    build_simple_substance_database,
    calc_adsorption_energy,
    calc_doping_energy,
    calc_surface_excess_energies_batch,
    calc_surface_excess_energy_from_directory,
    find_energy_file,
    find_structure_file,
    iter_calculation_directories,
    read_calculation_energy,
)
from matkit.parsers import expand_atomic_elements, read_poscar


STATIC_DIR = Path(__file__).resolve().parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0") or 0)
    raw = handler.rfile.read(length).decode("utf-8") if length else "{}"
    return json.loads(raw or "{}")


def _resolve_path(value):
    value = str(value or "").strip()
    if not value:
        return ""
    return os.path.abspath(os.path.expanduser(value))


def _energy_from_path(path):
    energy_file = find_energy_file(path)
    if energy_file is None:
        raise FileNotFoundError(f"未找到 OUTCAR/OSZICAR/log 能量文件: {path}")
    energy = read_calculation_energy(energy_file)
    return energy["final_energy"], energy_file, energy["is_converged"]


def _write_json_file(path, data):
    path = _resolve_path(path)
    if not path:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def _flatten_dict(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    return value


def _write_csv_file(path, rows):
    path = _resolve_path(path)
    if not path:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = list(rows)
    if rows:
        headers = sorted({key for row in rows for key in row.keys()})
    else:
        headers = ["status"]
    preferred = [
        "label",
        "task_number",
        "task_suffix",
        "status",
        "surface_energy_eV_per_surface",
        "adsorption_energy_eV",
        "formation_energy_eV",
        "excess_energy_eV",
        "energy_diff",
        "reference_energies",
        "E_slab",
        "E_slab_ads",
        "E_doped",
        "E_defect",
        "error",
    ]
    headers = [key for key in preferred if key in headers] + [
        key for key in headers if key not in preferred
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flatten_dict(row.get(key, "")) for key in headers})
    return path


def _with_surface_alias(row):
    if "surface_excess_energy_eV_per_surface" in row:
        row["surface_energy_eV_per_surface"] = row["surface_excess_energy_eV_per_surface"]
        del row["surface_excess_energy_eV_per_surface"]
    terms = row.get("reference_terms") or []
    if terms:
        row["reference_energies"] = "; ".join(
            f"mu_{term.get('element')}"
            for term in terms
            if isinstance(term.get("mu_eV_per_atom"), (int, float))
        )
    return row


def _is_calculation_path(path):
    if os.path.isfile(path):
        return find_energy_file(path) is not None
    return find_structure_file(path) is not None and find_energy_file(path) is not None


def _infer_task_identity(path):
    name = os.path.basename(os.path.normpath(str(path)))
    match = re.search(r"task[._-]*(\d+)(.*)$", name, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2).lstrip("._-")
    match = re.search(r"(\d+)(.*)$", name)
    if match:
        return match.group(1), match.group(2).lstrip("._-")
    return None, ""


def _parse_optional_float(payload, key):
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _parse_optional_int(payload, key):
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _charge_inputs(payload):
    charge = _parse_optional_int(payload, "chargeState")
    efermi = _parse_optional_float(payload, "efermi")
    correction = _parse_optional_float(payload, "correction")
    uses_charge = any(value not in (None, 0, 0.0) for value in (charge, efermi, correction))
    return uses_charge, int(charge or 0), float(efermi or 0.0), float(correction or 0.0)


def _remove_charge_fields_if_unused(row, uses_charge):
    if uses_charge:
        return row
    for key in ("charge_state", "efermi", "charge_correction_term", "correction"):
        row.pop(key, None)
    return row


def _formula_with_optional_charge(base):
    return f"{base} + q * E_fermi + E_corr"


def _calculation_dirs_from_path(path):
    if _is_calculation_path(path):
        return [path]
    calc_dirs = list(iter_calculation_directories(path))
    return [
        item for item in calc_dirs
        if re.search(r"task[._-]*\d+", os.path.basename(os.path.normpath(item)), re.IGNORECASE)
    ] or calc_dirs


def _read_atom_element(path, atom_index):
    atom_index = int(atom_index)
    if atom_index <= 0:
        raise ValueError(f"原子序号必须从 1 开始，但得到 {atom_index}。")
    structure_file = find_structure_file(path)
    if structure_file is None:
        raise FileNotFoundError(f"未找到 POSCAR/CONTCAR: {path}")
    data = read_poscar(structure_file)
    atom_elements = expand_atomic_elements(data["elements"], data["n_atoms"])
    if atom_index > len(atom_elements):
        raise ValueError(
            f"原子序号 {atom_index} 超出结构原子总数 {len(atom_elements)}: {structure_file}"
        )
    return atom_elements[atom_index - 1], structure_file


def _reference_mu(references, element):
    if element not in references:
        raise ValueError(f"单质库中缺少元素 {element} 的参考能量。")
    return float(references[element]["mu_eV_per_atom"])


def _calculate_surface(payload):
    final_path = _resolve_path(
        payload.get("finalPath")
        or payload.get("root")
        or payload.get("slab")
    )
    initial_path = _resolve_path(payload.get("initialPath"))
    reference_db = _resolve_path(payload.get("referenceDb")) or str(PROJECT_ROOT / "simple_substance_database")
    n_surfaces = int(payload.get("nSurfaces") or 2)
    strict = bool(payload.get("strict"))

    if not final_path:
        raise ValueError("请选择表面终态/结果目录。")

    if _is_calculation_path(final_path):
        result = calc_surface_excess_energy_from_directory(
            slab_path=final_path,
            simple_substance_db=reference_db,
            n_surfaces=n_surfaces,
        )
        result.update({
            "status": "ok",
            "initial_path": initial_path,
            "initial_role": "not_required" if not initial_path else "user_reference",
        })
        results = [_with_surface_alias(result)]
        kind = "surface_single"
    else:
        results = calc_surface_excess_energies_batch(
            root_dir=final_path,
            simple_substance_db=reference_db,
            n_surfaces=n_surfaces,
            strict=strict,
        )
        for row in results:
            row["initial_path"] = initial_path
            row["initial_role"] = "not_required" if not initial_path else "user_reference"
            _with_surface_alias(row)
        kind = "surface_batch"

    ok_count = sum(1 for row in results if row.get("status") == "ok")
    return {
        "kind": kind,
        "formula": "E_surface = (E_slab - sum(n_i * mu_i)) / n_surfaces",
        "reference_summary": "mu_i from simple_substance_database",
        "summary": {
            "total": len(results),
            "ok": ok_count,
            "errors": len(results) - ok_count,
        },
        "results": results,
    }


def _calculate_adsorption(payload):
    system = _resolve_path(payload.get("finalPath") or payload.get("system"))
    slab = _resolve_path(payload.get("initialPath") or payload.get("slab"))
    adsorbate = _resolve_path(payload.get("adsorbatePath") or payload.get("adsorbate"))
    n_ads = int(payload.get("nAdsorbate") or 1)
    E_system, system_energy_file, system_converged = _energy_from_path(system)
    E_slab, slab_energy_file, slab_converged = _energy_from_path(slab)
    E_adsorbate, adsorbate_energy_file, adsorbate_converged = _energy_from_path(adsorbate)
    result = calc_adsorption_energy(E_system, E_slab, E_adsorbate, n_ads)
    result.update({
        "status": "ok",
        "label": payload.get("label") or os.path.basename(os.path.normpath(system)),
        "reference_energies": "E_adsorbate",
        "system_path": system,
        "slab_path": slab,
        "adsorbate_path": adsorbate,
        "system_energy_file": system_energy_file,
        "slab_energy_file": slab_energy_file,
        "adsorbate_energy_file": adsorbate_energy_file,
        "system_is_converged": system_converged,
        "slab_is_converged": slab_converged,
        "adsorbate_is_converged": adsorbate_converged,
    })
    return {
        "kind": "adsorption",
        "formula": "E_ads = E_final - E_initial - n_adsorbate * E_adsorbate",
        "reference_summary": "E_adsorbate from selected adsorbate calculation",
        "summary": {"total": 1, "ok": 1, "errors": 0},
        "results": [result],
    }


def _calculate_doping(payload):
    final_path = _resolve_path(payload.get("finalPath") or payload.get("doped"))
    pristine = _resolve_path(payload.get("initialPath") or payload.get("pristine"))
    n_dopant = int(payload.get("nDopant") or 1)
    dopant_atom_index = int(payload.get("dopantAtomIndex") or 0)
    host_atom_index = int(payload.get("hostAtomIndex") or 0)
    reference_db = _resolve_path(payload.get("referenceDb")) or str(PROJECT_ROOT / "simple_substance_database")
    uses_charge, charge, efermi, correction = _charge_inputs(payload)

    if not final_path:
        raise ValueError("请选择掺杂后终态目录。")
    if not pristine:
        raise ValueError("请选择掺杂前初态目录。")
    if dopant_atom_index <= 0:
        raise ValueError("请输入终态结构中的掺杂原子序号。")
    if host_atom_index <= 0:
        raise ValueError("请输入初态结构中的宿主元素原子序号。")

    E_pristine, pristine_energy_file, pristine_converged = _energy_from_path(pristine)
    host_element, host_structure_file = _read_atom_element(pristine, host_atom_index)
    references = build_simple_substance_database(reference_db)
    mu_host = _reference_mu(references, host_element)

    doped_dirs = _calculation_dirs_from_path(final_path)

    if not doped_dirs:
        raise ValueError(f"未在终态目录下找到包含 POSCAR/CONTCAR 和 OUTCAR/log 的计算子目录: {final_path}")

    results = []
    for doped in doped_dirs:
        try:
            E_doped, doped_energy_file, doped_converged = _energy_from_path(doped)
            dopant_element, doped_structure_file = _read_atom_element(doped, dopant_atom_index)
            mu_dopant = _reference_mu(references, dopant_element)
            result = calc_doping_energy(
                E_doped=E_doped,
                E_pristine=E_pristine,
                n_dopant=n_dopant,
                mu_dopant=mu_dopant,
                mu_host=mu_host,
                charge_state=charge,
                efermi=efermi,
                correction=correction,
            )
            _remove_charge_fields_if_unused(result, uses_charge)
            result.update({
                "status": "ok",
                "dopant_element": dopant_element,
                "host_element": host_element,
                "dopant_atom_index": dopant_atom_index,
                "host_atom_index": host_atom_index,
                "reference_energies": f"mu_dopant({dopant_element}); mu_host({host_element})",
                "doped_energy_file": doped_energy_file,
                "doped_structure_file": doped_structure_file,
                "pristine_energy_file": pristine_energy_file,
                "pristine_structure_file": host_structure_file,
                "doped_is_converged": doped_converged,
                "pristine_is_converged": pristine_converged,
                "reference_database": reference_db,
            })
        except Exception as exc:
            result = {
                "status": "error",
                "host_element": host_element,
                "host_atom_index": host_atom_index,
                "reference_energies": f"mu_host({host_element})",
                "error": str(exc),
            }
        result.update({
            "label": payload.get("label") or os.path.basename(os.path.normpath(doped)),
            "doped_path": doped,
            "pristine_path": pristine,
        })
        results.append(result)

    ok_count = sum(1 for row in results if row.get("status") == "ok")
    return {
        "kind": "doping",
        "formula": _formula_with_optional_charge(
            "E_doping = E_final - E_initial - n_dopant * mu_dopant + n_host * mu_host"
        ) if uses_charge else "E_doping = E_final - E_initial - n_dopant * mu_dopant + n_host * mu_host",
        "reference_summary": "dopant/host elements are identified by atom index, then mu is read from simple_substance_database",
        "summary": {"total": len(results), "ok": ok_count, "errors": len(results) - ok_count},
        "results": results,
    }


def _calculate_defect(payload):
    final_path = _resolve_path(payload.get("finalPath"))
    pristine = _resolve_path(payload.get("initialPath"))
    removed_atom_index = int(payload.get("removedAtomIndex") or 0)
    n_removed = int(payload.get("nRemoved") or 1)
    reference_db = _resolve_path(payload.get("referenceDb")) or str(PROJECT_ROOT / "simple_substance_database")
    uses_charge, charge, efermi, correction = _charge_inputs(payload)

    if not final_path:
        raise ValueError("请选择缺陷后终态目录。")
    if not pristine:
        raise ValueError("请选择缺陷前初态目录。")
    if removed_atom_index <= 0:
        raise ValueError("请输入初态结构中被移除元素的原子序号。")
    if n_removed <= 0:
        raise ValueError("被移除原子数必须为正整数。")

    E_pristine, pristine_energy_file, pristine_converged = _energy_from_path(pristine)
    defect_element, pristine_structure_file = _read_atom_element(pristine, removed_atom_index)
    references = build_simple_substance_database(reference_db)
    mu_removed = _reference_mu(references, defect_element)
    defect_dirs = _calculation_dirs_from_path(final_path)

    if not defect_dirs:
        raise ValueError(f"未在终态目录下找到包含 POSCAR/CONTCAR 和 OUTCAR/log 的计算子目录: {final_path}")

    results = []
    for defect_dir in defect_dirs:
        try:
            E_defect, defect_energy_file, defect_converged = _energy_from_path(defect_dir)
            energy_diff = E_defect - E_pristine
            chemical_potential_term = n_removed * mu_removed
            charge_correction_term = charge * efermi + correction
            formation_energy = energy_diff + chemical_potential_term + charge_correction_term
            result = {
                "status": "ok",
                "formation_energy_eV": formation_energy,
                "E_defect": E_defect,
                "E_pristine": E_pristine,
                "energy_diff": energy_diff,
                "n_removed": n_removed,
                "defect_element": defect_element,
                "removed_atom_index": removed_atom_index,
                "mu_removed": mu_removed,
                "chemical_potential_term": chemical_potential_term,
                "reference_energies": f"mu_removed({defect_element})",
                "defect_energy_file": defect_energy_file,
                "pristine_energy_file": pristine_energy_file,
                "pristine_structure_file": pristine_structure_file,
                "defect_is_converged": defect_converged,
                "pristine_is_converged": pristine_converged,
                "reference_database": reference_db,
            }
            if uses_charge:
                result.update({
                    "charge_state": charge,
                    "efermi": efermi,
                    "charge_correction_term": charge_correction_term,
                    "correction": correction,
                })
            else:
                result.pop("chemical_potential_term", None)
        except Exception as exc:
            result = {
                "status": "error",
                "defect_element": defect_element,
                "removed_atom_index": removed_atom_index,
                "reference_energies": f"mu_removed({defect_element})",
                "error": str(exc),
            }
        result.update({
            "label": payload.get("label") or os.path.basename(os.path.normpath(defect_dir)),
            "defect_path": defect_dir,
            "pristine_path": pristine,
        })
        results.append(result)

    ok_count = sum(1 for row in results if row.get("status") == "ok")
    return {
        "kind": "defect",
        "formula": _formula_with_optional_charge(
            "E_defect = E_final - E_initial + n_removed * mu_removed"
        ) if uses_charge else "E_defect = E_final - E_initial + n_removed * mu_removed",
        "reference_summary": "removed element is identified by atom index in the initial structure, then mu is read from simple_substance_database",
        "summary": {"total": len(results), "ok": ok_count, "errors": len(results) - ok_count},
        "results": results,
    }


def calculate_energy(payload):
    energy_type = payload.get("energyType")
    if energy_type in {"surface", "surface_batch", "surface_single"}:
        data = _calculate_surface(payload)
    elif energy_type == "adsorption":
        data = _calculate_adsorption(payload)
    elif energy_type == "doping":
        data = _calculate_doping(payload)
    elif energy_type == "defect":
        data = _calculate_defect(payload)
    else:
        raise ValueError(f"未知能量类型: {energy_type}")

    saved = {}
    csv_path = _write_csv_file(payload.get("csvPath"), data["results"])
    json_path = _write_json_file(payload.get("jsonPath"), data)
    if csv_path:
        saved["csv"] = csv_path
    if json_path:
        saved["json"] = json_path
    data["saved"] = saved
    return data


def browse_path(path):
    path = _resolve_path(path) or str(PROJECT_ROOT)
    if not os.path.exists(path):
        path = os.path.dirname(path) or str(PROJECT_ROOT)
    if os.path.isfile(path):
        path = os.path.dirname(path)
    path = os.path.abspath(path)

    entries = []
    try:
        names = sorted(os.listdir(path), key=lambda name: (not os.path.isdir(os.path.join(path, name)), name.lower()))
    except PermissionError as exc:
        raise PermissionError(f"无权限访问目录: {path}") from exc

    for name in names:
        if name in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
            continue
        if name.startswith(".") and name not in {".gitignore"}:
            continue
        full_path = os.path.join(path, name)
        entries.append({
            "name": name,
            "path": full_path,
            "isDir": os.path.isdir(full_path),
        })

    parent = os.path.dirname(path) if os.path.dirname(path) != path else path
    quick_roots = []
    for item in [
            "/Users/simon/Documents/AISI/Cr2O3/Results",
            "/Users/simon/Documents/ANU/multi_agent"]:
        if item not in quick_roots:
            quick_roots.append(item)

    return {
        "path": path,
        "parent": parent,
        "entries": entries,
        "quickRoots": quick_roots,
    }


class MatKitUIHandler(BaseHTTPRequestHandler):
    server_version = "MatKitUI/0.1"

    def log_message(self, fmt, *args):
        if getattr(self.server, "quiet", False):
            return
        super().log_message(fmt, *args)

    def _send_json(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_error_json(self, exc, status=400):
        self._send_json({"ok": False, "error": str(exc)}, status=status)

    def _serve_static(self, route):
        if route in {"", "/"}:
            route = "/index.html"
        clean = posixpath.normpath(urllib.parse.unquote(route)).lstrip("/")
        path = (STATIC_DIR / clean).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        raw = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        route = "/index.html" if parsed.path in {"", "/"} else parsed.path
        clean = posixpath.normpath(urllib.parse.unquote(route)).lstrip("/")
        path = (STATIC_DIR / clean).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/browse":
            params = urllib.parse.parse_qs(parsed.query)
            try:
                data = browse_path(params.get("path", [""])[0])
                self._send_json({"ok": True, **data})
            except Exception as exc:
                self._send_error_json(exc)
            return
        if parsed.path == "/api/defaults":
            self._send_json({
                "ok": True,
                "projectRoot": str(PROJECT_ROOT),
                "defaultReferenceDb": str(PROJECT_ROOT / "simple_substance_database"),
                "defaultOutputDir": str(PROJECT_ROOT / "analysis_outputs"),
            })
            return
        self._serve_static(parsed.path)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/calculate":
            self.send_error(404)
            return
        try:
            payload = _read_json_body(self)
            data = calculate_energy(payload)
            self._send_json({"ok": True, **data})
        except Exception as exc:
            self._send_error_json(exc)


def run_ui_server(host="127.0.0.1", port=8765, open_browser=False, quiet=False):
    """Start the local MatKit web UI server."""
    server = ThreadingHTTPServer((host, int(port)), MatKitUIHandler)
    server.quiet = quiet
    url = f"http://{host}:{int(port)}"
    print(f"MatKit UI running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMatKit UI stopped.")
    finally:
        server.server_close()
