"""
Local web UI server for MatKit.

This module intentionally uses only Python's standard library so the UI works in
the existing analysis environment without adding web framework dependencies.
"""

import csv
import json
import mimetypes
import os
import posixpath
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from matkit.energy import (
    calc_adsorption_energy,
    calc_doping_energy,
    calc_surface_excess_energies_batch,
    calc_surface_excess_energy_from_directory,
    find_energy_file,
    find_structure_file,
    read_calculation_energy,
)


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
        "E_slab",
        "E_slab_ads",
        "E_doped",
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
    return row


def _is_calculation_path(path):
    if os.path.isfile(path):
        return find_energy_file(path) is not None
    return find_structure_file(path) is not None and find_energy_file(path) is not None


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
        "summary": {"total": 1, "ok": 1, "errors": 0},
        "results": [result],
    }


def _calculate_doping(payload):
    doped = _resolve_path(payload.get("finalPath") or payload.get("doped"))
    pristine = _resolve_path(payload.get("initialPath") or payload.get("pristine"))
    n_dopant = int(payload.get("nDopant") or 1)
    mu_dopant = float(payload.get("muDopant") or 0)
    mu_host = float(payload.get("muHost") or 0)
    charge = int(payload.get("chargeState") or 0)
    efermi = float(payload.get("efermi") or 0)
    correction = float(payload.get("correction") or 0)
    E_doped, doped_energy_file, doped_converged = _energy_from_path(doped)
    E_pristine, pristine_energy_file, pristine_converged = _energy_from_path(pristine)
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
    result.update({
        "status": "ok",
        "label": payload.get("label") or os.path.basename(os.path.normpath(doped)),
        "doped_path": doped,
        "pristine_path": pristine,
        "doped_energy_file": doped_energy_file,
        "pristine_energy_file": pristine_energy_file,
        "doped_is_converged": doped_converged,
        "pristine_is_converged": pristine_converged,
    })
    return {
        "kind": "doping",
        "summary": {"total": 1, "ok": 1, "errors": 0},
        "results": [result],
    }


def calculate_energy(payload):
    energy_type = payload.get("energyType")
    if energy_type in {"surface", "surface_batch", "surface_single"}:
        data = _calculate_surface(payload)
    elif energy_type == "adsorption":
        data = _calculate_adsorption(payload)
    elif energy_type == "doping":
        data = _calculate_doping(payload)
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
            str(PROJECT_ROOT),
            str(Path.home()),
            "/Users/simon/Documents/ANU/multi_agent/Analyse_Software",
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
