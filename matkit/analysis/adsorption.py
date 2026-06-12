"""
Adsorption geometry workflows.

The module is intentionally dependency-light: it uses the existing MatKit
POSCAR parser plus NumPy geometry routines. The SO4/Cu2O workflow is a
specialized wrapper around a generic adsorbate-on-surface analysis so the same
pattern can be reused for organic corrosion inhibitors.
"""

from __future__ import annotations

from collections import Counter
import itertools
import math
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from matkit.parsers import read_poscar
from matkit.structure import calc_angle, calc_distance, identify_molecules, identify_surface_layers


DEFAULT_INTERNAL_BOND_CUTOFFS: Dict[Tuple[str, str], float] = {
    ("H", "H"): 0.9,
    ("C", "H"): 1.25,
    ("N", "H"): 1.25,
    ("O", "H"): 1.25,
    ("S", "H"): 1.55,
    ("C", "C"): 1.85,
    ("C", "N"): 1.65,
    ("C", "O"): 1.65,
    ("C", "S"): 1.95,
    ("N", "N"): 1.6,
    ("N", "O"): 1.65,
    ("N", "S"): 1.9,
    ("O", "O"): 1.75,
    ("S", "O"): 1.9,
    ("S", "S"): 2.2,
    ("P", "O"): 1.9,
    ("P", "S"): 2.15,
    ("Cl", "C"): 2.0,
}


def _normalize_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def _normalized_cutoffs(
    cutoffs: Optional[Mapping[Tuple[str, str], float]] = None,
) -> Dict[Tuple[str, str], float]:
    source = DEFAULT_INTERNAL_BOND_CUTOFFS if cutoffs is None else cutoffs
    return {_normalize_pair(a, b): float(value) for (a, b), value in source.items()}


def _get_cutoff(cutoffs: Mapping[Tuple[str, str], float], a: str, b: str) -> Optional[float]:
    return cutoffs.get(_normalize_pair(a, b))


def _formula_counts(formula: str) -> Counter:
    if not formula or not isinstance(formula, str):
        raise ValueError("formula must be a non-empty string")
    matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not matches:
        raise ValueError(f"Cannot parse formula: {formula}")
    counts: Counter = Counter()
    for element, number in matches:
        counts[element] += int(number) if number else 1
    return counts


def _counts(elements: Iterable[str]) -> Counter:
    return Counter(str(element) for element in elements)


def _atom_elements(structure: Mapping) -> List[str]:
    if "atom_elements" in structure:
        return list(structure["atom_elements"])
    atom_elements: List[str] = []
    for elem in structure["elements"]:
        atom_elements.extend([elem] * int(structure["n_atoms"][elem]))
    return atom_elements


def _surface_side_from_adsorbate(coords: np.ndarray, adsorbate_indices: Sequence[int]) -> str:
    if not adsorbate_indices:
        return "top"
    z_mid = 0.5 * (float(np.min(coords[:, 2])) + float(np.max(coords[:, 2])))
    ads_z = float(np.mean(coords[list(adsorbate_indices), 2]))
    return "top" if ads_z >= z_mid else "bottom"


def _select_surface_reference(
    atom_elements: Sequence[str],
    coords: np.ndarray,
    surface_element: str,
    layer_tolerance: float,
    n_layers: int,
    side: str,
    adsorbate_indices: Sequence[int],
) -> Dict:
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if side not in {"top", "bottom", "auto"}:
        raise ValueError("surface_side must be one of: top, bottom, auto")

    layers = identify_surface_layers(atom_elements, coords, surface_element, layer_tolerance)
    if side == "auto":
        side = _surface_side_from_adsorbate(coords, adsorbate_indices)

    ordered = layers if side == "bottom" else list(reversed(layers))
    selected_layers = ordered[:n_layers]
    if not selected_layers:
        raise ValueError(f"No {surface_element} surface layers were selected")

    surface_indices: List[int] = []
    for layer in selected_layers:
        surface_indices.extend(int(index) for index in layer["indices"])

    outermost = selected_layers[0]
    plane_z = float(outermost["z_avg"])
    return {
        "side": side,
        "surface_element": surface_element,
        "layers": layers,
        "selected_layers": selected_layers,
        "surface_indices": sorted(surface_indices),
        "reference_plane_z": plane_z,
        "outermost_layer": outermost,
    }


def _molecule_distance_to_indices(
    molecule: Mapping,
    coords: np.ndarray,
    reference_indices: Sequence[int],
) -> float:
    if not reference_indices:
        return math.inf
    mol_indices = list(molecule["indices"])
    if not mol_indices:
        return math.inf
    ref_coords = coords[list(reference_indices)]
    distances = []
    for idx in mol_indices:
        atom_distances = np.linalg.norm(ref_coords - coords[int(idx)], axis=1)
        distances.append(float(np.min(atom_distances)))
    return min(distances) if distances else math.inf


def _select_adsorbate(
    molecules: Sequence[Mapping],
    formula: Optional[str],
    surface_element: Optional[str],
    coords: np.ndarray,
    surface_indices: Optional[Sequence[int]] = None,
    max_adsorbate_atoms: int = 120,
) -> Mapping:
    candidates: List[Mapping] = []

    if formula:
        target = _formula_counts(formula)
        candidates = [
            mol for mol in molecules
            if _counts(mol["elements"]) == target
        ]
    else:
        for mol in molecules:
            elements = list(mol["elements"])
            if len(elements) <= 1 or len(elements) > max_adsorbate_atoms:
                continue
            if surface_element and all(elem == surface_element for elem in elements):
                continue
            candidates.append(mol)

    if not candidates:
        label = formula if formula else "an adsorbate candidate"
        raise ValueError(f"Could not identify {label}. Check formula or bond cutoffs.")

    if surface_indices:
        return min(
            candidates,
            key=lambda mol: _molecule_distance_to_indices(mol, coords, surface_indices),
        )

    return max(candidates, key=lambda mol: len(mol["indices"]))


def _internal_bonds(
    molecule: Mapping,
    coords: np.ndarray,
    cutoffs: Mapping[Tuple[str, str], float],
) -> List[Dict]:
    rows: List[Dict] = []
    mol_indices = list(molecule["indices"])
    mol_elements = list(molecule["elements"])
    for local_i, local_j in itertools.combinations(range(len(mol_indices)), 2):
        elem_i = mol_elements[local_i]
        elem_j = mol_elements[local_j]
        cutoff = _get_cutoff(cutoffs, elem_i, elem_j)
        if cutoff is None:
            continue
        global_i = int(mol_indices[local_i])
        global_j = int(mol_indices[local_j])
        distance = float(calc_distance(coords[global_i], coords[global_j]))
        if distance <= cutoff:
            rows.append({
                "index0_i": global_i,
                "index0_j": global_j,
                "index1_i": global_i + 1,
                "index1_j": global_j + 1,
                "element_i": elem_i,
                "element_j": elem_j,
                "distance_A": distance,
                "cutoff_A": float(cutoff),
            })
    rows.sort(key=lambda item: (item["element_i"], item["element_j"], item["distance_A"]))
    return rows


def _adsorption_contacts(
    molecule: Mapping,
    coords: np.ndarray,
    atom_elements: Sequence[str],
    surface_indices: Sequence[int],
    surface_cutoff: float,
) -> List[Dict]:
    contacts: List[Dict] = []
    if not surface_indices:
        return contacts

    surface_coords = coords[list(surface_indices)]
    for local_index, global_index in enumerate(molecule["indices"]):
        global_index = int(global_index)
        distances = np.linalg.norm(surface_coords - coords[global_index], axis=1)
        within = np.where(distances <= surface_cutoff)[0]
        for ref_pos in within:
            surface_index = int(surface_indices[int(ref_pos)])
            contacts.append({
                "adsorbate_index0": global_index,
                "adsorbate_index1": global_index + 1,
                "adsorbate_element": molecule["elements"][local_index],
                "surface_index0": surface_index,
                "surface_index1": surface_index + 1,
                "surface_element": atom_elements[surface_index],
                "distance_A": float(distances[int(ref_pos)]),
            })
    contacts.sort(key=lambda item: (item["distance_A"], item["adsorbate_index0"]))
    return contacts


def analyze_adsorbate_geometry(
    poscar_path: str,
    adsorbate_formula: Optional[str] = None,
    surface_element: str = "Cu",
    internal_cutoffs: Optional[Mapping[Tuple[str, str], float]] = None,
    surface_cutoff: float = 2.5,
    layer_tolerance: float = 0.5,
    n_surface_layers: int = 1,
    surface_side: str = "auto",
    max_adsorbate_atoms: int = 120,
) -> Dict:
    """
    Identify an adsorbate and its contacts to a crystalline surface.

    Parameters use VASP/POSCAR conventions and atom indices in the returned
    dictionaries include both 0-based and 1-based forms.
    """
    if surface_cutoff <= 0:
        raise ValueError("surface_cutoff must be positive")

    structure = read_poscar(poscar_path)
    coords = np.asarray(structure["coords"], dtype=float)
    atom_elements = _atom_elements(structure)
    cutoffs = _normalized_cutoffs(internal_cutoffs)

    molecules = identify_molecules(atom_elements, coords, cutoffs)

    preliminary_adsorbate = None
    preliminary_indices: Sequence[int] = []
    if adsorbate_formula:
        matching = [
            mol for mol in molecules
            if _counts(mol["elements"]) == _formula_counts(adsorbate_formula)
        ]
        if matching:
            preliminary_adsorbate = max(matching, key=lambda mol: len(mol["indices"]))
            preliminary_indices = list(preliminary_adsorbate["indices"])

    surface = _select_surface_reference(
        atom_elements=atom_elements,
        coords=coords,
        surface_element=surface_element,
        layer_tolerance=layer_tolerance,
        n_layers=n_surface_layers,
        side=surface_side,
        adsorbate_indices=preliminary_indices,
    )

    adsorbate = preliminary_adsorbate or _select_adsorbate(
        molecules=molecules,
        formula=adsorbate_formula,
        surface_element=surface_element,
        coords=coords,
        surface_indices=surface["surface_indices"],
        max_adsorbate_atoms=max_adsorbate_atoms,
    )

    # Re-evaluate auto side if formula selection was only possible after the
    # surface reference pass.
    if surface_side == "auto" and not preliminary_indices:
        surface = _select_surface_reference(
            atom_elements=atom_elements,
            coords=coords,
            surface_element=surface_element,
            layer_tolerance=layer_tolerance,
            n_layers=n_surface_layers,
            side="auto",
            adsorbate_indices=list(adsorbate["indices"]),
        )

    contacts = _adsorption_contacts(
        molecule=adsorbate,
        coords=coords,
        atom_elements=atom_elements,
        surface_indices=surface["surface_indices"],
        surface_cutoff=surface_cutoff,
    )
    bonds = _internal_bonds(adsorbate, coords, cutoffs)

    ads_indices = [int(index) for index in adsorbate["indices"]]
    ads_coords = coords[ads_indices]
    plane_z = float(surface["reference_plane_z"])
    z_signed = ads_coords[:, 2] - plane_z

    return {
        "input_file": str(poscar_path),
        "adsorbate_formula_requested": adsorbate_formula,
        "adsorbate": {
            "formula": adsorbate["formula"],
            "indices0": ads_indices,
            "indices1": [idx + 1 for idx in ads_indices],
            "elements": list(adsorbate["elements"]),
            "n_atoms": len(ads_indices),
            "center_A": np.mean(ads_coords, axis=0).tolist(),
            "z_min_A": float(np.min(ads_coords[:, 2])),
            "z_max_A": float(np.max(ads_coords[:, 2])),
        },
        "surface": {
            "element": surface_element,
            "side": surface["side"],
            "reference_plane_z_A": plane_z,
            "surface_indices0": surface["surface_indices"],
            "surface_indices1": [idx + 1 for idx in surface["surface_indices"]],
            "outermost_layer": surface["outermost_layer"],
            "n_layers_used": n_surface_layers,
            "layer_tolerance_A": float(layer_tolerance),
        },
        "contacts": contacts,
        "internal_bonds": bonds,
        "height_stats_A": {
            "min_signed_z_to_surface": float(np.min(z_signed)),
            "max_signed_z_to_surface": float(np.max(z_signed)),
            "mean_signed_z_to_surface": float(np.mean(z_signed)),
            "min_abs_z_to_surface": float(np.min(np.abs(z_signed))),
            "max_abs_z_to_surface": float(np.max(np.abs(z_signed))),
            "mean_abs_z_to_surface": float(np.mean(np.abs(z_signed))),
        },
        "molecules_found": [
            {
                "formula": mol["formula"],
                "n_atoms": len(mol["indices"]),
                "indices0": [int(idx) for idx in mol["indices"]],
                "indices1": [int(idx) + 1 for idx in mol["indices"]],
            }
            for mol in molecules
        ],
        "settings": {
            "surface_cutoff_A": float(surface_cutoff),
            "internal_cutoffs": {f"{a}-{b}": value for (a, b), value in cutoffs.items()},
        },
    }


def analyze_so4_cu2o(
    poscar_path: str,
    surface_element: str = "Cu",
    surface_cutoff: float = 2.5,
    s_o_cutoff: float = 1.9,
    layer_tolerance: float = 0.5,
    n_surface_layers: int = 1,
    surface_side: str = "auto",
) -> Dict:
    """
    Specialized SO4 adsorbed on Cu2O analysis.

    Returns the requested metrics:
    - S-O bond lengths for O atoms bound to Cu surface atoms.
    - O-S-O angles between surface-bound O atoms.
    - Z-direction distance from S to the Cu2O reference surface plane.
    """
    cutoffs = dict(DEFAULT_INTERNAL_BOND_CUTOFFS)
    cutoffs[("S", "O")] = float(s_o_cutoff)

    result = analyze_adsorbate_geometry(
        poscar_path=poscar_path,
        adsorbate_formula="SO4",
        surface_element=surface_element,
        internal_cutoffs=cutoffs,
        surface_cutoff=surface_cutoff,
        layer_tolerance=layer_tolerance,
        n_surface_layers=n_surface_layers,
        surface_side=surface_side,
    )

    structure = read_poscar(poscar_path)
    coords = np.asarray(structure["coords"], dtype=float)
    ads = result["adsorbate"]
    ads_indices = ads["indices0"]
    ads_elements = ads["elements"]

    s_indices = [idx for idx, elem in zip(ads_indices, ads_elements) if elem == "S"]
    o_indices = [idx for idx, elem in zip(ads_indices, ads_elements) if elem == "O"]
    if len(s_indices) != 1:
        raise ValueError(f"Expected exactly one S atom in SO4, found {len(s_indices)}")
    if len(o_indices) != 4:
        raise ValueError(f"Expected four O atoms in SO4, found {len(o_indices)}")

    s_index = int(s_indices[0])
    surface_bound_o = sorted({
        int(contact["adsorbate_index0"])
        for contact in result["contacts"]
        if contact["adsorbate_element"] == "O"
        and contact["surface_element"] == surface_element
    })

    all_s_o_bonds = []
    for o_index in sorted(o_indices):
        distance = float(calc_distance(coords[s_index], coords[o_index]))
        nearest_contacts = [
            contact for contact in result["contacts"]
            if contact["adsorbate_index0"] == o_index
        ]
        all_s_o_bonds.append({
            "s_index0": s_index,
            "s_index1": s_index + 1,
            "o_index0": o_index,
            "o_index1": o_index + 1,
            "is_surface_bound_o": o_index in surface_bound_o,
            "s_o_distance_A": distance,
            "nearest_surface_distance_A": (
                float(nearest_contacts[0]["distance_A"]) if nearest_contacts else None
            ),
        })

    adsorbed_s_o_bonds = [
        row for row in all_s_o_bonds if row["is_surface_bound_o"]
    ]

    o_s_o_angles = []
    for o_i, o_j in itertools.combinations(surface_bound_o, 2):
        angle = float(calc_angle(coords[o_i], coords[s_index], coords[o_j]))
        o_s_o_angles.append({
            "o1_index0": int(o_i),
            "o1_index1": int(o_i) + 1,
            "s_index0": s_index,
            "s_index1": s_index + 1,
            "o2_index0": int(o_j),
            "o2_index1": int(o_j) + 1,
            "angle_deg": angle,
        })

    plane_z = float(result["surface"]["reference_plane_z_A"])
    s_z = float(coords[s_index, 2])
    s_z_signed = s_z - plane_z

    result["so4_metrics"] = {
        "s_index0": s_index,
        "s_index1": s_index + 1,
        "s_z_A": s_z,
        "surface_reference_z_A": plane_z,
        "s_to_surface_signed_z_A": s_z_signed,
        "s_to_surface_abs_z_A": abs(s_z_signed),
        "surface_bound_o_indices0": surface_bound_o,
        "surface_bound_o_indices1": [idx + 1 for idx in surface_bound_o],
        "n_surface_bound_o": len(surface_bound_o),
        "all_s_o_bonds": all_s_o_bonds,
        "adsorbed_s_o_bonds": adsorbed_s_o_bonds,
        "o_s_o_angles": o_s_o_angles,
    }

    result["warnings"] = []
    if not surface_bound_o:
        result["warnings"].append(
            "No SO4 oxygen atom was found within the surface cutoff. "
            "Increase --surface-cutoff or check the selected surface element."
        )
    if len(surface_bound_o) < 2:
        result["warnings"].append(
            "Fewer than two surface-bound O atoms were found, so no O-S-O "
            "angle between adsorbing O atoms can be reported."
        )

    return result


def flatten_so4_metrics(result: Mapping) -> List[Dict]:
    """Flatten SO4 metrics into CSV-friendly rows."""
    metrics = result["so4_metrics"]
    rows: List[Dict] = []
    rows.append({
        "metric": "S_to_surface_signed_z",
        "atom_indices_1based": str(metrics["s_index1"]),
        "value": metrics["s_to_surface_signed_z_A"],
        "unit": "A",
    })
    rows.append({
        "metric": "S_to_surface_abs_z",
        "atom_indices_1based": str(metrics["s_index1"]),
        "value": metrics["s_to_surface_abs_z_A"],
        "unit": "A",
    })
    for bond in metrics["adsorbed_s_o_bonds"]:
        rows.append({
            "metric": "adsorbed_O_S_bond",
            "atom_indices_1based": f"{bond['o_index1']}-{bond['s_index1']}",
            "value": bond["s_o_distance_A"],
            "unit": "A",
        })
    for angle in metrics["o_s_o_angles"]:
        rows.append({
            "metric": "adsorbed_O_S_O_angle",
            "atom_indices_1based": (
                f"{angle['o1_index1']}-{angle['s_index1']}-{angle['o2_index1']}"
            ),
            "value": angle["angle_deg"],
            "unit": "deg",
        })
    return rows
