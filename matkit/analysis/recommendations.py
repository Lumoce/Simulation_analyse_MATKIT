"""
Rule-based next-step recommendations.

The LLM assistant can be plugged in for richer language, but these rules keep
the software useful offline and make common quality checks deterministic.
"""

from __future__ import annotations

from typing import Dict, List, Mapping


def suggest_next_analyses(analysis_result: Mapping) -> List[Dict[str, str]]:
    """Return practical next-step suggestions from an adsorption analysis."""
    suggestions: List[Dict[str, str]] = []

    contacts = list(analysis_result.get("contacts", []))
    if not contacts:
        suggestions.append({
            "priority": "high",
            "topic": "adsorbate identification",
            "suggestion": (
                "No adsorbate-surface contacts were detected. Check the surface "
                "element, adsorption cutoff, and whether the adsorbate formula "
                "matches the structure."
            ),
        })

    metrics = analysis_result.get("so4_metrics")
    if metrics:
        n_bound = int(metrics.get("n_surface_bound_o", 0))
        if n_bound == 0:
            suggestions.append({
                "priority": "high",
                "topic": "SO4 binding mode",
                "suggestion": (
                    "No Cu-bound SO4 oxygen was detected. Increase the surface "
                    "cutoff cautiously or inspect whether SO4 is detached."
                ),
            })
        elif n_bound == 1:
            suggestions.append({
                "priority": "medium",
                "topic": "SO4 binding mode",
                "suggestion": (
                    "Only one Cu-bound O was detected. Compare this monodentate "
                    "geometry with bidentate candidates before ranking stability."
                ),
            })
        else:
            suggestions.append({
                "priority": "medium",
                "topic": "SO4 binding mode",
                "suggestion": (
                    "Multiple Cu-bound O atoms were detected. Use the O-S-O angle "
                    "and Cu-O distances to classify bridge or chelating adsorption."
                ),
            })

        s_height = float(metrics.get("s_to_surface_abs_z_A", 0.0))
        if s_height > 3.5:
            suggestions.append({
                "priority": "medium",
                "topic": "geometry sanity",
                "suggestion": (
                    "The S atom is far from the reference surface plane. Verify the "
                    "chosen top/bottom surface and whether the structure is weakly "
                    "adsorbed or still unrelaxed."
                ),
            })

        abnormal_bonds = [
            bond for bond in metrics.get("all_s_o_bonds", [])
            if bond["s_o_distance_A"] < 1.35 or bond["s_o_distance_A"] > 1.75
        ]
        if abnormal_bonds:
            suggestions.append({
                "priority": "medium",
                "topic": "SO4 internal geometry",
                "suggestion": (
                    "At least one S-O bond is outside a typical sulfate range. "
                    "Check whether the structure is converged and whether bond "
                    "cutoffs included the intended atoms."
                ),
            })

    suggestions.append({
        "priority": "normal",
        "topic": "next calculations",
        "suggestion": (
            "For a complete corrosion-inhibitor workflow, follow geometry analysis "
            "with adsorption energy, Bader charge transfer, differential charge "
            "density, and a clean-vs-adsorbed surface displacement comparison."
        ),
    })

    return suggestions
