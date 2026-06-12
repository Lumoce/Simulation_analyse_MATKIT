from pathlib import Path
import tempfile
import unittest

import numpy as np

from matkit.analysis import analyze_so4_cu2o
from matkit.parsers import read_poscar
from matkit.structure import identify_molecules


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_POSCAR = ROOT / "examples" / "so4_cu2o" / "POSCAR"


class AdsorptionAnalysisTests(unittest.TestCase):
    def test_read_poscar_expands_atom_elements(self):
        data = read_poscar(EXAMPLE_POSCAR)

        self.assertEqual(data["total_atoms"], 12)
        self.assertEqual(data["atom_elements"][:4], ["Cu", "Cu", "Cu", "Cu"])
        self.assertEqual(data["atom_elements"][7:11], ["O", "O", "O", "O"])
        self.assertEqual(data["atom_elements"][11], "S")

    def test_read_poscar_preserves_duplicate_species_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "POSCAR"
            path.write_text(
                "\n".join([
                    "duplicate species",
                    "1.0",
                    "1 0 0",
                    "0 1 0",
                    "0 0 1",
                    "Cr Zr Cr O",
                    "2 1 2 1",
                    "Direct",
                    "0 0 0",
                    "0 0 0",
                    "0 0 0",
                    "0 0 0",
                    "0 0 0",
                    "0 0 0",
                ])
                + "\n"
            )
            data = read_poscar(path)

        self.assertEqual(data["n_atoms"], {"Cr": 4, "Zr": 1, "O": 1})
        self.assertEqual(data["atom_elements"], ["Cr", "Cr", "Zr", "Cr", "Cr", "O"])
        self.assertEqual(data["atom_elements"][2], "Zr")

    def test_so4_cu2o_key_metrics(self):
        result = analyze_so4_cu2o(str(EXAMPLE_POSCAR))
        metrics = result["so4_metrics"]

        self.assertEqual(metrics["n_surface_bound_o"], 2)
        self.assertEqual(metrics["surface_bound_o_indices1"], [8, 9])
        self.assertAlmostEqual(metrics["s_to_surface_signed_z_A"], 2.0)

        bond_lengths = sorted(row["s_o_distance_A"] for row in metrics["adsorbed_s_o_bonds"])
        expected_bond = np.sqrt(0.8**2 + 1.1**2)
        for value in bond_lengths:
            self.assertAlmostEqual(value, expected_bond)

        self.assertEqual(len(metrics["o_s_o_angles"]), 1)
        self.assertAlmostEqual(metrics["o_s_o_angles"][0]["angle_deg"], 72.055, places=2)

    def test_identify_molecules_length_validation_message(self):
        with self.assertRaises(ValueError) as ctx:
            identify_molecules(["H"], np.zeros((2, 3)), {})

        self.assertIn("atoms has 1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
