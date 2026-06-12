import os
import tempfile
import unittest

from matkit.ui.server import _calculate_defect, _calculate_doping


class UIServerEnergyTests(unittest.TestCase):
    def _write_poscar(self, path, elements, counts):
        total_atoms = sum(counts)
        with open(path, "w") as f:
            f.write("test structure\n")
            f.write("1.0\n")
            f.write("1.0 0.0 0.0\n")
            f.write("0.0 1.0 0.0\n")
            f.write("0.0 0.0 1.0\n")
            f.write(" ".join(elements) + "\n")
            f.write(" ".join(str(count) for count in counts) + "\n")
            f.write("Direct\n")
            for _ in range(total_atoms):
                f.write("0.0 0.0 0.0\n")

    def _write_outcar(self, path, energy):
        with open(path, "w") as f:
            f.write(f" free  energy   TOTEN  =      {energy:.8f} eV\n")
            f.write(" reached required accuracy - stopping structural energy minimisation\n")

    def _write_calc(self, directory, energy):
        os.makedirs(directory)
        self._write_poscar(os.path.join(directory, "POSCAR"), ["Cr", "O"], [2, 3])
        self._write_outcar(os.path.join(directory, "OUTCAR"), energy)

    def _write_doped_calc(self, directory, energy):
        os.makedirs(directory)
        self._write_poscar(os.path.join(directory, "POSCAR"), ["Cr", "Ni", "O"], [1, 1, 3])
        self._write_outcar(os.path.join(directory, "OUTCAR"), energy)

    def _write_reference_db(self, tmpdir):
        db_dir = os.path.join(tmpdir, "simple_substance_database")
        cr_dir = os.path.join(db_dir, "Cr")
        ni_dir = os.path.join(db_dir, "Ni")
        o_dir = os.path.join(db_dir, "O")
        for path in [cr_dir, ni_dir, o_dir]:
            os.makedirs(path)
        self._write_poscar(os.path.join(cr_dir, "POSCAR"), ["Cr"], [1])
        self._write_outcar(os.path.join(cr_dir, "OUTCAR"), -4.0)
        self._write_poscar(os.path.join(ni_dir, "POSCAR"), ["Ni"], [1])
        self._write_outcar(os.path.join(ni_dir, "OUTCAR"), -3.0)
        self._write_poscar(os.path.join(o_dir, "POSCAR"), ["O"], [1])
        self._write_outcar(os.path.join(o_dir, "OUTCAR"), -5.0)
        return db_dir

    def test_doping_final_parent_traverses_task_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pristine = os.path.join(tmpdir, "pristine")
            final_root = os.path.join(tmpdir, "Doping_energy")
            db_dir = self._write_reference_db(tmpdir)
            self._write_calc(pristine, -100.0)
            self._write_doped_calc(os.path.join(final_root, "task.1_top"), -101.0)
            self._write_doped_calc(os.path.join(final_root, "task.2_bridge"), -102.0)

            data = _calculate_doping({
                "finalPath": final_root,
                "initialPath": pristine,
                "referenceDb": db_dir,
                "nDopant": 1,
                "dopantAtomIndex": 2,
                "hostAtomIndex": 1,
            })

            self.assertEqual(data["summary"], {"total": 2, "ok": 2, "errors": 0})
            self.assertEqual([row["dopant_element"] for row in data["results"]], ["Ni", "Ni"])
            self.assertEqual([row["host_element"] for row in data["results"]], ["Cr", "Cr"])
            self.assertNotIn("formula_terms", data["results"][0])
            self.assertTrue(all("reference_energies" in row for row in data["results"]))
            self.assertIn("E_doping", data["formula"])

    def test_defect_uses_removed_atom_index_and_reference_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pristine = os.path.join(tmpdir, "pristine")
            final_root = os.path.join(tmpdir, "Defect_energy")
            db_dir = self._write_reference_db(tmpdir)
            self._write_calc(pristine, -100.0)
            self._write_calc(os.path.join(final_root, "task.1_vac"), -96.0)

            data = _calculate_defect({
                "finalPath": final_root,
                "initialPath": pristine,
                "referenceDb": db_dir,
                "nRemoved": 1,
                "removedAtomIndex": 1,
            })

            self.assertEqual(data["summary"], {"total": 1, "ok": 1, "errors": 0})
            row = data["results"][0]
            self.assertEqual(row["defect_element"], "Cr")
            self.assertAlmostEqual(row["formation_energy_eV"], 0.0)
            self.assertNotIn("charge_state", row)
            self.assertIn("E_defect", data["formula"])


if __name__ == "__main__":
    unittest.main()
