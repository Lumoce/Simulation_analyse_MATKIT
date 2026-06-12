import os
import tempfile
import unittest

from matkit.ui.server import _calculate_doping


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

    def test_doping_final_parent_traverses_task_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pristine = os.path.join(tmpdir, "pristine")
            final_root = os.path.join(tmpdir, "Doping_energy")
            self._write_calc(pristine, -100.0)
            self._write_calc(os.path.join(final_root, "task.1_top"), -101.0)
            self._write_calc(os.path.join(final_root, "task.2_bridge"), -102.0)

            data = _calculate_doping({
                "finalPath": final_root,
                "initialPath": pristine,
                "nDopant": 1,
                "muDopant": -3.0,
                "muHost": -4.0,
            })

            self.assertEqual(data["summary"], {"total": 2, "ok": 2, "errors": 0})
            self.assertEqual([row["task_number"] for row in data["results"]], ["1", "2"])
            self.assertEqual([row["task_suffix"] for row in data["results"]], ["top", "bridge"])
            self.assertTrue(all("reference_energies" in row for row in data["results"]))
            self.assertTrue(all("formula_terms" in row for row in data["results"]))


if __name__ == "__main__":
    unittest.main()
