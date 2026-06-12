import unittest
import os
import tempfile

from matkit.energy import (
    build_simple_substance_database,
    calc_adsorption_energy,
    calc_doping_energy,
    calc_surface_energy,
    calc_surface_excess_energies_batch,
    calc_surface_excess_energy_from_directory,
)


class EnergyTests(unittest.TestCase):
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

    def _write_oszicar_log(self, path, energy):
        with open(path, "w") as f:
            f.write(f"  1 F= {energy:.8E} E0= {energy:.8E}  d E = 0.00000000E+00\n")

    def test_adsorption_energy_formula(self):
        result = calc_adsorption_energy(-25.0, -20.0, -3.0, n_adsorbate=1)

        self.assertAlmostEqual(result["adsorption_energy_eV"], -2.0)
        self.assertAlmostEqual(result["adsorption_energy_per_molecule"], -2.0)

    def test_surface_energy_formula(self):
        result = calc_surface_energy(-20.0, -3.5, n_atoms_slab=6, surface_area=12.0)

        self.assertAlmostEqual(result["surface_energy_eV_A2"], 1.0 / 24.0)

    def test_doping_energy_formula_neutral(self):
        result = calc_doping_energy(
            E_doped=-101.0,
            E_pristine=-100.0,
            n_dopant=1,
            mu_dopant=-3.0,
            mu_host=-4.0,
        )

        self.assertAlmostEqual(result["formation_energy_eV"], -2.0)

    def test_simple_substance_surface_excess_energy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_dir = os.path.join(tmpdir, "simple_substance_database")
            cu_dir = os.path.join(db_dir, "Cu")
            o_dir = os.path.join(db_dir, "O2")
            slab_dir = os.path.join(tmpdir, "Surface_energy", "task.1_top")
            os.makedirs(cu_dir)
            os.makedirs(o_dir)
            os.makedirs(slab_dir)

            self._write_poscar(os.path.join(cu_dir, "POSCAR"), ["Cu"], [1])
            self._write_outcar(os.path.join(cu_dir, "OUTCAR"), -3.0)
            self._write_poscar(os.path.join(o_dir, "POSCAR"), ["O"], [2])
            self._write_oszicar_log(os.path.join(o_dir, "log"), -10.0)

            self._write_poscar(os.path.join(slab_dir, "POSCAR"), ["Cu", "O"], [2, 1])
            self._write_outcar(os.path.join(slab_dir, "OUTCAR"), -20.0)

            refs = build_simple_substance_database(db_dir)
            self.assertAlmostEqual(refs["Cu"]["mu_eV_per_atom"], -3.0)
            self.assertAlmostEqual(refs["O"]["mu_eV_per_atom"], -5.0)

            result = calc_surface_excess_energy_from_directory(
                slab_dir,
                simple_substance_db=db_dir,
                n_surfaces=2,
            )

            self.assertEqual(result["composition"], {"Cu": 2, "O": 1})
            self.assertAlmostEqual(result["reference_energy_eV"], -11.0)
            self.assertAlmostEqual(result["excess_energy_eV"], -9.0)
            self.assertAlmostEqual(result["surface_excess_energy_eV_per_surface"], -4.5)
            self.assertEqual(result["task_number"], "1")
            self.assertEqual(result["task_suffix"], "top")

    def test_surface_batch_keeps_same_task_number_variants_separate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_dir = os.path.join(tmpdir, "simple_substance_database")
            cu_dir = os.path.join(db_dir, "Cu")
            o_dir = os.path.join(db_dir, "O")
            root_dir = os.path.join(tmpdir, "Surface_energy")
            top_dir = os.path.join(root_dir, "task.1_top")
            bridge_dir = os.path.join(root_dir, "task.1_bridge")
            for path in [cu_dir, o_dir, top_dir, bridge_dir]:
                os.makedirs(path)

            self._write_poscar(os.path.join(cu_dir, "POSCAR"), ["Cu"], [1])
            self._write_outcar(os.path.join(cu_dir, "OUTCAR"), -3.0)
            self._write_poscar(os.path.join(o_dir, "POSCAR"), ["O"], [1])
            self._write_outcar(os.path.join(o_dir, "OUTCAR"), -5.0)

            for slab_dir, energy in [(top_dir, -20.0), (bridge_dir, -19.0)]:
                self._write_poscar(os.path.join(slab_dir, "POSCAR"), ["Cu", "O"], [2, 1])
                self._write_outcar(os.path.join(slab_dir, "OUTCAR"), energy)

            results = calc_surface_excess_energies_batch(
                root_dir,
                simple_substance_db=db_dir,
                n_surfaces=2,
            )

            self.assertEqual(len(results), 2)
            self.assertEqual({row["task_number"] for row in results}, {"1"})
            self.assertEqual({row["task_suffix"] for row in results}, {"top", "bridge"})
            self.assertEqual([row["status"] for row in results], ["ok", "ok"])


if __name__ == "__main__":
    unittest.main()
