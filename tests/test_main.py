import unittest

from types import SimpleNamespace

from ase import Atoms
from ase.constraints import FixAtoms

from intergen.__main__ import apply_database_constraints, get_initial_atoms_list


class TestMain(unittest.TestCase):
    def test_get_initial_atoms_list_includes_pure_atoms_by_default(self):
        pure_atoms = ["slab-a", "slab-b"]

        atoms_list = get_initial_atoms_list(
            pure_atoms=pure_atoms,
            only_last_generation=False,
        )

        self.assertEqual(atoms_list, pure_atoms)
        self.assertIsNot(atoms_list, pure_atoms)

    def test_get_initial_atoms_list_skips_pure_atoms_for_last_generation_only(self):
        pure_atoms = ["slab-a", "slab-b"]

        atoms_list = get_initial_atoms_list(
            pure_atoms=pure_atoms,
            only_last_generation=True,
        )

        self.assertEqual(atoms_list, [])


    def test_apply_database_constraints_is_noop_when_disabled(self):
        atoms = Atoms(
            symbols=['Pt', 'Pt', 'N'],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.5, 1.2)],
            cell=[(5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, True),
        )
        cfg = SimpleNamespace(
            database=SimpleNamespace(
                constrain_bottom_layers=0,
                constraint_z_tolerance=0.5,
                constraint_lowest_z_tolerance=0.5,
            ),
            adsorbate=SimpleNamespace(coords=[(0.0, 0.0, 0.0)]),
        )

        constrained = apply_database_constraints(cfg=cfg, atoms_list=[atoms])

        self.assertEqual(len(constrained), 1)
        self.assertEqual(constrained[0].constraints, [])

    def test_apply_database_constraints_adds_fixatoms_when_enabled(self):
        slab_positions = []
        for z in (0.0, 2.0, 4.0):
            for x in (0.0, 1.5):
                for y in (0.0, 1.5):
                    slab_positions.append((x, y, z))
        atoms = Atoms(
            symbols=['Pt'] * len(slab_positions) + ['N'],
            positions=slab_positions + [(0.75, 0.75, 5.5)],
            cell=[(6.0, 0.0, 0.0), (0.0, 6.0, 0.0), (0.0, 0.0, 18.0)],
            pbc=(True, True, True),
        )
        cfg = SimpleNamespace(
            database=SimpleNamespace(
                constrain_bottom_layers=2,
                constraint_z_tolerance=0.5,
                constraint_lowest_z_tolerance=0.5,
            ),
            adsorbate=SimpleNamespace(coords=[(0.0, 0.0, 0.0)]),
        )

        constrained = apply_database_constraints(cfg=cfg, atoms_list=[atoms])[0]

        self.assertEqual(len(constrained.constraints), 1)
        self.assertIsInstance(constrained.constraints[0], FixAtoms)
        self.assertEqual(sorted(constrained.constraints[0].index.tolist()), list(range(8)))


if __name__ == "__main__":
    unittest.main()
