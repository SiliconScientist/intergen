import unittest

from intergen.__main__ import get_initial_atoms_list


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


if __name__ == "__main__":
    unittest.main()
