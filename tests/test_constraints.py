import unittest

from ase import Atoms
from ase.build import fcc111
from ase.constraints import FixAtoms

from intergen.constraints import constrain_adsorbate_bottom_layers, get_layer_indices


class TestConstraints(unittest.TestCase):
    def test_get_layer_indices_finds_four_fcc111_layers(self):
        slab = fcc111('Pt', size=(3, 3, 4), vacuum=10.0, a=3.92)[::-1]

        layer_indices = get_layer_indices(slab)

        self.assertEqual(len(layer_indices), 4)
        self.assertTrue(all(len(indices) == 9 for indices in layer_indices.values()))

    def test_constrain_adsorbate_bottom_layers_fixes_bottom_two_layers_only(self):
        slab = fcc111('Pt', size=(3, 3, 4), vacuum=10.0, a=3.92)[::-1]
        adslab = slab.copy()
        adslab += Atoms(
            symbols=['N'],
            positions=[(adslab.cell[0, 0] / 2.0, adslab.cell[1, 1] / 2.0, 16.0)],
            cell=adslab.cell,
            pbc=adslab.pbc,
        )

        constrained = constrain_adsorbate_bottom_layers(
            adslab,
            adsorbate_len=1,
            bottom_layers=2,
        )

        self.assertEqual(len(constrained.constraints), 1)
        self.assertIsInstance(constrained.constraints[0], FixAtoms)
        self.assertEqual(len(constrained.constraints[0].index.tolist()), 18)
        self.assertNotIn(len(adslab) - 1, constrained.constraints[0].index.tolist())


if __name__ == '__main__':
    unittest.main()
