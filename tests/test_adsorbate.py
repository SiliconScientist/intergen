import unittest

from ase.build import fcc111
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from intergen.adsorbate import get_adsorbate_comparison_indices
from intergen.config import Config
from intergen.surface import find_unique_structures, get_substructure


class TestConfig(unittest.TestCase):
    def test_adsorbate_surface_layers_for_matching_is_parsed(self):
        cfg = Config(
            structure={
                "hcp_list": [],
                "fcc_list": ["Pt"],
                "size": (3, 3, 4),
                "vacuum": 10.0,
            },
            generation={
                "layers_to_swap": 1,
                "num_swaps": 1,
                "swap_elements": ["Cu"],
                "only_last_generation": True,
            },
            database={"path": "data/test.db"},
            adsorbate={
                "matcher": {"ltol": 0.05, "stol": 0.1, "angle_tol": 5.0},
                "species": "N",
                "coords": [(0.0, 0.0, 0.0)],
                "sites": ["hollow"],
                "tag": 0,
                "surface_layers_for_matching": 2,
            },
        )

        self.assertEqual(cfg.adsorbate.surface_layers_for_matching, 2)


class TestHollowSiteRegistry(unittest.TestCase):
    def setUp(self):
        atoms = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)[::-1]
        atoms.set_pbc(True)
        self.slab = AseAtomsAdaptor().get_structure(atoms)
        self.site_finder = AdsorbateSiteFinder(self.slab)
        self.adsorbate = Molecule(["N"], [[0.0, 0.0, 0.0]])
        self.matcher = StructureMatcher(ltol=0.05, stol=0.1, angle_tol=5.0)
        self.atoms_per_layer = 9
        self.adsorbate_index = [len(self.slab)]

    def _comparison_substructures(self, structures, surface_layers):
        comparison_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=self.atoms_per_layer,
            surface_layers=surface_layers,
            adsorbate_indices=self.adsorbate_index,
        )
        return [
            get_substructure(structure, indices=comparison_indices)
            for structure in structures
        ]

    def test_get_adsorbate_comparison_indices_uses_requested_surface_layers(self):
        comparison_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=9,
            surface_layers=2,
            adsorbate_indices=[36, 37],
        )

        self.assertEqual(comparison_indices, list(range(18)) + [36])

    def test_get_adsorbate_comparison_indices_can_include_multiple_adsorbate_atoms(self):
        comparison_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=9,
            surface_layers=2,
            adsorbate_indices=[36, 37],
            adsorbate_atoms_for_matching=2,
        )

        self.assertEqual(comparison_indices, list(range(18)) + [36, 37])

    def test_hollow_site_matching_distinguishes_fcc_and_hcp_with_second_layer(self):
        hollow_sites = self.site_finder.find_adsorption_sites(symm_reduce=False)[
            "hollow"
        ]
        hollow_structures = [
            self.site_finder.add_adsorbate(self.adsorbate, coords)
            for coords in hollow_sites
        ]

        two_layer_substructures = self._comparison_substructures(
            hollow_structures, surface_layers=2
        )
        representative_indices = find_unique_structures(
            two_layer_substructures, matcher=self.matcher
        )
        representative_structures = [
            hollow_structures[index] for index in representative_indices
        ]

        self.assertEqual(len(representative_structures), 2)

        one_layer_unique = find_unique_structures(
            self._comparison_substructures(
                representative_structures, surface_layers=1
            ),
            matcher=self.matcher,
        )
        two_layer_unique = find_unique_structures(
            self._comparison_substructures(
                representative_structures, surface_layers=2
            ),
            matcher=self.matcher,
        )

        self.assertEqual(one_layer_unique, [0])
        self.assertEqual(two_layer_unique, [0, 1])


if __name__ == "__main__":
    unittest.main()
